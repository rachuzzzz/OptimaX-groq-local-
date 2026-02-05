"""
Dynamic Join Path Inference (DJPI) v3 - Database-Agnostic Join Discovery
=========================================================================

==========================================================================
v5.0 ARCHITECTURAL NOTE - ADVISORY MODE ONLY
==========================================================================
As of OptimaX v5.0, DJPI operates in ADVISORY MODE ONLY:

- DJPI does NOT inject guidance into LLM prompts
- DJPI does NOT influence SQL generation
- SQL generation is now handled by LlamaIndex's NLSQLTableQueryEngine

DJPI is retained for:
[OK] Join path explanation / explainability
[OK] Single-table vs multi-table detection (logged for debugging)
[OK] Interpretability and debugging
[OK] Future analytics and governance insights

DJPI failures are completely NON-BLOCKING - they never affect SQL execution.
==========================================================================

Purpose:
- Provides join path analysis for explainability and debugging
- Identifies single-table vs multi-table queries
- Logs relationship discovery for interpretability

Design Principles:
1. Database Agnostic: Uses runtime schema introspection only
2. Deterministic: Rule-based column/type matching for join paths
3. Advisory Only (v5.0): Output is logged, not injected
4. Non-Blocking: Failures never affect query execution
5. Conservative: Minimal overhead, optional execution

DJPI v3 Features (now advisory):
[OK] Acyclic path enforcement (no table visited twice)
[OK] Strengthened join scoring with timestamp penalties
[OK] Cost-aware path selection (max depth: 4 hops)
[OK] Enhanced debug logging for rejected paths
[OK] Improved guidance format with explicit constraints

Author: OptimaX Team
Version: 3.0 (DJPI v3) - Advisory Mode in v5.0
"""

from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import heapq
import logging
import json

logger = logging.getLogger(__name__)


class SchemaGraph:
    """
    Represents database schema as a graph for join path discovery.

    Nodes: Tables
    Edges: Inferred foreign key relationships based on:
        - Column name matching (e.g., account_id in both tables)
        - Type compatibility
        - Naming conventions (*_id, *_code, etc.)
    """

    def __init__(self):
        # Adjacency list: {table_name: [(related_table, join_column, related_column, score)]}
        # Score represents join strength: higher = more semantically meaningful
        self.graph: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)
        self.tables: Set[str] = set()
        self.schema_cache: Optional[Dict] = None

    def build_from_schema(self, schema_dict: Dict[str, List[Dict]]) -> None:
        """
        Build schema graph from runtime schema introspection.

        Args:
            schema_dict: {table_name: [{"name": col_name, "type": col_type}, ...]}

        Infers relationships using:
        - Exact column name matches with compatible types
        - Foreign key naming patterns (*_id, *_code)
        """
        self.schema_cache = schema_dict
        self.tables = set(schema_dict.keys())

        logger.info(f"Building schema graph for {len(self.tables)} tables...")

        # For each table pair, check if columns suggest a relationship
        tables_list = list(self.tables)
        relationship_count = 0

        for i, table_a in enumerate(tables_list):
            cols_a = {col["name"]: col["type"] for col in schema_dict[table_a]}

            for table_b in tables_list[i+1:]:  # Avoid duplicate checks
                cols_b = {col["name"]: col["type"] for col in schema_dict[table_b]}

                # Find matching columns that could indicate a join
                relationships = self._infer_relationships(
                    table_a, cols_a,
                    table_b, cols_b
                )

                for col_a, col_b in relationships:
                    # Calculate join strength score
                    score = self._calculate_join_score(
                        table_a, col_a, cols_a[col_a],
                        table_b, col_b, cols_b[col_b]
                    )

                    # Add bidirectional edge with score
                    self.graph[table_a].append((table_b, col_a, col_b, score))
                    self.graph[table_b].append((table_a, col_b, col_a, score))
                    relationship_count += 1

                    # DJPI v3 DEBUG: Log all discovered edges with scores
                    logger.info(f"  Edge: {table_a}.{col_a} <-> {table_b}.{col_b} (score: {score:.2f})")

        logger.info(f"[OK] Schema graph built: {relationship_count} relationships discovered")

    def _infer_relationships(
        self,
        table_a: str,
        cols_a: Dict[str, str],
        table_b: str,
        cols_b: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Infer relationships between two tables based on column analysis.

        Returns:
            List of (col_a, col_b) tuples representing potential joins
        """
        relationships = []

        # Strategy 1: Exact column name match with type compatibility
        for col_a, type_a in cols_a.items():
            # FIX #2: Hard-ban timestamp columns - skip entirely (not just penalize)
            if self._is_timestamp_column(col_a, type_a):
                continue  # Do not consider this column for joins at all

            if col_a in cols_b:
                type_b = cols_b[col_a]
                # FIX #2: Also check target column
                if self._is_timestamp_column(col_a, type_b):
                    continue
                if self._types_compatible(type_a, type_b):
                    # Same column name + compatible types = likely join key
                    relationships.append((col_a, col_a))

        # Strategy 2: Foreign key naming convention
        # Example: frequent_flyer_id in account -> frequent_flyer.frequent_flyer_id
        for col_a, type_a in cols_a.items():
            if self._is_id_column(col_a):
                # Check if col_a references table_b
                if self._column_references_table(col_a, table_b):
                    # Find primary key in table_b
                    pk_b = self._find_primary_key(table_b, cols_b)
                    if pk_b and self._types_compatible(type_a, cols_b[pk_b]):
                        relationships.append((col_a, pk_b))

        # Strategy 3: Reverse check (table_b columns referencing table_a)
        for col_b, type_b in cols_b.items():
            if self._is_id_column(col_b):
                if self._column_references_table(col_b, table_a):
                    pk_a = self._find_primary_key(table_a, cols_a)
                    if pk_a and self._types_compatible(type_b, cols_a[pk_a]):
                        relationships.append((pk_a, col_b))

        return relationships

    def _types_compatible(self, type_a: str, type_b: str) -> bool:
        """Check if two SQL types are compatible for joining."""
        # Normalize types (e.g., INTEGER, BIGINT both count as int)
        type_a_norm = type_a.upper()
        type_b_norm = type_b.upper()

        # Integer types
        int_types = {"INTEGER", "BIGINT", "SMALLINT", "INT", "SERIAL", "BIGSERIAL"}
        if type_a_norm in int_types and type_b_norm in int_types:
            return True

        # String types
        str_types = {"TEXT", "VARCHAR", "CHAR", "CHARACTER"}
        if any(t in type_a_norm for t in str_types) and any(t in type_b_norm for t in str_types):
            return True

        # Exact match
        return type_a_norm == type_b_norm

    def _is_id_column(self, col_name: str) -> bool:
        """Check if column name suggests it's an ID/foreign key."""
        col_lower = col_name.lower()
        return col_lower.endswith("_id") or col_lower.endswith("_code")

    def _is_timestamp_column(self, col_name: str, col_type: str) -> bool:
        """
        FIX #2: Hard-ban timestamp-based joins.

        Completely exclude columns that are timestamp/date-related from join consideration.
        These columns almost never represent valid foreign key relationships.

        Returns:
            True if column should be EXCLUDED from join inference
        """
        col_lower = col_name.lower()
        type_lower = col_type.lower()

        # Hard-ban patterns for column names
        timestamp_name_patterns = [
            "_ts",           # update_ts, create_ts
            "timestamp",     # created_timestamp
            "created_at",    # Rails convention
            "updated_at",    # Rails convention
            "modified_at",
            "deleted_at",
            "_date",         # flight_date, booking_date (unless it's an ID)
            "_time",         # departure_time, arrival_time
        ]

        # Check if column name contains any timestamp pattern
        for pattern in timestamp_name_patterns:
            if pattern in col_lower:
                # Exception: if it ends with _id, it's probably a foreign key
                if col_lower.endswith("_id"):
                    return False
                return True

        # Hard-ban based on data types (date/time types that aren't explicit FKs)
        timestamp_types = ["timestamp", "datetime", "date", "time"]
        for ts_type in timestamp_types:
            if ts_type in type_lower:
                # Only ban if column name doesn't suggest it's an ID
                if not col_lower.endswith("_id") and not col_lower.endswith("_code"):
                    return True

        return False

    def _column_references_table(self, col_name: str, table_name: str) -> bool:
        """
        Check if column name suggests it references a table.
        Example: frequent_flyer_id likely references frequent_flyer table
        """
        col_lower = col_name.lower()
        table_lower = table_name.lower()

        # Remove schema prefix if present
        if "." in table_lower:
            table_lower = table_lower.split(".")[-1]

        # Check if table name is in column name
        # frequent_flyer_id contains "frequent_flyer"
        return table_lower in col_lower

    def _find_primary_key(self, table_name: str, columns: Dict[str, str]) -> Optional[str]:
        """
        Heuristically find the primary key column.
        Convention: {table_name}_id or just 'id'
        """
        table_lower = table_name.lower()
        if "." in table_lower:
            table_lower = table_lower.split(".")[-1]

        # Try {table_name}_id first
        pk_candidate = f"{table_lower}_id"
        for col in columns.keys():
            if col.lower() == pk_candidate:
                return col

        # Try just "id"
        for col in columns.keys():
            if col.lower() == "id":
                return col

        # Try first column ending with _id
        for col in columns.keys():
            if col.lower().endswith("_id"):
                return col

        return None

    def _calculate_join_score(
        self,
        table_a: str,
        col_a: str,
        type_a: str,
        table_b: str,
        col_b: str,
        type_b: str
    ) -> float:
        """
        Calculate join strength score for a relationship.

        Scoring Logic (database-agnostic, heuristic-based):
        - STRONG (100+ points): Identity-based joins via *_id columns
        - MEDIUM (50-80 points): Exact column matches with good types
        - WEAK (10-40 points): Descriptive attribute matches (email, name, etc.)

        Higher scores = semantically stronger, more reliable joins

        Args:
            table_a, col_a, type_a: First table, column, type
            table_b, col_b, type_b: Second table, column, type

        Returns:
            Float score (higher is better)
        """
        score = 0.0

        # === STRONG SIGNALS (Identity-based joins) ===

        # Signal 1: Column ends with _id (primary/foreign key indicator)
        if col_a.lower().endswith("_id") or col_b.lower().endswith("_id"):
            score += 100.0

        # Signal 2: Column references the other table by name
        # Example: frequent_flyer_id in account table references frequent_flyer
        table_a_clean = table_a.split(".")[-1].lower()  # Remove schema prefix
        table_b_clean = table_b.split(".")[-1].lower()

        if table_b_clean in col_a.lower() or table_a_clean in col_b.lower():
            score += 50.0  # Strong FK naming convention

        # Signal 3: Integer-like types (typical for IDs)
        int_types = {"INTEGER", "BIGINT", "SMALLINT", "INT", "SERIAL", "BIGSERIAL"}
        type_a_upper = type_a.upper()
        type_b_upper = type_b.upper()

        if type_a_upper in int_types and type_b_upper in int_types:
            score += 30.0  # Integer join = likely ID-based

        # Signal 4: UUID types (strong identifier)
        if "UUID" in type_a_upper or "UUID" in type_b_upper:
            score += 40.0

        # === MEDIUM SIGNALS (Type compatibility) ===

        # Signal 5: Exact column name match
        if col_a == col_b:
            score += 40.0

        # Signal 6: Compatible types (already checked before this, but boost)
        if self._types_compatible(type_a, type_b):
            score += 20.0

        # === WEAK SIGNALS (Penalties for descriptive attributes) ===

        # ✅ DJPI v3 FIX #2a: Penalty for descriptive text fields (weak join candidates)
        weak_columns = {
            "email", "name", "first_name", "last_name", "phone",
            "status", "description", "title", "address", "city",
            "state", "country", "zip", "comment", "notes"
        }

        col_a_lower = col_a.lower()
        col_b_lower = col_b.lower()

        if col_a_lower in weak_columns or col_b_lower in weak_columns:
            score -= 60.0  # Heavy penalty for attribute-based joins

        # ✅ DJPI v3 FIX #2b: Penalty for timestamp/date columns (very weak join candidates)
        timestamp_indicators = {
            "timestamp", "created", "updated", "modified", "deleted",
            "update_ts", "create_ts", "time", "date", "_at", "_on"
        }

        # Check if column name contains timestamp indicators
        is_timestamp_a = any(indicator in col_a_lower for indicator in timestamp_indicators)
        is_timestamp_b = any(indicator in col_b_lower for indicator in timestamp_indicators)

        if is_timestamp_a or is_timestamp_b:
            score -= 80.0  # Very heavy penalty for timestamp joins (almost never correct)

        # ✅ DJPI v3 FIX #2c: Penalty for TEXT/VARCHAR without _id suffix (likely descriptive)
        str_types = {"TEXT", "VARCHAR", "CHAR"}
        is_text_a = any(t in type_a_upper for t in str_types)
        is_text_b = any(t in type_b_upper for t in str_types)

        if (is_text_a or is_text_b) and not (col_a_lower.endswith("_id") or col_b_lower.endswith("_id")):
            score -= 30.0  # Penalize text joins unless they're ID columns

        # Ensure minimum score (never negative, allows weak joins as fallback)
        score = max(score, 10.0)

        return score

    def find_join_path(
        self,
        source_table: str,
        target_table: str,
        max_depth: int = 4
    ) -> Optional[List[Tuple[str, str, str, str]]]:
        """
        Find highest-scoring join path between two tables.

        Uses a weighted path algorithm (modified Dijkstra) that maximizes
        cumulative join strength scores instead of minimizing hops.

        DJPI v3 GUARANTEES:
        - Acyclic paths only (no table visited twice)
        - Maximum depth of 4 hops (HARD limit)
        - Minimum edge score threshold (rejects weak joins if alternatives exist)
        - Cost-aware optimization (prefers fewer hops when scores are close)

        Args:
            source_table: Starting table (e.g., 'frequent_flyer')
            target_table: Destination table (e.g., 'flight')
            max_depth: Maximum join hops allowed (default: 4, HARD cap)

        Returns:
            List of join steps: [(from_table, from_col, to_table, to_col), ...]
            None if no path exists
        """
        if source_table not in self.tables or target_table not in self.tables:
            logger.warning(f"Table not found: {source_table} or {target_table}")
            return None

        if source_table == target_table:
            return []  # Same table, no join needed

        # Minimum edge score threshold - reject very weak joins
        # Unless no other path exists, edges scoring below this are ignored
        MIN_EDGE_SCORE = 0.0

        # Priority queue: (-total_score, hop_count, current_table, path, visited_tables)
        # Negative score because heapq is a min-heap (we want max score)
        # visited_tables: set of tables already in this path (for cycle prevention)
        pq = [(-0.0, 0, source_table, [], {source_table})]

        # Track best score to each table (for pruning)
        best_scores: Dict[str, float] = {source_table: 0.0}

        # Store all valid paths to target (for debugging)
        all_paths_to_target = []
        rejected_paths = []  # Track why paths were rejected (for debug logging)

        while pq:
            neg_score, hop_count, current_table, path, visited_tables = heapq.heappop(pq)
            current_score = -neg_score  # Convert back to positive

            # Check depth limit (HARD cap at 4)
            if hop_count >= max_depth:
                continue

            # If we've already found a better path to this table, skip
            if current_table in best_scores and current_score < best_scores[current_table]:
                continue

            # Explore neighbors
            for next_table, join_col, related_col, edge_score in self.graph[current_table]:
                # ✅ DJPI v3 FIX #1: Enforce acyclic paths (CRITICAL)
                # Reject any path that revisits a table
                if next_table in visited_tables:
                    rejected_paths.append((
                        path + [(current_table, join_col, next_table, related_col)],
                        f"Cycle detected: {next_table} already in path"
                    ))
                    continue

                # ✅ DJPI v3 FIX #2: Minimum edge score threshold
                # Reject very weak joins (but allow if no alternatives exist)
                if edge_score < MIN_EDGE_SCORE:
                    rejected_paths.append((
                        path + [(current_table, join_col, next_table, related_col)],
                        f"Edge score too low: {edge_score:.1f} < {MIN_EDGE_SCORE}"
                    ))
                    continue

                # Calculate new path score
                new_score = current_score + edge_score
                new_hop_count = hop_count + 1

                # Build new path with updated visited set
                new_path = path + [(current_table, join_col, next_table, related_col)]
                new_visited = visited_tables | {next_table}  # Add next_table to visited set

                # Found target?
                if next_table == target_table:
                    all_paths_to_target.append((new_score, new_hop_count, new_path))

                    # Continue exploring to find ALL paths within max_depth
                    # (we'll select best one at the end)
                    continue

                # Only explore if this is a better path to next_table
                if next_table not in best_scores or new_score > best_scores[next_table]:
                    best_scores[next_table] = new_score
                    heapq.heappush(pq, (-new_score, new_hop_count, next_table, new_path, new_visited))

        # ✅ DJPI v3 FIX #3: Cost-aware path selection
        # Select best path from all candidates
        if not all_paths_to_target:
            logger.warning(f"No join path found: {source_table} -> {target_table}")

            # ✅ DJPI v3 FIX #4: Enhanced debug logging
            if rejected_paths:
                logger.debug(f"  All paths were rejected:")
                rejection_reasons = {}
                for _, reason in rejected_paths[:10]:  # Show first 10 rejections
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                for reason, count in rejection_reasons.items():
                    logger.debug(f"    - {reason} ({count} path(s))")

            return None

        # Sort by:
        # 1. Primary: Highest cumulative score
        # 2. Secondary: Fewest hops (when scores are close)
        # This implements cost-aware optimization
        all_paths_to_target.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        best_score, best_hop_count, best_path = all_paths_to_target[0]

        # ✅ DJPI v3 FIX #4: Enhanced debug logging
        if len(all_paths_to_target) > 1:
            logger.info(f"[OK] Join path selected: {source_table} -> {target_table} (score: {best_score:.1f}, {best_hop_count} hops)")
            logger.debug(f"  Selected path details:")
            for i, (from_t, from_c, to_t, to_c) in enumerate(best_path, 1):
                logger.debug(f"    {i}. {from_t}.{from_c} -> {to_t}.{to_c}")

            # Log rejected alternatives
            logger.debug(f"  Considered {len(all_paths_to_target)} alternative path(s):")
            for score, hops, alt_path in all_paths_to_target[1:4]:  # Show top 3 alternatives
                path_str = " -> ".join([alt_path[0][0]] + [step[2] for step in alt_path])
                logger.debug(f"    Score {score:.1f} ({hops} hops): {path_str}")

            # Show cycle rejections if any
            cycle_rejections = [r for r in rejected_paths if "Cycle" in r[1]]
            if cycle_rejections:
                logger.debug(f"  Rejected {len(cycle_rejections)} path(s) due to cycles")
        else:
            logger.info(f"[OK] Join path found: {source_table} -> {target_table} (score: {best_score:.1f}, {best_hop_count} hops)")

        return best_path


def format_join_guidance(join_path: List[Tuple[str, str, str, str]]) -> str:
    """
    Format join path into LLM-friendly guidance message.

    DJPI v3: Emphasizes CONSTRAINTS, not just join chains.
    This is NOT SQL injection - it's high-level relational guidance.
    The LLM still generates the actual SQL.

    Args:
        join_path: [(from_table, from_col, to_table, to_col), ...]

    Returns:
        Guidance message for the agent
    """
    if not join_path:
        return ""

    # Build human-readable join chain
    tables_involved = [join_path[0][0]]  # Start with first table
    for _, _, to_table, _ in join_path:
        tables_involved.append(to_table)

    # ✅ DJPI v3 FIX #5: Improved guidance format with constraints
    guidance = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RELATIONAL GUIDANCE (DJPI v3 - Schema-Discovered)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED JOIN PATH:
{' -> '.join(tables_involved)}

JOIN SPECIFICATIONS (use these EXACT relationships):
"""

    for i, (from_table, from_col, to_table, to_col) in enumerate(join_path, 1):
        guidance += f"{i}. {from_table}.{from_col} = {to_table}.{to_col}\n"

    guidance += f"""
CRITICAL CONSTRAINTS:
[OK] Use ONLY identity-based joins ({len(join_path)} join{'s' if len(join_path) != 1 else ''} required)
[OK] Do NOT repeat tables in your JOIN chain (acyclic path enforced)
[OK] Do NOT use attribute joins (email, name, etc.) or timestamp joins
[OK] Follow the exact path above - shortest valid path with highest confidence

REMINDER:
- Use schema prefix if required (check schema section in your system prompt)
- This is the OPTIMAL path determined by database schema analysis
- Alternative join paths were rejected due to cycles, weak joins, or excessive depth

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return guidance


def identify_tables_for_query(llm, user_query: str, available_tables: List[str]) -> Dict[str, any]:
    """
    Use LLM to autonomously identify which tables are involved in the query.

    This is the AUTONOMOUS component - LLM reasons about table selection.

    Args:
        llm: Groq LLM instance
        user_query: User's natural language question
        available_tables: List of all tables in the schema

    Returns:
        Dict with:
        - primary_table: Main entity table (e.g., 'frequent_flyer' for "most frequent flyers")
        - metric_table: Table containing the metric to aggregate (e.g., 'booking' for count)
        - needs_join: Boolean indicating if tables are different
    """
    prompt = f"""You are analyzing a database query to identify which tables are involved.

USER QUERY: "{user_query}"

AVAILABLE TABLES:
{chr(10).join(f"- {table}" for table in available_tables)}

TASK:
Identify the tables needed for this query:
1. PRIMARY TABLE: The main entity the user is asking about (e.g., for "most frequent flyers", it's the frequent_flyer table)
2. METRIC TABLE: The table containing events/data to count/aggregate (e.g., flights, bookings)

If the query can be answered from a single table, both should be the same.

RESPONSE FORMAT (JSON only):
{{
    "primary_table": "full_table_name_from_list_above",
    "metric_table": "full_table_name_from_list_above",
    "needs_join": true/false,
    "reasoning": "Brief explanation of why these tables"
}}

Respond with ONLY the JSON object, nothing else."""

    try:
        response = llm.complete(prompt)
        response_text = str(response).strip()

        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        # Validate tables exist
        if result["primary_table"] not in available_tables:
            logger.warning(f"LLM suggested invalid primary_table: {result['primary_table']}")
            return {"needs_join": False}

        if result["metric_table"] not in available_tables:
            logger.warning(f"LLM suggested invalid metric_table: {result['metric_table']}")
            return {"needs_join": False}

        logger.info(f"[OK] LLM identified tables: {result['primary_table']} -> {result['metric_table']}")
        logger.info(f"  Reasoning: {result.get('reasoning', 'N/A')}")

        return result

    except Exception as e:
        logger.error(f"Table identification failed: {str(e)}")
        return {"needs_join": False}  # Fallback: let agent handle it normally
