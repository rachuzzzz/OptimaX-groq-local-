"""
RelationalCorrector - FK-Based SQL Correction

Corrects structurally invalid SQL using declared foreign key metadata.

BEHAVIOR:
- Detects column references in wrong tables
- If EXACTLY ONE FK path exists -> rewrite SQL with JOIN
- If MULTIPLE FK paths exist -> return structured ambiguity
- If NO FK path exists -> pass through to validators

Uses ONLY declared FK metadata. Deterministic. No LLM.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ForeignKeyInfo:
    """
    Represents a foreign key relationship.

    Attributes:
        source_table: Table containing the FK column
        source_column: FK column name in source table
        target_table: Referenced table
        target_column: Referenced column (usually PK)
    """
    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass
class SchemaMetadata:
    """
    Schema metadata required for relational correction.

    Attributes:
        tables: Dict mapping table names to column sets
        foreign_keys: List of all FK relationships
    """
    tables: Dict[str, Set[str]]  # {table_name: {col1, col2, ...}}
    foreign_keys: List[ForeignKeyInfo]

    def get_table_columns(self, table_name: str) -> Set[str]:
        """Get columns for a table (case-insensitive lookup)."""
        table_lower = table_name.lower()
        for t, cols in self.tables.items():
            if t.lower() == table_lower or t.lower().endswith(f".{table_lower}"):
                return cols
        return set()

    def column_exists_in_table(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table (case-insensitive)."""
        columns = self.get_table_columns(table_name)
        return column_name.lower() in {c.lower() for c in columns}

    def find_fks_from_table(self, table_name: str) -> List[ForeignKeyInfo]:
        """Find all FKs originating from a table."""
        table_lower = table_name.lower()
        result = []
        for fk in self.foreign_keys:
            src_lower = fk.source_table.lower()
            if src_lower == table_lower or src_lower.endswith(f".{table_lower}"):
                result.append(fk)
        return result


@dataclass
class AmbiguityOption:
    """
    Represents one option for resolving an ambiguous FK path.

    Attributes:
        fk_column: The foreign key column name (e.g., "arrival_airport")
        target_table: The table being joined to
        target_column: The column in the target table
        description: Human-readable description derived from FK name
    """
    fk_column: str
    target_table: str
    target_column: str
    description: str


@dataclass
class RelationalAmbiguity:
    """
    Structured representation of a relational ambiguity.

    This is emitted when multiple FK paths exist and the system
    cannot deterministically choose one.

    CRITICAL: This is NOT an error - it's a request for clarification.

    Attributes:
        type: Always "RELATIONAL_AMBIGUITY"
        source_table: The table in the FROM clause
        target_table: The table that would be joined
        column: The column that triggered the ambiguity
        options: List of valid FK paths to choose from
    """
    type: str = "RELATIONAL_AMBIGUITY"
    source_table: str = ""
    target_table: str = ""
    column: str = ""
    options: List[AmbiguityOption] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "source_table": self.source_table,
            "target_table": self.target_table,
            "column": self.column,
            "options": [
                {
                    "fk_column": opt.fk_column,
                    "target_table": opt.target_table,
                    "target_column": opt.target_column,
                    "description": opt.description,
                }
                for opt in self.options
            ]
        }


@dataclass
class CorrectionResult:
    """
    Result of relational correction attempt.

    Attributes:
        success: Whether correction was successful
        corrected_sql: The corrected SQL (if success)
        original_sql: The original input SQL
        applied_fixes: List of fixes applied
        error: Error information (if hard failure)
        ambiguity: Structured ambiguity (if clarification needed)
    """
    success: bool
    corrected_sql: Optional[str]
    original_sql: str
    applied_fixes: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    ambiguity: Optional[RelationalAmbiguity] = None

    def needs_clarification(self) -> bool:
        """Check if this result requires user clarification."""
        return self.ambiguity is not None


# =============================================================================
# SQL PARSING UTILITIES
# =============================================================================

class SQLParser:
    """
    Minimal SQL parser for extracting structural information.

    CRITICAL: This parser handles BOTH:
    - Qualified column references (table.column)
    - Unqualified column references (column) in SELECT, WHERE, GROUP BY, ORDER BY
    """

    # SQL keywords and functions to exclude from column detection
    SQL_KEYWORDS = {
        # Core SQL keywords
        'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not',
        'as', 'left', 'right', 'inner', 'outer', 'full', 'cross', 'natural',
        'group', 'order', 'by', 'having', 'limit', 'offset', 'union',
        'distinct', 'all', 'asc', 'desc', 'nulls', 'first', 'last',
        'null', 'is', 'in', 'between', 'like', 'ilike', 'exists',
        'case', 'when', 'then', 'else', 'end', 'true', 'false',
        # Aggregate functions
        'count', 'sum', 'avg', 'min', 'max', 'array_agg', 'string_agg',
        # Other functions
        'extract', 'epoch', 'cast', 'coalesce', 'nullif', 'greatest', 'least',
        'concat', 'substring', 'trim', 'upper', 'lower', 'length',
        'abs', 'ceil', 'floor', 'round', 'trunc',
        'now', 'current_date', 'current_time', 'current_timestamp',
        'date', 'time', 'timestamp', 'interval',
        'year', 'month', 'day', 'hour', 'minute', 'second',
    }

    @staticmethod
    def normalize_sql(sql: str) -> str:
        """Normalize SQL for parsing (remove comments, normalize whitespace)."""
        sql = re.sub(r'--.*?$', ' ', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        sql = re.sub(r'\s+', ' ', sql).strip()
        return sql

    @staticmethod
    def extract_tables_with_aliases(sql: str) -> Tuple[Dict[str, str], str]:
        """
        Extract table references with their aliases.

        Returns:
            Tuple of:
            - Dict mapping alias/name -> full_table_name
            - Primary FROM table name (for unqualified column resolution)
        """
        sql = SQLParser.normalize_sql(sql)
        tables = {}
        primary_table = None

        # Pattern: FROM/JOIN [schema.]table [AS] [alias]
        pattern = re.compile(
            r'(?:FROM|JOIN)\s+'
            r'(?:([A-Za-z_][A-Za-z0-9_]*)\.)?'  # Optional schema
            r'([A-Za-z_][A-Za-z0-9_]*)'          # Table name
            r'(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?',  # Optional alias
            re.IGNORECASE
        )

        is_first = True
        for match in pattern.finditer(sql):
            schema = match.group(1)
            table = match.group(2)
            alias = match.group(3)

            # Skip SQL keywords captured as aliases
            if alias and alias.upper() in (
                'ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT', 'INNER',
                'OUTER', 'CROSS', 'FULL', 'JOIN', 'GROUP', 'ORDER',
                'HAVING', 'LIMIT', 'OFFSET', 'AS', 'NATURAL', 'SET'
            ):
                alias = None

            full_name = f"{schema}.{table}" if schema else table

            # First table is the primary FROM table
            if is_first:
                primary_table = full_name
                is_first = False

            if alias:
                tables[alias.lower()] = full_name.lower()
            tables[table.lower()] = full_name.lower()

        return tables, primary_table.lower() if primary_table else None

    @staticmethod
    def extract_unqualified_columns(sql: str, table_aliases: Dict[str, str]) -> List[str]:
        """
        Extract UNQUALIFIED column references from SQL.

        These are column names used without table prefix in:
        - SELECT clause
        - WHERE clause
        - GROUP BY clause
        - ORDER BY clause

        v6.2 FIXES:
        - Excludes SELECT output aliases (AS alias) from column detection
        - Excludes quoted string literals ('JFK', "ATL") from column detection

        Returns:
            List of unqualified column names
        """
        sql = SQLParser.normalize_sql(sql)
        columns = []

        # =====================================================================
        # v6.2 FIX: Strip quoted literals BEFORE extracting identifiers
        # =====================================================================
        # String literals like 'JFK' or "ATL" are VALUES, not column names.
        # We remove them entirely to prevent false detection.
        # =====================================================================
        sql_no_literals = SQLParser._strip_quoted_literals(sql)

        # =====================================================================
        # v6.2 FIX: Extract SELECT output aliases to EXCLUDE from detection
        # =====================================================================
        # Aliases declared via "AS alias_name" are OUTPUT names, not column refs.
        # Examples: COUNT(*) AS num_flights, SUM(price) AS total_value
        # These must NOT be treated as missing columns.
        # =====================================================================
        select_aliases = SQLParser._extract_select_aliases(sql_no_literals)

        # Extract clauses that may contain column references
        # We need to parse SELECT, WHERE, GROUP BY, ORDER BY
        clauses_to_check = []

        # SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_no_literals, re.IGNORECASE | re.DOTALL)
        if select_match:
            clauses_to_check.append(select_match.group(1))

        # WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+HAVING|\s+LIMIT|$)', sql_no_literals, re.IGNORECASE | re.DOTALL)
        if where_match:
            clauses_to_check.append(where_match.group(1))

        # GROUP BY clause
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)', sql_no_literals, re.IGNORECASE | re.DOTALL)
        if group_match:
            clauses_to_check.append(group_match.group(1))

        # ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', sql_no_literals, re.IGNORECASE | re.DOTALL)
        if order_match:
            clauses_to_check.append(order_match.group(1))

        # Pattern for identifiers (column names)
        identifier_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')

        # Pattern to detect qualified references (table.column) - we skip these
        qualified_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b')

        for clause in clauses_to_check:
            # Find positions of qualified references to skip
            qualified_positions = set()
            for match in qualified_pattern.finditer(clause):
                # Mark positions of both qualifier and column
                qualified_positions.add(match.start(1))
                qualified_positions.add(match.start(2))

            # Extract all identifiers
            for match in identifier_pattern.finditer(clause):
                identifier = match.group(1).lower()
                pos = match.start(1)

                # Skip if:
                # 1. It's a SQL keyword/function
                # 2. It's a table alias
                # 3. It's part of a qualified reference
                # 4. v6.2: It's a SELECT output alias
                if identifier in SQLParser.SQL_KEYWORDS:
                    continue
                if identifier in table_aliases:
                    continue
                if pos in qualified_positions:
                    continue
                if identifier in select_aliases:
                    # v6.2: Skip SELECT output aliases (AS alias)
                    continue

                # Check if preceded by a dot (part of qualified ref)
                pre_context = clause[max(0, pos-5):pos].strip()
                if pre_context.endswith('.'):
                    continue

                columns.append(identifier)

        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)

        return unique_columns

    @staticmethod
    def _strip_quoted_literals(sql: str) -> str:
        """
        Remove quoted string literals from SQL.

        v6.2: Prevents false detection of literals as missing columns.
        'JFK', "ATL", 'Chicago O''Hare' -> removed

        Returns SQL with quoted strings replaced by empty placeholders.
        """
        # Remove single-quoted strings (handles escaped quotes)
        sql = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)
        # Remove double-quoted strings (handles escaped quotes)
        sql = re.sub(r'"(?:[^"\\]|\\.)*"', '""', sql)
        return sql

    @staticmethod
    def _extract_select_aliases(sql: str) -> set:
        """
        Extract output aliases declared in SELECT clause via AS keyword.

        v6.2: These are OUTPUT names, not column references.
        Examples:
            COUNT(*) AS num_flights -> "num_flights" is an alias
            SUM(price) AS total_value -> "total_value" is an alias

        Returns:
            Set of alias names (lowercase) to exclude from column detection.
        """
        aliases = set()

        # Find SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return aliases

        select_clause = select_match.group(1)

        # Pattern: "AS alias_name" - capture the alias after AS
        alias_pattern = re.compile(r'\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b', re.IGNORECASE)

        for match in alias_pattern.finditer(select_clause):
            aliases.add(match.group(1).lower())

        return aliases


# =============================================================================
# RELATIONAL CORRECTOR
# =============================================================================

class RelationalCorrector:
    """
    Corrects structurally invalid SQL using foreign key metadata.

    BEHAVIOR:
    1. Parse SQL to extract table and column references (qualified AND unqualified)
    2. For each column reference, check if column exists in the FROM table
    3. If NOT, check if column exists in a table reachable via FK
    4. If EXACTLY ONE FK path exists -> rewrite with JOIN
    5. If MULTIPLE FK paths exist -> return structured error
    6. If NO FK path exists -> pass through (let column validator handle)

    CRITICAL GUARANTEES:
    - Uses ONLY declared FK metadata (no guessing)
    - Rewrites are DETERMINISTIC (same input -> same output)
    - Ambiguous cases ALWAYS fail (never guess)
    - Original SQL passes through if no correction needed
    """

    def __init__(self, schema_metadata: SchemaMetadata):
        """Initialize with schema metadata."""
        self.schema = schema_metadata
        logger.info(
            f"[RCL] Initialized with {len(schema_metadata.tables)} tables, "
            f"{len(schema_metadata.foreign_keys)} foreign keys"
        )

    def correct(self, sql: str) -> CorrectionResult:
        """
        Attempt to correct relational errors in SQL.

        Args:
            sql: SQL query to correct

        Returns:
            CorrectionResult with corrected SQL or error
        """
        if not sql or not sql.strip():
            return CorrectionResult(
                success=True,
                corrected_sql=sql,
                original_sql=sql
            )

        # Parse SQL structure
        table_aliases, primary_table = SQLParser.extract_tables_with_aliases(sql)

        if not primary_table:
            # No FROM table found - pass through
            return CorrectionResult(
                success=True,
                corrected_sql=sql,
                original_sql=sql
            )

        # Extract unqualified column references
        unqualified_columns = SQLParser.extract_unqualified_columns(sql, table_aliases)

        if not unqualified_columns:
            # No unqualified columns - pass through
            logger.debug("[RCL] No unqualified columns found - no correction needed")
            return CorrectionResult(
                success=True,
                corrected_sql=sql,
                original_sql=sql
            )

        # v6.20: Belt-and-suspenders guard against duplicate JOINs.
        #
        # The Canonical Alias Injector (pre-RCL) pre-qualifies columns that are
        # accessible from already-joined tables, so RCL should see no unqualified
        # references for those columns at this point.  However, if the injector
        # skips normalization (e.g. CTE, UNION, or an internal error), we apply
        # a secondary check here: a column is only "missing" if it cannot be
        # found in ANY table that is already present in the query (primary OR
        # joined tables).  Without this guard, RCL would add a duplicate JOIN
        # for a table that is already in the FROM/JOIN clause, producing
        # inconsistent SQL that fails alias validation.
        all_query_tables: set = set(table_aliases.values())

        missing_columns = []
        for col in unqualified_columns:
            col_found_in_query = any(
                self.schema.column_exists_in_table(t, col)
                for t in all_query_tables
            )
            if not col_found_in_query:
                missing_columns.append(col)

        if not missing_columns:
            # All unqualified columns are accessible from tables already in the
            # query — no structural correction is required.
            logger.debug("[RCL] All columns accessible from existing query tables - no correction needed")
            return CorrectionResult(
                success=True,
                corrected_sql=sql,
                original_sql=sql
            )

        logger.info(f"[RCL] Found {len(missing_columns)} missing column(s): {missing_columns}")

        # Attempt to find FK-based fixes
        fixes = []
        errors = []

        for column in missing_columns:
            fix_result = self._find_fk_fix(primary_table, column)

            if fix_result["status"] == "fixable":
                fixes.append(fix_result)
            elif fix_result["status"] == "ambiguous":
                errors.append(fix_result)
            # "not_resolvable" cases are passed through (let column validator handle)

        # If any ambiguous cases, emit STRUCTURED AMBIGUITY (not error)
        # This allows the system to ask for clarification rather than failing
        if errors:
            # Build structured ambiguity for the FIRST ambiguous column
            # (Multiple ambiguities would be resolved one at a time)
            first_ambiguity = errors[0]

            structured_ambiguity = self._build_structured_ambiguity(
                source_table=primary_table,
                column=first_ambiguity["column"],
                candidates=first_ambiguity.get("candidates", [])
            )

            logger.info(
                f"[RCL_AMBIGUITY] Structured ambiguity detected: "
                f"column='{first_ambiguity['column']}', "
                f"options={[opt.fk_column for opt in structured_ambiguity.options]}"
            )

            # Return with ambiguity field set (NOT error)
            # success=False but ambiguity is set means "needs clarification"
            return CorrectionResult(
                success=False,
                corrected_sql=None,
                original_sql=sql,
                ambiguity=structured_ambiguity
            )

        # Apply fixes if any
        if fixes:
            try:
                corrected_sql = self._apply_fixes(sql, fixes, primary_table, table_aliases)
                logger.info(f"[RCL] Applied {len(fixes)} fix(es)")
                for fix in fixes:
                    logger.info(
                        f"  - Column '{fix['column']}' resolved via "
                        f"{primary_table}.{fix['fk_column']} -> {fix['target_table']}.{fix['target_column']}"
                    )

                return CorrectionResult(
                    success=True,
                    corrected_sql=corrected_sql,
                    original_sql=sql,
                    applied_fixes=[{
                        "column": f["column"],
                        "source_table": primary_table,
                        "target_table": f["target_table"],
                        "join_condition": f"{primary_table}.{f['fk_column']} = {f['target_table']}.{f['target_column']}"
                    } for f in fixes]
                )
            except Exception as e:
                logger.error(f"[RCL] Failed to apply fixes: {e}")
                return CorrectionResult(
                    success=False,
                    corrected_sql=None,
                    original_sql=sql,
                    error={"type": "rewrite_failed", "message": str(e)}
                )

        # No fixes applied - pass through
        return CorrectionResult(
            success=True,
            corrected_sql=sql,
            original_sql=sql
        )

    def _find_fk_fix(self, source_table: str, missing_column: str) -> Dict[str, Any]:
        """
        Attempt to find a FK-based fix for a missing column.

        Returns dict with:
            status: "fixable" | "ambiguous" | "not_resolvable"
            (+ fix details if fixable)
            (+ candidate_paths if ambiguous)
        """
        column_lower = missing_column.lower()

        # Find all FKs from source table
        fks = self.schema.find_fks_from_table(source_table)

        # Find which target tables have the missing column
        candidates = []

        for fk in fks:
            target_columns = self.schema.get_table_columns(fk.target_table)
            target_columns_lower = {c.lower() for c in target_columns}

            if column_lower in target_columns_lower:
                candidates.append({
                    "fk": fk,
                    "fk_column": fk.source_column,
                    "target_table": fk.target_table,
                    "target_column": fk.target_column,
                })

        if not candidates:
            # Column not found via any FK path
            logger.debug(f"[RCL] Column '{missing_column}' not resolvable via FK from {source_table}")
            return {
                "status": "not_resolvable",
                "source_table": source_table,
                "column": missing_column
            }

        if len(candidates) == 1:
            # Exactly one path - fixable
            c = candidates[0]
            return {
                "status": "fixable",
                "source_table": source_table,
                "column": missing_column,
                "fk_column": c["fk_column"],
                "target_table": c["target_table"],
                "target_column": c["target_column"]
            }

        # Multiple paths - ambiguous
        # Return full candidate info for structured ambiguity
        return {
            "status": "ambiguous",
            "source_table": source_table,
            "column": missing_column,
            "candidates": candidates,  # Full candidate info for structured ambiguity
            "candidate_paths": [
                f"{source_table}.{c['fk_column']} -> {c['target_table']}"
                for c in candidates
            ]
        }

    def _format_ambiguity_error(self, errors: List[Dict[str, Any]]) -> str:
        """Format a user-friendly ambiguity error message."""
        messages = []
        for err in errors:
            col = err["column"]
            paths = err["candidate_paths"]
            messages.append(
                f"Ambiguous column reference \"{col}\".\n"
                f"Multiple join paths found:\n" +
                "\n".join(f"  - {p}" for p in paths)
            )
        return "\n\n".join(messages)

    def _derive_fk_description(self, fk_column: str, target_table: str) -> str:
        """
        Derive a human-readable description from FK column name.

        This is purely mechanical string transformation - NO semantic inference.

        Examples:
            arrival_airport -> "arriving flights" (for flight table)
            departure_airport -> "departing flights"
            origin_warehouse -> "origin warehouse"
            destination_warehouse -> "destination warehouse"
        """
        # Normalize
        col_lower = fk_column.lower()
        target_name = target_table.split(".")[-1].lower()

        # Common patterns for directional FKs
        if "arrival" in col_lower or "destination" in col_lower or "to_" in col_lower:
            return f"arriving/destination ({fk_column})"
        if "departure" in col_lower or "origin" in col_lower or "from_" in col_lower:
            return f"departing/origin ({fk_column})"

        # Generic: use column name with target table
        # Remove common suffixes for readability
        clean_name = col_lower.replace("_id", "").replace("_code", "").replace("_", " ")
        return f"{clean_name} ({fk_column})"

    def _build_structured_ambiguity(
        self,
        source_table: str,
        column: str,
        candidates: List[Dict[str, Any]]
    ) -> RelationalAmbiguity:
        """
        Build a structured ambiguity object from candidates.

        CRITICAL: This is derived ONLY from FK metadata.
        No semantic inference. No ranking. No guessing.
        """
        options = []
        target_table = candidates[0]["target_table"] if candidates else ""

        for c in candidates:
            description = self._derive_fk_description(c["fk_column"], c["target_table"])
            options.append(AmbiguityOption(
                fk_column=c["fk_column"],
                target_table=c["target_table"],
                target_column=c["target_column"],
                description=description
            ))

        return RelationalAmbiguity(
            type="RELATIONAL_AMBIGUITY",
            source_table=source_table,
            target_table=target_table,
            column=column,
            options=options
        )

    def _parse_sql_sections(self, sql: str) -> Dict[str, str]:
        """
        Parse SQL into structured sections for safe clause manipulation.

        Returns dict with keys: select, from_table, joins, where, group_by,
        having, order_by, limit, and any trailing content.

        This is NOT a full SQL parser - it's a minimal section splitter
        that handles the common patterns produced by NL-SQL engines.
        """
        # Normalize whitespace for easier parsing
        normalized = re.sub(r'\s+', ' ', sql).strip()

        sections = {
            'select': '',
            'from_table': '',
            'joins': '',
            'where': '',
            'group_by': '',
            'having': '',
            'order_by': '',
            'limit': '',
        }

        # Keywords that delimit sections (order matters for detection)
        # Each tuple: (keyword_pattern, section_name, terminators)
        section_markers = [
            (r'\bSELECT\b', 'select', [r'\bFROM\b']),
            (r'\bFROM\b', 'from_table', [r'\bJOIN\b', r'\bWHERE\b', r'\bGROUP\s+BY\b', r'\bHAVING\b', r'\bORDER\s+BY\b', r'\bLIMIT\b']),
            (r'\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+)?JOIN\b', 'joins', [r'\bWHERE\b', r'\bGROUP\s+BY\b', r'\bHAVING\b', r'\bORDER\s+BY\b', r'\bLIMIT\b']),
            (r'\bWHERE\b', 'where', [r'\bGROUP\s+BY\b', r'\bHAVING\b', r'\bORDER\s+BY\b', r'\bLIMIT\b']),
            (r'\bGROUP\s+BY\b', 'group_by', [r'\bHAVING\b', r'\bORDER\s+BY\b', r'\bLIMIT\b']),
            (r'\bHAVING\b', 'having', [r'\bORDER\s+BY\b', r'\bLIMIT\b']),
            (r'\bORDER\s+BY\b', 'order_by', [r'\bLIMIT\b']),
            (r'\bLIMIT\b', 'limit', []),
        ]

        remaining = normalized

        for keyword_pattern, section_name, terminators in section_markers:
            match = re.search(keyword_pattern, remaining, re.IGNORECASE)
            if not match:
                continue

            start_pos = match.end()

            # Find where this section ends (at the next terminator or end of string)
            end_pos = len(remaining)
            for term_pattern in terminators:
                term_match = re.search(term_pattern, remaining[start_pos:], re.IGNORECASE)
                if term_match:
                    end_pos = min(end_pos, start_pos + term_match.start())

            # Extract section content (without the keyword itself)
            section_content = remaining[start_pos:end_pos].strip()

            # For JOIN section, include multiple JOINs
            if section_name == 'joins':
                # Find all consecutive JOINs
                join_pattern = re.compile(
                    r'(\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+)?JOIN\b.*?)(?=\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+)?JOIN\b|\bWHERE\b|\bGROUP\s+BY\b|\bHAVING\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
                    re.IGNORECASE
                )
                joins = join_pattern.findall(remaining)
                if joins:
                    sections['joins'] = ' '.join(j.strip() for j in joins)
            else:
                sections[section_name] = section_content

        return sections

    def _reassemble_sql(self, sections: Dict[str, str], new_joins: List[str]) -> str:
        """
        Reassemble SQL from sections with new JOINs inserted in correct position.

        Clause order: SELECT ... FROM ... [JOINs] [WHERE] [GROUP BY] [HAVING] [ORDER BY] [LIMIT]
        """
        parts = []

        # SELECT clause
        if sections['select']:
            parts.append(f"SELECT {sections['select']}")

        # FROM clause
        if sections['from_table']:
            parts.append(f"FROM {sections['from_table']}")

        # Existing JOINs (if any)
        if sections['joins']:
            parts.append(sections['joins'])

        # New JOINs
        for join in new_joins:
            parts.append(join)

        # WHERE clause
        if sections['where']:
            parts.append(f"WHERE {sections['where']}")

        # GROUP BY clause
        if sections['group_by']:
            parts.append(f"GROUP BY {sections['group_by']}")

        # HAVING clause
        if sections['having']:
            parts.append(f"HAVING {sections['having']}")

        # ORDER BY clause
        if sections['order_by']:
            parts.append(f"ORDER BY {sections['order_by']}")

        # LIMIT clause
        if sections['limit']:
            parts.append(f"LIMIT {sections['limit']}")

        return '\n'.join(parts)

    def _apply_fixes(
        self,
        sql: str,
        fixes: List[Dict[str, Any]],
        primary_table: str,
        table_aliases: Dict[str, str]
    ) -> str:
        """
        Apply fixes by adding JOINs and qualifying column references.

        STRATEGY (v6.1.2 - Structured SQL Rewriting):
        1. Parse SQL into structured sections (SELECT, FROM, GROUP BY, etc.)
        2. Build JOIN clauses with proper aliases
        3. Reassemble SQL in correct clause order
        4. Qualify missing columns with JOIN aliases
        5. v6.7: Qualify ALL unqualified columns to prevent ambiguity (BUG 2 fix)

        This ensures JOINs always appear after FROM and before GROUP BY/ORDER BY.

        Args:
            sql: Original SQL
            fixes: List of fixes to apply
            primary_table: The FROM table
            table_aliases: Existing table aliases

        Returns:
            Corrected SQL string
        """
        # =====================================================================
        # v6.7: BUG 3 FIX - Deduplicate by JOIN signature, not just target table
        # =====================================================================
        # A JOIN signature is (fk_column, target_table, target_column).
        # This prevents duplicate JOINs when multiple columns resolve via
        # the same FK path.
        # =====================================================================
        seen_join_signatures: Set[Tuple[str, str, str]] = set()
        unique_fixes: List[Dict[str, Any]] = []

        for fix in fixes:
            signature = (
                fix["fk_column"].lower(),
                fix["target_table"].lower(),
                fix["target_column"].lower()
            )
            if signature not in seen_join_signatures:
                seen_join_signatures.add(signature)
                unique_fixes.append(fix)
                logger.debug(f"[RCL] JOIN signature added: {signature}")
            else:
                logger.debug(f"[RCL] Duplicate JOIN signature skipped: {signature}")

        # Group unique fixes by target table
        target_to_fixes: Dict[str, List[Dict[str, Any]]] = {}
        for fix in unique_fixes:
            target = fix["target_table"]
            if target not in target_to_fixes:
                target_to_fixes[target] = []
            target_to_fixes[target].append(fix)

        # Generate aliases for joined tables
        new_aliases: Dict[str, str] = {}
        alias_counter = 1

        for target_table in target_to_fixes.keys():
            base = target_table.split(".")[-1]
            # Use short alias: first letter + counter
            new_aliases[target_table] = f"{base[0]}{alias_counter}"
            alias_counter += 1

        # Get or create primary table alias
        primary_ref = None
        for alias, table in table_aliases.items():
            if table.lower() == primary_table.lower() and alias != table.split(".")[-1].lower():
                primary_ref = alias
                break

        # If no alias exists, we need to add one to the FROM clause
        primary_short = primary_table.split(".")[-1]
        if not primary_ref:
            primary_ref = primary_short[0].lower()  # Use first letter as alias

        # Parse SQL into sections
        sections = self._parse_sql_sections(sql)

        # Update FROM clause to include alias if needed
        from_content = sections['from_table'].strip()
        # Check if FROM already has an alias
        from_parts = from_content.split()
        if len(from_parts) == 1:
            # No alias - add one
            sections['from_table'] = f"{from_content} {primary_ref}"
        elif len(from_parts) >= 2 and from_parts[1].upper() == 'AS':
            # Has "AS alias" - use existing alias
            primary_ref = from_parts[2] if len(from_parts) > 2 else primary_ref
        elif len(from_parts) == 2:
            # Has implicit alias
            primary_ref = from_parts[1]

        # Build JOIN clauses (using unique_fixes to avoid duplicates)
        join_clauses = []
        for target_table, fix_list in target_to_fixes.items():
            alias = new_aliases[target_table]
            fix = fix_list[0]

            join_clause = (
                f"JOIN {target_table} {alias} "
                f"ON {primary_ref}.{fix['fk_column']} = {alias}.{fix['target_column']}"
            )
            join_clauses.append(join_clause)
            logger.info(f"[RCL] Added JOIN: {primary_ref}.{fix['fk_column']} = {alias}.{fix['target_column']}")

        # Reassemble SQL with JOINs in correct position
        corrected_sql = self._reassemble_sql(sections, join_clauses)

        # =====================================================================
        # v6.7: BUG 2 FIX - Qualify ALL unqualified columns to prevent ambiguity
        # =====================================================================
        # After adding JOINs, columns like first_name may exist in BOTH the
        # primary table AND joined tables. We must qualify ALL unqualified
        # columns, not just the ones we fixed.
        #
        # Strategy:
        # 1. Get columns from primary table
        # 2. Get columns from all joined tables
        # 3. Find unqualified columns in SQL
        # 4. Qualify each with the appropriate alias
        # =====================================================================
        primary_columns = self.schema.get_table_columns(primary_table)
        primary_columns_lower = {c.lower() for c in primary_columns}

        # Build mapping: column -> alias for joined tables
        joined_column_to_alias: Dict[str, str] = {}
        for target_table, alias in new_aliases.items():
            target_columns = self.schema.get_table_columns(target_table)
            for col in target_columns:
                col_lower = col.lower()
                # Only map if not already mapped (first join wins for ambiguous columns)
                if col_lower not in joined_column_to_alias:
                    joined_column_to_alias[col_lower] = alias

        # Qualify the missing columns FIRST (they go to joined tables)
        for fix in fixes:
            column = fix["column"]
            target = fix["target_table"]
            alias = new_aliases[target]

            # Replace unqualified column with alias.column
            pattern = rf'(?<!\.)\b{re.escape(column)}\b(?!\s*\.)'
            replacement = f"{alias}.{column}"
            corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)

        # Now qualify remaining unqualified columns that exist in primary table
        # These need primary_ref prefix to avoid ambiguity
        all_table_aliases = {primary_ref.lower(), *[a.lower() for a in new_aliases.values()]}

        # Find all unqualified columns in the corrected SQL
        unqualified = SQLParser.extract_unqualified_columns(corrected_sql, all_table_aliases)

        for col in unqualified:
            col_lower = col.lower()
            # Skip if it's a SQL keyword or already qualified
            if col_lower in SQLParser.SQL_KEYWORDS:
                continue

            # Determine which alias to use
            if col_lower in primary_columns_lower:
                # Column exists in primary table - qualify with primary alias
                pattern = rf'(?<!\.)\b{re.escape(col)}\b(?!\s*\.)'
                replacement = f"{primary_ref}.{col}"
                corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                logger.debug(f"[RCL] Qualified column '{col}' with primary alias '{primary_ref}'")

        return corrected_sql

    def correct_with_forced_fk(
        self,
        sql: str,
        forced_fk: Dict[str, str]
    ) -> CorrectionResult:
        """
        Correct SQL with a FORCED FK preference (from resolved ambiguity).

        This is called AFTER the user has clarified which FK to use.
        The forced_fk dict maps table -> fk_column to use.

        Args:
            sql: SQL query to correct
            forced_fk: Dict of {source_table: fk_column} to force

        Returns:
            CorrectionResult with deterministically corrected SQL

        Example:
            forced_fk = {"flight": "arrival_airport"}
            -> Forces flight -> airport join via arrival_airport
        """
        if not sql or not sql.strip():
            return CorrectionResult(success=True, corrected_sql=sql, original_sql=sql)

        logger.info(f"[AMBIGUITY_RESOLVED] Applying forced FK: {forced_fk}")

        # Parse SQL structure
        table_aliases, primary_table = SQLParser.extract_tables_with_aliases(sql)

        if not primary_table:
            return CorrectionResult(success=True, corrected_sql=sql, original_sql=sql)

        # Extract unqualified columns
        unqualified_columns = SQLParser.extract_unqualified_columns(sql, table_aliases)

        if not unqualified_columns:
            return CorrectionResult(success=True, corrected_sql=sql, original_sql=sql)

        # Find missing columns
        missing_columns = [
            col for col in unqualified_columns
            if not self.schema.column_exists_in_table(primary_table, col)
        ]

        if not missing_columns:
            return CorrectionResult(success=True, corrected_sql=sql, original_sql=sql)

        # Build fixes using FORCED FK preferences
        fixes = []
        for column in missing_columns:
            fix = self._find_fk_fix_with_forced(primary_table, column, forced_fk)
            if fix:
                fixes.append(fix)

        if not fixes:
            return CorrectionResult(success=True, corrected_sql=sql, original_sql=sql)

        # Apply fixes
        try:
            corrected_sql = self._apply_fixes(sql, fixes, primary_table, table_aliases)
            logger.info(f"[AMBIGUITY_RESOLVED] Applied {len(fixes)} fix(es) with forced FK")

            return CorrectionResult(
                success=True,
                corrected_sql=corrected_sql,
                original_sql=sql,
                applied_fixes=[{
                    "column": f["column"],
                    "source_table": primary_table,
                    "target_table": f["target_table"],
                    "join_condition": f"{primary_table}.{f['fk_column']} = {f['target_table']}.{f['target_column']}",
                    "forced": True
                } for f in fixes]
            )
        except Exception as e:
            logger.error(f"[RCL] Failed to apply forced FK fix: {e}")
            return CorrectionResult(
                success=False,
                corrected_sql=None,
                original_sql=sql,
                error={"type": "forced_fix_failed", "message": str(e)}
            )

    def _find_fk_fix_with_forced(
        self,
        source_table: str,
        column: str,
        forced_fk: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Find FK fix using a FORCED preference.

        If the source table has a forced FK, use ONLY that FK.
        """
        column_lower = column.lower()
        source_lower = source_table.lower()
        source_short = source_table.split(".")[-1].lower()

        # Check if we have a forced FK for this table
        forced_col = None
        for table_key, fk_col in forced_fk.items():
            if table_key.lower() == source_lower or table_key.lower() == source_short:
                forced_col = fk_col.lower()
                break

        # Find all FKs from source table
        fks = self.schema.find_fks_from_table(source_table)

        for fk in fks:
            # If forced, only use the forced FK
            if forced_col and fk.source_column.lower() != forced_col:
                continue

            # Check if target table has the column
            target_columns = self.schema.get_table_columns(fk.target_table)
            target_columns_lower = {c.lower() for c in target_columns}

            if column_lower in target_columns_lower:
                return {
                    "status": "fixable",
                    "source_table": source_table,
                    "column": column,
                    "fk_column": fk.source_column,
                    "target_table": fk.target_table,
                    "target_column": fk.target_column
                }

        return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_relational_corrector(schema: Dict[str, Any]) -> Optional[RelationalCorrector]:
    """
    Create a RelationalCorrector from the standard schema format.

    Args:
        schema: Schema dict with structure:
                {
                    "tables": {
                        "table_name": {
                            "columns": [{"name": "col", "type": "..."}, ...],
                            "foreign_keys": [
                                {"column": "fk_col", "target_table": "...", "target_column": "..."}
                            ]
                        }
                    }
                }

    Returns:
        RelationalCorrector instance or None if schema is invalid
    """
    if not schema or "tables" not in schema:
        logger.warning("[RCL] No schema provided - relational correction disabled")
        return None

    # Build metadata
    tables: Dict[str, Set[str]] = {}
    foreign_keys: List[ForeignKeyInfo] = []

    for table_name, table_info in schema["tables"].items():
        # Extract columns
        columns = table_info.get("columns", [])
        tables[table_name] = {col["name"] for col in columns}

        # Extract foreign keys
        fks = table_info.get("foreign_keys", [])
        for fk in fks:
            foreign_keys.append(ForeignKeyInfo(
                source_table=table_name,
                source_column=fk["column"],
                target_table=fk["target_table"],
                target_column=fk["target_column"]
            ))

    if not foreign_keys:
        logger.info("[RCL] No foreign keys in schema - relational correction will be limited")

    metadata = SchemaMetadata(tables=tables, foreign_keys=foreign_keys)
    return RelationalCorrector(metadata)


def correct_sql(sql: str, schema: Dict[str, Any]) -> CorrectionResult:
    """
    Convenience function to correct SQL in one call.

    Args:
        sql: SQL query to correct
        schema: Schema metadata

    Returns:
        CorrectionResult
    """
    corrector = create_relational_corrector(schema)
    if not corrector:
        return CorrectionResult(
            success=True,
            corrected_sql=sql,
            original_sql=sql
        )
    return corrector.correct(sql)
