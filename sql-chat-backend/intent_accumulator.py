"""
IntentAccumulator - Sole Clarification Authority

Decides whether to clarify or proceed to NL-SQL execution.

DECISION RULE:
    if entity AND metric are known -> PROCEED
    else -> CLARIFY (ask for missing field)

No confidence scores. No magic numbers. Schema-blind for decisions.

v6.3: Schema-backed clarification prompts
- Uses schema metadata to suggest concrete metric options
- Suggestions are ADVISORY only (user must still choose)
- Decision logic remains unchanged
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA REFERENCE (v6.3 - For Clarification Suggestions Only)
# =============================================================================
# This module-level reference holds schema metadata for generating
# schema-backed clarification prompts. It does NOT affect decision logic.
#
# CRITICAL: This is used ONLY for suggestion generation, never for decisions.
# =============================================================================
_schema_reference: Optional[Dict[str, Any]] = None


def set_schema_reference(schema: Dict[str, Any]) -> None:
    """
    Set schema reference for clarification suggestions.

    Called once during system initialization. Does NOT affect decision logic.

    Args:
        schema: DatabaseManager.schema structure
    """
    global _schema_reference
    _schema_reference = schema
    if schema and "tables" in schema:
        logger.info(f"[ACCUMULATOR] Schema reference set ({len(schema['tables'])} tables)")
    else:
        logger.info("[ACCUMULATOR] Schema reference set (empty or invalid)")


def get_schema_reference() -> Optional[Dict[str, Any]]:
    """Get the current schema reference (for clarification suggestions only)."""
    return _schema_reference


# Safe defaults applied when values not specified
SAFE_DEFAULTS = {
    "time_scope": "all_time",
    "n": 10,
    "ranking": "none",
}

# =============================================================================
# INVALID METRIC TOKENS (v6.4 - Non-Metric Verb Hardening)
# =============================================================================
# These tokens are VERBS/ACTIONS, not metrics. They describe the operation
# the user wants ("list customers") but say nothing about HOW to measure.
#
# CRITICAL BUG FIX: "Who are the best customers?" was extracting metric="list"
# which passed validation and caused silent inference of "best" = award_points.
#
# RULE: If metric is in INVALID_METRIC_TOKENS -> has_metric() returns False
#       This naturally triggers clarification via can_proceed().
#
# Examples:
#   ❌ "list customers" + metric=list -> INVALID (list is not a metric)
#   ❌ "show top passengers" + metric=show -> INVALID (show is not a metric)
#   ✅ "list customers by booking count" + metric=booking_count -> VALID
#
# v6.10 EXCEPTION: Row-selection queries (no ranking, just listing rows)
#   ✅ "list flights" + entity=flight + no ranking -> ROW_SELECT (proceed)
#   ✅ "show 50 passengers" + entity=passenger + no ranking -> ROW_SELECT
# =============================================================================
INVALID_METRIC_TOKENS = {
    "list", "show", "get", "display", "fetch", "find", "retrieve",
    "give", "tell", "select", "return", "view", "see", "lookup",
}

# =============================================================================
# ROW SELECT TOKENS (v6.10 - MVP Row Selection Support)
# =============================================================================
# These tokens indicate the user wants to SELECT ROWS, not aggregate.
# When combined with a known entity and NO ranking, these form valid
# row-selection queries that should proceed without clarification.
#
# RULE: metric in ROW_SELECT_TOKENS AND entity known AND no ranking -> PROCEED
#
# Examples:
#   ✅ "list flights" -> SELECT * FROM flight LIMIT 50
#   ✅ "show passengers" -> SELECT * FROM passenger LIMIT 50
#   ❌ "list top flights" -> NEEDS METRIC (ranking requires ordering dimension)
# =============================================================================
ROW_SELECT_TOKENS = {
    "list", "show", "get", "display", "fetch", "find", "retrieve",
    "give", "tell", "select", "return", "view", "see", "lookup",
}

# Internal marker for row-selection intent (not user-facing)
ROW_SELECT_METRIC = "_row_select"


# =============================================================================
# METRIC COMPLETENESS (v6.2 - Aggregation Primitive Hardening)
# =============================================================================
# These are AGGREGATION PRIMITIVES, not complete business metrics.
# "count" alone is ambiguous: count of what? bookings? flights? passengers?
# A complete metric must be BOUND to a domain concept.
#
# RULE: Aggregation primitive + ranking -> clarification required
#       Aggregation primitive + domain context -> proceed
#
# Examples:
#   ❌ "top passengers" + metric=count -> INCOMPLETE (count of what?)
#   ✅ "top passengers by total booking value" -> COMPLETE (bound to booking value)
#   ✅ "count of flights" -> COMPLETE (bound to flights via entity)
# =============================================================================
BARE_AGGREGATION_PRIMITIVES = {
    "count", "sum", "total", "avg", "average", "min", "max",
}

# =============================================================================
# COMPARISON GUARDRAIL (v6.2 - Unsafe Comparison Detection)
# =============================================================================
# Comparison queries imply segmentation, multiple joins, and undefined
# behavioral metrics. These MUST be blocked or clarified, never executed.
#
# Detection: metric == "comparison" OR keywords in query
# =============================================================================
COMPARISON_KEYWORDS = {
    "compare", "comparison", "vs", "versus", "vs.", "v.s.",
    "difference between", "differ from", "compared to",
}


# =============================================================================
# AMBIGUOUS SEMANTIC ALIASES (v6.7 - Entity Disambiguation)
# =============================================================================
# These semantic terms map to MULTIPLE base tables in typical airline schemas.
# They MUST trigger clarification, NEVER silent inference.
#
# Example: "customer" could mean:
#   - passenger (person who flies)
#   - account (login/billing entity)
#   - frequent_flyer (loyalty program member)
#
# RULE: If entity_type matches an ambiguous alias -> CLARIFY
# =============================================================================
AMBIGUOUS_SEMANTIC_ALIASES = {
    "customer": {
        "options": ["passenger", "account", "frequent_flyer"],
        "question": "By 'customer', do you mean:",
        "descriptions": {
            "passenger": "People who have booked flights",
            "account": "User accounts (login/billing)",
            "frequent_flyer": "Loyalty program members",
        }
    },
    "user": {
        "options": ["passenger", "account"],
        "question": "By 'user', do you mean:",
        "descriptions": {
            "passenger": "People who have booked flights",
            "account": "System user accounts",
        }
    },
    "member": {
        "options": ["passenger", "frequent_flyer"],
        "question": "By 'member', do you mean:",
        "descriptions": {
            "passenger": "Any passenger",
            "frequent_flyer": "Loyalty program members",
        }
    },
}


# =============================================================================
# NUMERIC TYPE DETECTION (v6.3 - For Schema-Backed Suggestions)
# =============================================================================
# SQL type patterns that indicate numeric columns suitable for aggregation.
# These are database-agnostic patterns matching common SQL type names.
# =============================================================================
NUMERIC_TYPE_PATTERNS = {
    "int", "integer", "bigint", "smallint", "tinyint",
    "decimal", "numeric", "float", "double", "real",
    "money", "currency", "number",
}


# =============================================================================
# SCHEMA-BACKED METRIC SUGGESTIONS (v6.3)
# =============================================================================

@dataclass
class MetricSuggestion:
    """
    A single metric suggestion derived from schema.

    This is ADVISORY only - the system does NOT auto-select.

    Attributes:
        label: Human-readable label (e.g., "Total booking value")
        expression: SQL expression (e.g., "SUM(booking.amount)")
        source: Source column (e.g., "booking.amount")
        aggregation_type: Type of aggregation (count/sum/avg/etc.)
    """
    label: str
    expression: str
    source: str
    aggregation_type: str

    def to_option_string(self) -> str:
        """Format as user-facing option string."""
        return f"{self.label} ({self.expression})"


def _is_numeric_type(type_str: str) -> bool:
    """
    Check if a SQL type string represents a numeric type.

    Database-agnostic: matches common patterns across PostgreSQL, MySQL, etc.
    """
    type_lower = type_str.lower()
    return any(pattern in type_lower for pattern in NUMERIC_TYPE_PATTERNS)


def _find_table_for_entity(entity_type: str, schema: Dict[str, Any]) -> Optional[str]:
    """
    Find the primary table for an entity type.

    Uses simple name matching (entity name appears in table name).
    Returns fully-qualified table name or None.

    IMPORTANT: This is heuristic matching for suggestion purposes only.
    It does NOT affect query execution or decision logic.
    """
    if not schema or "tables" not in schema:
        return None

    entity_lower = entity_type.lower().strip()

    # Common entity-to-table mappings (singular to potentially plural)
    entity_variants = [entity_lower]
    if entity_lower.endswith("s"):
        entity_variants.append(entity_lower[:-1])  # passengers -> passenger
    else:
        entity_variants.append(entity_lower + "s")  # passenger -> passengers

    # Also handle common synonyms
    if entity_lower in ("customer", "customers"):
        entity_variants.extend(["passenger", "passengers", "client", "clients"])
    if entity_lower in ("passenger", "passengers"):
        entity_variants.extend(["customer", "customers"])

    for table_name in schema["tables"].keys():
        # Extract simple table name (without schema prefix)
        simple_name = table_name.split(".")[-1].lower()

        for variant in entity_variants:
            if variant == simple_name or variant in simple_name:
                return table_name

    return None


def _get_numeric_columns(table_name: str, schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Get numeric columns from a table.

    Returns list of (column_name, column_type) tuples.
    """
    if not schema or "tables" not in schema:
        return []

    table_info = schema["tables"].get(table_name)
    if not table_info:
        return []

    numeric_cols = []
    for col in table_info.get("columns", []):
        col_name = col.get("name", "")
        col_type = str(col.get("type", ""))

        if _is_numeric_type(col_type):
            # Skip ID/key columns (not meaningful for aggregation)
            if not col_name.lower().endswith("_id") and col_name.lower() != "id":
                numeric_cols.append((col_name, col_type))

    return numeric_cols


def _get_fk_targets(table_name: str, schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Get foreign key targets from a table (1-hop relationships).

    Returns list of {fk_column, target_table, target_column} dicts.
    """
    if not schema or "tables" not in schema:
        return []

    table_info = schema["tables"].get(table_name)
    if not table_info:
        return []

    return table_info.get("foreign_keys", [])


def suggest_metrics_for_entity(
    entity_type: str,
    schema: Optional[Dict[str, Any]] = None
) -> List[MetricSuggestion]:
    """
    Generate metric suggestions from schema for a given entity type.

    v6.3: Schema-backed clarification prompts.

    This function MUST NOT decide or filter intent.
    It only suggests. The user must still explicitly choose.

    Args:
        entity_type: The entity being queried (e.g., "passenger", "flight")
        schema: Database schema (uses module reference if not provided)

    Returns:
        List of MetricSuggestion objects (max 5, per UX rule)

    INVARIANTS:
        - Does NOT execute SQL
        - Does NOT modify intent state
        - Does NOT make decisions
        - Returns empty list if no suggestions available
    """
    # Use provided schema or module reference
    if schema is None:
        schema = get_schema_reference()

    if not schema or not entity_type:
        return []

    suggestions: List[MetricSuggestion] = []

    # Find the primary table for this entity
    primary_table = _find_table_for_entity(entity_type, schema)

    if not primary_table:
        logger.debug(f"[ACCUMULATOR] No table found for entity: {entity_type}")
        return []

    simple_table = primary_table.split(".")[-1]
    logger.info(f"[ACCUMULATOR] Finding metrics for entity={entity_type} -> table={simple_table}")

    # 1. Add count of entity (always valid for any table)
    suggestions.append(MetricSuggestion(
        label=f"Number of {entity_type}s",
        expression=f"COUNT({simple_table}.*)",
        source=f"{simple_table}",
        aggregation_type="count"
    ))

    # 2. Add numeric columns from the primary table
    numeric_cols = _get_numeric_columns(primary_table, schema)
    for col_name, col_type in numeric_cols:
        # Generate appropriate aggregations based on column name patterns
        col_lower = col_name.lower()

        if "amount" in col_lower or "price" in col_lower or "value" in col_lower or "cost" in col_lower:
            suggestions.append(MetricSuggestion(
                label=f"Total {col_name.replace('_', ' ')}",
                expression=f"SUM({simple_table}.{col_name})",
                source=f"{simple_table}.{col_name}",
                aggregation_type="sum"
            ))
        elif "count" in col_lower or "points" in col_lower or "miles" in col_lower:
            suggestions.append(MetricSuggestion(
                label=f"Total {col_name.replace('_', ' ')}",
                expression=f"SUM({simple_table}.{col_name})",
                source=f"{simple_table}.{col_name}",
                aggregation_type="sum"
            ))
        elif "duration" in col_lower or "distance" in col_lower or "time" in col_lower:
            suggestions.append(MetricSuggestion(
                label=f"Average {col_name.replace('_', ' ')}",
                expression=f"AVG({simple_table}.{col_name})",
                source=f"{simple_table}.{col_name}",
                aggregation_type="avg"
            ))
        else:
            # Generic numeric column - offer sum
            suggestions.append(MetricSuggestion(
                label=f"Total {col_name.replace('_', ' ')}",
                expression=f"SUM({simple_table}.{col_name})",
                source=f"{simple_table}.{col_name}",
                aggregation_type="sum"
            ))

    # 3. Check FK-related tables (1 hop) for additional metrics
    fk_targets = _get_fk_targets(primary_table, schema)

    for fk in fk_targets:
        target_table = fk.get("target_table", "")
        if not target_table:
            continue

        target_simple = target_table.split(".")[-1]
        target_numeric = _get_numeric_columns(target_table, schema)

        for col_name, col_type in target_numeric[:2]:  # Limit to 2 per FK table
            col_lower = col_name.lower()

            if "amount" in col_lower or "price" in col_lower or "value" in col_lower:
                suggestions.append(MetricSuggestion(
                    label=f"Total {target_simple} {col_name.replace('_', ' ')}",
                    expression=f"SUM({target_simple}.{col_name})",
                    source=f"{target_simple}.{col_name}",
                    aggregation_type="sum"
                ))

    # 4. Also check reverse FKs (tables that reference this entity)
    for other_table, table_info in schema.get("tables", {}).items():
        if other_table == primary_table:
            continue

        for fk in table_info.get("foreign_keys", []):
            if fk.get("target_table") == primary_table:
                # This table references our entity - count relationship
                other_simple = other_table.split(".")[-1]
                suggestions.append(MetricSuggestion(
                    label=f"Number of {other_simple}s",
                    expression=f"COUNT({other_simple}.*)",
                    source=f"{other_simple}",
                    aggregation_type="count"
                ))
                break  # One per table

    # Cap at 5 suggestions (UX rule)
    if len(suggestions) > 5:
        logger.info(
            f"[ACCUMULATOR] Too many suggestions ({len(suggestions)}) - "
            f"capping at 5"
        )
        suggestions = suggestions[:5]

    # Log suggestions for verification
    if suggestions:
        sources = [s.source for s in suggestions]
        logger.info(f"[ACCUMULATOR] Suggested metrics for entity={entity_type}: {sources}")

    return suggestions


def format_schema_suggestions(suggestions: List[MetricSuggestion]) -> Tuple[str, List[str]]:
    """
    Format metric suggestions for clarification display.

    Returns:
        Tuple of (header_text, list of option strings)
    """
    if not suggestions:
        return "", []

    header = "\n\n**Suggested metrics based on your database:**"
    options = [s.to_option_string() for s in suggestions]

    return header, options


# =============================================================================
# INTENT STATE
# =============================================================================

@dataclass
class PendingAmbiguity:
    """
    Tracks a pending relational ambiguity awaiting user clarification.

    This is populated when RCL detects multiple FK paths and requires
    user input to resolve.

    Attributes:
        source_table: Table with ambiguous FK
        target_table: Table being joined
        column: Column that triggered ambiguity
        options: List of FK options [{fk_column, description}, ...]
        original_query: The query that triggered this
    """
    source_table: str
    target_table: str
    column: str
    options: List[Dict[str, str]]
    original_query: str


@dataclass
class IntentState:
    """
    Session-scoped accumulated intent state.

    CRITICAL FIELDS (must be known to proceed):
    - entity_type: What are we querying?
    - metric: How are we measuring?

    Everything else has safe defaults.

    v6.1.1: Added relational_ambiguity for RCL clarification loop.
    v6.10: Added row_select flag for MVP row-selection queries.
    """
    entity_type: Optional[str] = None
    metric: Optional[str] = None
    event: Optional[str] = None
    aggregation: Optional[str] = None
    ranking: Optional[str] = None
    n: Optional[int] = None
    time_scope: Optional[str] = None
    filter_conditions: List[str] = field(default_factory=list)

    # Turn tracking
    turn_count: int = 0
    original_query: Optional[str] = None

    # Relational ambiguity tracking (v6.1.1)
    pending_ambiguity: Optional[PendingAmbiguity] = None
    resolved_fk_preferences: Dict[str, str] = field(default_factory=dict)

    # Row selection flag (v6.10)
    # True when query is a simple row-selection (no aggregation needed)
    row_select: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "metric": self.metric,
            "event": self.event,
            "aggregation": self.aggregation,
            "ranking": self.ranking,
            "n": self.n,
            "time_scope": self.time_scope,
            "filter_conditions": self.filter_conditions,
            "turn_count": self.turn_count,
            "original_query": self.original_query,
            "pending_ambiguity": self.pending_ambiguity.__dict__ if self.pending_ambiguity else None,
            "resolved_fk_preferences": self.resolved_fk_preferences,
            "row_select": self.row_select,
        }

    def has_entity(self) -> bool:
        """Is entity known?"""
        return self.entity_type is not None and self.entity_type != "unknown"

    def has_metric(self) -> bool:
        """
        Is metric known AND complete?

        v6.2 CHANGE: Bare aggregation primitives (count, sum, etc.) are NOT
        complete metrics when used with rankings like "top passengers".
        They must be bound to a domain column to be considered complete.

        v6.4 CHANGE: Invalid metric tokens (list, show, get, etc.) are NOT
        metrics at all. They are action verbs that describe the operation,
        not how to measure. These MUST trigger clarification.

        v6.8 CHANGE: Bare aggregation + event = COMPLETE.
        "top passengers by booking count" has metric=count AND event=bookings,
        which together form a complete metric. This should NOT clarify.

        RULE:
        - metric in INVALID_METRIC_TOKENS -> INVALID (not a metric at all)
        - metric in BARE_AGGREGATION_PRIMITIVES AND ranking AND NO event -> INCOMPLETE
        - metric in BARE_AGGREGATION_PRIMITIVES AND event -> COMPLETE (bound to event)
        - metric has domain binding (e.g., "booking_value", "flight_count") -> COMPLETE
        - explicit aggregation field set -> COMPLETE (user clarified)
        """
        metric_val = self.metric
        if metric_val is None or metric_val == "unknown":
            # Check aggregation as fallback
            if self.aggregation is not None and self.aggregation not in (None, "none"):
                return True
            return False

        metric_lower = metric_val.lower().strip()

        # v6.4: Invalid metric tokens are NOT metrics
        # "list", "show", "get", etc. are action verbs, not measurement dimensions.
        # "Who are the best customers?" with metric=list is INVALID because
        # "list" says nothing about what makes a customer "best".
        if metric_lower in INVALID_METRIC_TOKENS:
            logger.debug(f"[INTENT] Invalid metric token rejected: '{metric_lower}'")
            return False

        # v6.2 + v6.8: Bare aggregation primitives with ranking
        # "top passengers" with metric=count is INCOMPLETE (count of what?)
        # BUT "top passengers by booking count" is COMPLETE (count bound to bookings)
        if metric_lower in BARE_AGGREGATION_PRIMITIVES:
            if self.ranking in ("top_n", "bottom_n"):
                # v6.8: Check if event provides the binding
                # If event is set (e.g., "bookings"), the aggregation is bound
                if self.event and self.event.lower() not in ("unknown", "none", ""):
                    logger.debug(
                        f"[INTENT] Bare aggregation '{metric_lower}' bound to event "
                        f"'{self.event}' - COMPLETE"
                    )
                    return True

                # Bare primitive + ranking + no event = incomplete
                logger.debug(
                    f"[INTENT] Bare aggregation '{metric_lower}' with ranking but no event - INCOMPLETE"
                )
                return False

        return True

    def is_row_select_query(self) -> bool:
        """
        Detect if this is a simple row-selection query (MVP).

        v6.10: Row-selection queries are queries that:
        - Have a known entity (what table to query)
        - Use a row-select token (list, show, get, etc.) as "metric"
        - Do NOT have a ranking (top/bottom) which would require ordering

        These are valid queries that should PROCEED without metric clarification.

        Examples:
            ✅ "list flights" -> entity=flight, metric=list, no ranking
            ✅ "show 50 passengers" -> entity=passenger, metric=show, no ranking
            ❌ "list top flights" -> has ranking, needs metric for ordering
            ❌ "show best customers" -> implies ranking, needs metric

        RULE: entity known AND metric in ROW_SELECT_TOKENS AND no ranking -> True
        """
        # Must have entity
        if not self.has_entity():
            return False

        # Must have a row-select token as metric
        metric_val = self.metric
        if metric_val is None or metric_val == "unknown":
            return False

        metric_lower = metric_val.lower().strip()
        if metric_lower not in ROW_SELECT_TOKENS:
            return False

        # Must NOT have ranking (ranking requires an ordering metric)
        if self.ranking in ("top_n", "bottom_n"):
            logger.debug(
                f"[INTENT] Row-select blocked: ranking={self.ranking} requires metric"
            )
            return False

        # Check original query for ranking keywords that extractor might miss
        if self.original_query:
            query_lower = self.original_query.lower()
            ranking_keywords = ["top", "bottom", "best", "worst", "highest", "lowest"]
            for kw in ranking_keywords:
                if kw in query_lower:
                    logger.debug(
                        f"[INTENT] Row-select blocked: ranking keyword '{kw}' in query"
                    )
                    return False

            # Also check for comparison keywords (should go through comparison handler)
            for kw in COMPARISON_KEYWORDS:
                if kw in query_lower:
                    logger.debug(
                        f"[INTENT] Row-select blocked: comparison keyword '{kw}' in query"
                    )
                    return False

        logger.info(
            f"[INTENT] Row-select query detected: entity={self.entity_type}, "
            f"metric={metric_val}"
        )
        return True

    def can_proceed(self) -> bool:
        """
        THE ONLY DECISION RULE.

        entity known AND metric known AND not comparison AND not ambiguous entity -> proceed

        v6.2: Added comparison guardrail check.
        v6.7: Added ambiguous entity check (BUG 4 fix).
        v6.10: Added row-select query support (MVP).
        """
        # v6.10: Row-select queries can proceed even without traditional metric
        if self.is_row_select_query():
            self.row_select = True  # Mark for downstream processing
            logger.info("[INTENT] Row-select query -> can_proceed=True")
            return True

        return (
            self.has_entity() and
            self.has_metric() and
            not self.is_comparison_query() and
            not self.is_ambiguous_entity()
        )

    def is_comparison_query(self) -> bool:
        """
        Detect if this is a comparison query that requires clarification.

        v6.2: Comparison queries (vs, compare, etc.) imply:
        - Segmentation logic (frequent vs non-frequent)
        - Multiple joins
        - Undefined "behavior" metrics

        These MUST be blocked or clarified, NEVER executed blindly.
        """
        # Check metric field
        if self.metric and self.metric.lower() in ("comparison", "compare"):
            return True

        # Check original query for comparison keywords
        if self.original_query:
            query_lower = self.original_query.lower()
            for keyword in COMPARISON_KEYWORDS:
                if keyword in query_lower:
                    logger.info(f"[INTENT] Comparison keyword detected: '{keyword}'")
                    return True

        return False

    def is_ambiguous_entity(self) -> bool:
        """
        Check if entity_type is an ambiguous semantic alias.

        v6.7: BUG 4 FIX - Semantic alias drift detection.

        "customer" can mean passenger, account, or frequent_flyer.
        These MUST be clarified, NEVER silently inferred.

        Returns True if entity matches an ambiguous alias.
        """
        if not self.entity_type:
            return False

        entity_lower = self.entity_type.lower().strip()

        if entity_lower in AMBIGUOUS_SEMANTIC_ALIASES:
            logger.debug(
                f"[INTENT] Ambiguous entity detected: '{entity_lower}' "
                f"-> maps to {AMBIGUOUS_SEMANTIC_ALIASES[entity_lower]['options']}"
            )
            return True

        return False

    def get_ambiguous_entity_info(self) -> Optional[Dict[str, Any]]:
        """
        Get clarification info for an ambiguous entity.

        v6.7: Returns the disambiguation options for the current entity.
        """
        if not self.entity_type:
            return None

        entity_lower = self.entity_type.lower().strip()

        if entity_lower in AMBIGUOUS_SEMANTIC_ALIASES:
            return AMBIGUOUS_SEMANTIC_ALIASES[entity_lower]

        return None

    def get_missing_field(self) -> Optional[str]:
        """Get the first missing critical field."""
        if not self.has_entity():
            return "entity_type"
        if self.is_ambiguous_entity():
            return "entity_disambiguation"
        if not self.has_metric():
            return "metric"
        return None

    def apply_defaults(self) -> "IntentState":
        """Apply safe defaults for non-critical fields."""
        if self.time_scope is None or self.time_scope == "unknown":
            self.time_scope = SAFE_DEFAULTS["time_scope"]
        if self.ranking in ("top_n", "bottom_n") and self.n is None:
            self.n = SAFE_DEFAULTS["n"]
        return self

    def clear(self, preserve_fk_preferences: bool = True) -> "IntentState":
        """
        Reset state.

        Args:
            preserve_fk_preferences: If True (default), keep resolved FK preferences
                                     across queries to maintain consistency in
                                     multi-turn conversations.

        CRITICAL (v6.1.1): FK preferences are preserved by default so that
        follow-up queries like "What about the top 3?" continue to use the
        same FK resolution without re-asking.
        """
        self.entity_type = None
        self.metric = None
        self.event = None
        self.aggregation = None
        self.ranking = None
        self.n = None
        self.time_scope = None
        self.filter_conditions = []
        self.turn_count = 0
        self.original_query = None
        self.pending_ambiguity = None
        self.row_select = False  # v6.10: Reset row-select flag

        # Preserve FK preferences unless explicitly clearing everything
        if not preserve_fk_preferences:
            self.resolved_fk_preferences = {}
            logger.info("[INTENT] State cleared (including FK preferences)")
        else:
            logger.info(
                f"[INTENT] State cleared (FK preferences preserved: "
                f"{self.resolved_fk_preferences})"
            )
        return self

    def has_pending_ambiguity(self) -> bool:
        """Check if there's a pending relational ambiguity."""
        return self.pending_ambiguity is not None

    def set_pending_ambiguity(
        self,
        source_table: str,
        target_table: str,
        column: str,
        options: List[Dict[str, str]],
        original_query: str
    ) -> None:
        """
        Set a pending relational ambiguity for clarification.

        Called when RCL detects multiple FK paths.
        """
        self.pending_ambiguity = PendingAmbiguity(
            source_table=source_table,
            target_table=target_table,
            column=column,
            options=options,
            original_query=original_query
        )
        logger.info(
            f"[CLARIFICATION_REQUESTED] Relational ambiguity set: "
            f"{source_table} -> {target_table} via {[o['fk_column'] for o in options]}"
        )

    def resolve_ambiguity(self, chosen_fk: str) -> bool:
        """
        Resolve the pending ambiguity with user's choice.

        Args:
            chosen_fk: The FK column chosen by user

        Returns:
            True if resolution was successful
        """
        if not self.pending_ambiguity:
            logger.warning("[INTENT] No pending ambiguity to resolve")
            return False

        # Validate choice against options
        valid_fks = [opt["fk_column"] for opt in self.pending_ambiguity.options]
        if chosen_fk not in valid_fks:
            logger.warning(f"[INTENT] Invalid FK choice: {chosen_fk} not in {valid_fks}")
            return False

        # Store resolution
        self.resolved_fk_preferences[self.pending_ambiguity.source_table] = chosen_fk

        logger.info(
            f"[AMBIGUITY_RESOLVED] {self.pending_ambiguity.source_table} -> {chosen_fk}"
        )

        # Clear pending (but keep resolution)
        self.pending_ambiguity = None
        return True

    def get_resolved_fk_preferences(self) -> Dict[str, str]:
        """Get resolved FK preferences for RCL."""
        return self.resolved_fk_preferences.copy()


# =============================================================================
# NEW QUERY DETECTION (Robust)
# =============================================================================

NEW_QUERY_STARTS = [
    "show", "list", "find", "get", "what", "which", "who", "where",
    "how many", "give me", "tell me", "display", "fetch", "select",
    "can you", "could you", "i want", "i need", "please",
]

# Short fragments that are continuations (answering clarification)
CONTINUATION_PATTERNS = [
    "by ", "using ", "with ", "for ", "per ",
    "arrivals", "departures", "count", "total", "average",
    "top ", "bottom ", "first ", "last ",
    "yes", "no", "correct", "right",
]


def is_new_query(query: str, has_prior: bool) -> bool:
    """
    Detect if this is a new query or a continuation.

    Rules:
    - No prior intent -> always new
    - Short (≤3 words) matching continuation patterns -> continuation
    - Long queries (>4 words) -> likely new query (even with typos)
    - Starts with command word -> new query
    """
    if not has_prior:
        return True

    q = query.lower().strip()
    words = q.split()
    word_count = len(words)

    # Very short (1-3 words) - check if it looks like a continuation
    if word_count <= 3:
        # Check if it matches continuation patterns
        for pattern in CONTINUATION_PATTERNS:
            if q.startswith(pattern) or q == pattern.strip():
                logger.debug(f"[INTENT] Short continuation: {q}")
                return False

        # Short but doesn't match continuation -> check command words
        for start in NEW_QUERY_STARTS:
            if q.startswith(start):
                return True

        # Short, no pattern match -> assume continuation
        return False

    # Longer queries (>4 words) - check for command words
    for start in NEW_QUERY_STARTS:
        if q.startswith(start):
            return True

    # Typo tolerance for common command words
    first_word = words[0] if words else ""
    typo_matches = {
        "waht": "what", "wht": "what", "whta": "what",
        "shwo": "show", "sohw": "show", "hsow": "show",
        "lsit": "list", "litst": "list",
        "fnd": "find", "fidn": "find",
    }
    if first_word in typo_matches:
        logger.info(f"[INTENT] Typo corrected: '{first_word}' -> '{typo_matches[first_word]}' -> new query")
        return True

    # Long query (5+ words) without command word -> treat as new query anyway
    # Users don't type 5+ words to answer a clarification question
    if word_count >= 5:
        logger.info(f"[INTENT] Long query ({word_count} words) -> treating as new query")
        return True

    return False


# =============================================================================
# INTENT MERGER
# =============================================================================

class IntentMerger:
    """
    Merges extracted intent into accumulated state.

    v6.1: Simplified. No confidence math.
    """

    def merge(
        self,
        accumulated: IntentState,
        new_intent: Any,
        user_message: str
    ) -> IntentState:
        """Merge new intent into accumulated state."""
        # === DEBUG HEARTBEAT ===
        print(f"[ACCUMULATOR] IntentMerger.merge() invoked: {user_message[:50]}...", flush=True)
        logger.info(f"[ACCUMULATOR] IntentMerger.merge() invoked: {user_message[:50]}...")
        # === END DEBUG HEARTBEAT ===

        # =====================================================================
        # v6.9: BUG 3 FIX - Track can_proceed() state BEFORE merge
        # =====================================================================
        # We need to detect when can_proceed() transitions from False to True.
        # When this transition happens, original_query must be set to the
        # current user message to prevent drift.
        # =====================================================================
        could_proceed_before = accumulated.can_proceed()

        # Check if new query
        if is_new_query(user_message, accumulated.turn_count > 0):
            if accumulated.turn_count > 0:
                logger.info("[INTENT] New query - clearing state (preserving FK preferences)")
                # Clear intent fields but PRESERVE FK preferences (v6.1.1)
                # FK preferences persist across queries so follow-ups like
                # "What about the top 3?" use the same FK resolution.
                # Preferences are only overwritten when user provides new clarification.
                accumulated.clear(preserve_fk_preferences=True)
            accumulated.original_query = user_message

        # Store original query if not set
        if accumulated.original_query is None:
            accumulated.original_query = user_message

        accumulated.turn_count += 1

        # =====================================================================
        # v6.6: ENTITY OVERRIDE FIX
        # =====================================================================
        # BUG: _fill() only fills gaps, so an explicit entity in the new query
        # was ignored if accumulated already had an entity. This caused
        # "top flights" to incorrectly use entity=customer from a previous query.
        #
        # FIX: If current extraction has an explicit entity, ALWAYS override.
        # When entity changes, also clear entity-dependent state (metric).
        #
        # RULE:
        #   - new_entity specified -> OVERRIDE accumulated entity
        #   - new_entity not specified -> KEEP accumulated entity (continuation)
        # =====================================================================
        new_entity = self._get_value(new_intent, "entity_type")

        if new_entity is not None:
            old_entity = accumulated.entity_type
            if old_entity is not None and old_entity != new_entity:
                logger.debug(
                    f"[INTENT] Entity override: '{old_entity}' -> '{new_entity}' "
                    f"(clearing stale metric and updating NL query)"
                )
                # Clear metric when entity changes - metric may be entity-specific
                accumulated.metric = None
                # =========================================================
                # v6.7: BUG 1 FIX - Update NL query on entity override
                # =========================================================
                # When entity changes, the stored original_query becomes stale.
                # If can_proceed() returns True, execution would use the OLD
                # query text. Fix: update original_query to current message.
                # =========================================================
                accumulated.original_query = user_message
                logger.debug(f"[INTENT] NL query updated to: '{user_message[:50]}...'")
            accumulated.entity_type = new_entity

        # =====================================================================
        # v6.6: METRIC OVERRIDE
        # =====================================================================
        # Similarly, if extraction found an explicit metric, use it.
        # This allows "by booking count" to set metric during clarification.
        # =====================================================================
        new_metric = self._get_value(new_intent, "metric")

        if new_metric is not None:
            if accumulated.metric != new_metric:
                logger.debug(f"[INTENT] metric: '{accumulated.metric}' -> '{new_metric}'")
            accumulated.metric = new_metric

        # Fill gaps for remaining fields (non-override behavior is fine)
        self._fill(accumulated, new_intent, "event")
        self._fill(accumulated, new_intent, "aggregation")
        self._fill(accumulated, new_intent, "ranking")
        self._fill(accumulated, new_intent, "n")
        self._fill(accumulated, new_intent, "time_scope")

        # Merge filters
        if hasattr(new_intent, 'filter_conditions') and new_intent.filter_conditions:
            for f in new_intent.filter_conditions:
                if f not in accumulated.filter_conditions:
                    accumulated.filter_conditions.append(f)

        # =====================================================================
        # v6.9: BUG 3 FIX - Update original_query on can_proceed() transition
        # =====================================================================
        # When can_proceed() transitions from False to True, the user has
        # completed their query (via clarification or new input). At this
        # point, original_query MUST reflect the current user message to
        # prevent drift where NL-SQL executes a stale query.
        #
        # RULE: If can_proceed() transitions False -> True, set original_query
        #       to current user message.
        #
        # This happens REGARDLESS of:
        # - Whether entity changed
        # - Whether metric was resolved
        # - How clarification was completed
        #
        # ARCHITECTURAL NOTE: This fix lives entirely in IntentAccumulator,
        # preserving QueryPipeline as orchestration-only.
        # =====================================================================
        can_proceed_now = accumulated.can_proceed()

        if not could_proceed_before and can_proceed_now:
            # Transition detected: False -> True
            accumulated.original_query = user_message
            logger.info(
                f"[INTENT] can_proceed transition (False->True): "
                f"original_query = '{user_message[:50]}...'"
            )

        logger.info(
            f"[INTENT] After merge: entity={accumulated.entity_type}, "
            f"metric={accumulated.metric}, can_proceed={can_proceed_now}"
        )

        return accumulated

    def _get_value(self, intent: Any, field: str) -> Optional[Any]:
        """
        Extract a field value, normalizing 'unknown' to None.

        v6.6: Helper for override logic.
        """
        if not hasattr(intent, field):
            return None
        val = getattr(intent, field, None)
        if val == "unknown":
            return None
        return val

    def _fill(self, acc: IntentState, new: Any, field: str):
        """Fill a gap (don't overwrite known values)."""
        acc_val = getattr(acc, field, None)
        if acc_val == "unknown":
            acc_val = None

        new_val = getattr(new, field, None) if hasattr(new, field) else None
        if new_val == "unknown":
            new_val = None

        if acc_val is None and new_val is not None:
            setattr(acc, field, new_val)
            logger.debug(f"[INTENT] {field}: -> {new_val}")


# =============================================================================
# CLARIFICATION REASONS (v6.5 - Explanation-Aware Clarifications)
# =============================================================================
# Structured reasons for why clarification is required.
# These are deterministic, auditable, and schema-derived where applicable.
#
# INVARIANT: Reasons explain WHY, but do NOT affect decision logic.
# =============================================================================

class ClarificationReason:
    """
    Enumeration of clarification reasons.

    Each reason maps to a specific ambiguity type that triggered clarification.
    Used for structured logging and user-facing explanations.
    """
    MISSING_ENTITY = "missing_entity"
    MISSING_METRIC = "missing_metric"
    INVALID_METRIC_TOKEN = "invalid_metric_token"
    BARE_AGGREGATION = "bare_aggregation"
    RELATIONAL_AMBIGUITY = "relational_ambiguity"
    COMPARISON_QUERY = "comparison_query"
    ENTITY_REBINDING = "entity_rebinding"


@dataclass
class ClarificationContext:
    """
    Structured context for why clarification was triggered.

    v6.5: Explanation-aware clarifications.

    This is ADVISORY metadata - it does NOT affect decision logic.
    It explains to users WHY the system cannot proceed.

    Attributes:
        reason: ClarificationReason category
        explanation: Human-readable explanation of the ambiguity
        detected_value: The problematic value that triggered clarification (if any)
        entity_context: Dict with original/rebound entity info (if applicable)
        schema_source: Where the options came from (table/column names)
    """
    reason: str
    explanation: str
    detected_value: Optional[str] = None
    entity_context: Optional[Dict[str, str]] = None
    schema_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "explanation": self.explanation,
            "detected_value": self.detected_value,
            "entity_context": self.entity_context,
            "schema_source": self.schema_source,
        }


# =============================================================================
# EVALUATION (SOLE CLARIFICATION DECISION)
# =============================================================================

@dataclass
class IntentDecision:
    """
    Result of intent evaluation.

    v6.5: Added context field for explanation-aware clarifications.
    """
    proceed: bool
    clarification: Optional[str] = None
    options: List[str] = field(default_factory=list)
    # v6.5: Structured explanation context (does NOT affect decision logic)
    context: Optional[ClarificationContext] = None


def evaluate(accumulated: IntentState) -> IntentDecision:
    """
    THE SOLE CLARIFICATION DECISION POINT.

    Rule: entity AND metric known AND not comparison -> proceed

    v6.2: Added comparison guardrail and bare-metric clarification.
    v6.5: Added explanation-aware clarification context.
    """
    accumulated.apply_defaults()

    # v6.2: Check for comparison queries FIRST (hard block)
    if accumulated.is_comparison_query():
        logger.info("[INTENT] [X] BLOCK (comparison query detected)")
        return IntentDecision(
            proceed=False,
            clarification=(
                "Comparison queries require more specific criteria.\n\n"
                "Please clarify:\n"
                "- **What segments** are you comparing? (e.g., VIP vs regular)\n"
                "- **What metric** should be compared? (e.g., booking count, revenue)\n"
                "- **What time period**? (e.g., last month, last year)"
            ),
            options=[
                "Compare by booking count",
                "Compare by revenue",
                "Compare by flight count",
            ],
            context=ClarificationContext(
                reason=ClarificationReason.COMPARISON_QUERY,
                explanation="Comparison queries imply segmentation and undefined metrics",
                detected_value=accumulated.original_query,
            )
        )

    if accumulated.can_proceed():
        if accumulated.row_select:
            logger.info("[INTENT] [OK] PROCEED (row-select: entity known, no aggregation)")
        else:
            logger.info("[INTENT] [OK] PROCEED (entity + metric known)")
        return IntentDecision(proceed=True)

    # Generate clarification for missing field (with context)
    missing = accumulated.get_missing_field()
    question, options, context = _make_clarification_with_context(accumulated, missing)

    logger.debug(
        f"[INTENT] Clarification triggered: reason={context.reason}, "
        f"detected_value={context.detected_value}"
    )
    logger.info(f"[INTENT] [X] CLARIFY (missing: {missing})")

    return IntentDecision(
        proceed=False,
        clarification=question,
        options=options,
        context=context
    )


def _make_clarification_with_context(
    state: IntentState,
    missing: str
) -> Tuple[str, List[str], ClarificationContext]:
    """
    Generate clarification question WITH structured context.

    v6.5: Explanation-aware clarifications.

    Returns:
        Tuple of (question, options, context)

    INVARIANT: This function suggests, but the user MUST choose.
    The system does NOT auto-select a default metric.
    """

    if missing == "entity_type":
        return (
            "What would you like to query?",
            ["Airports", "Flights", "Passengers", "Bookings", "Routes"],
            ClarificationContext(
                reason=ClarificationReason.MISSING_ENTITY,
                explanation="No entity type was detected in your query",
            )
        )

    # =========================================================================
    # v6.7: BUG 4 FIX - Ambiguous Entity Clarification
    # =========================================================================
    # "customer" maps to multiple tables (passenger, account, frequent_flyer).
    # We MUST clarify, NEVER silently infer.
    # =========================================================================
    if missing == "entity_disambiguation":
        ambig_info = state.get_ambiguous_entity_info()
        if ambig_info:
            entity_val = state.entity_type
            question = (
                f"**Why I'm asking:** '{entity_val}' can refer to multiple "
                f"things in this database.\n\n"
                f"{ambig_info['question']}"
            )
            options = [
                f"{opt.title()} - {ambig_info['descriptions'].get(opt, opt)}"
                for opt in ambig_info['options']
            ]
            return (
                question,
                options,
                ClarificationContext(
                    reason=ClarificationReason.ENTITY_REBINDING,
                    explanation=f"'{entity_val}' maps to multiple base tables",
                    detected_value=entity_val,
                    entity_context={
                        "ambiguous_alias": entity_val,
                        "options": ambig_info['options'],
                    }
                )
            )

    if missing == "metric":
        entity = state.entity_type or "data"
        metric_val = state.metric

        # Determine the specific reason for metric clarification
        reason, explanation, detected = _classify_metric_ambiguity(state)

        # Build the clarification question with explanation
        if state.ranking in ("top_n", "bottom_n"):
            rank = "top" if state.ranking == "top_n" else "bottom"
            n = state.n or 10

            if reason == ClarificationReason.INVALID_METRIC_TOKEN:
                # v6.5: Explain that the token is not a metric
                q = (
                    f"**Why I'm asking:** '{metric_val}' describes an action, "
                    f"not how to measure '{entity}s'.\n\n"
                    f"What determines the {rank} {n} {entity}s?"
                )
            elif reason == ClarificationReason.BARE_AGGREGATION:
                # v6.5: Explain that the aggregation needs a target
                q = (
                    f"**Why I'm asking:** '{metric_val}' is ambiguous — "
                    f"{metric_val} of what?\n\n"
                    f"What dimension determines the {rank} {n} {entity}s?"
                )
            else:
                q = (
                    f"**Why I'm asking:** No ranking metric was specified.\n\n"
                    f"How should I rank the {rank} {n} {entity}s?"
                )
        else:
            if reason == ClarificationReason.INVALID_METRIC_TOKEN:
                q = (
                    f"**Why I'm asking:** '{metric_val}' describes an action, "
                    f"not a measurement.\n\n"
                    f"What would you like to know about {entity}s?"
                )
            else:
                q = (
                    f"**Why I'm asking:** No metric was specified.\n\n"
                    f"What would you like to know about {entity}s?"
                )

        # Build context
        context = ClarificationContext(
            reason=reason,
            explanation=explanation,
            detected_value=detected,
        )

        # =====================================================================
        # v6.3: Try schema-backed suggestions FIRST
        # =====================================================================
        schema_suggestions = suggest_metrics_for_entity(entity)

        if schema_suggestions:
            opts = [s.to_option_string() for s in schema_suggestions]
            q += "\n\n**Suggested metrics based on your database:**"
            context.schema_source = f"table:{_find_table_for_entity(entity, get_schema_reference())}"
            logger.info(
                f"[ACCUMULATOR] Using schema-backed suggestions for {entity}: "
                f"{len(opts)} options"
            )
            return q, opts, context

        # =====================================================================
        # Fallback: Generic context-aware options (when no schema available)
        # =====================================================================
        logger.debug(f"[ACCUMULATOR] No schema suggestions for {entity} - using defaults")

        if "airport" in (entity or "").lower():
            opts = ["By flight count", "By total passengers", "By departure count", "By arrival count"]
        elif "flight" in (entity or "").lower():
            opts = ["By passenger count", "By revenue", "By duration", "By distance"]
        elif "customer" in (entity or "").lower() or "passenger" in (entity or "").lower():
            opts = ["By total booking value", "By booking count", "By miles flown", "By loyalty points"]
        elif "route" in (entity or "").lower():
            opts = ["By passenger count", "By flight frequency", "By revenue", "By distance"]
        else:
            opts = ["By count", "By total value", "By average", "List all"]

        return q, opts, context

    # Fallback for unknown missing field
    return (
        "Could you clarify what you're looking for?",
        [],
        ClarificationContext(
            reason=ClarificationReason.MISSING_METRIC,
            explanation="Unable to determine query intent",
        )
    )


def _classify_metric_ambiguity(state: IntentState) -> Tuple[str, str, Optional[str]]:
    """
    Classify WHY the metric is considered invalid/incomplete.

    v6.5: Helper for explanation-aware clarifications.

    Returns:
        Tuple of (reason, explanation, detected_value)

    INVARIANT: This function classifies, it does NOT decide.
    Decision logic remains in has_metric() / can_proceed().
    """
    metric_val = state.metric

    if metric_val is None or metric_val == "unknown":
        return (
            ClarificationReason.MISSING_METRIC,
            "No metric was extracted from the query",
            None
        )

    metric_lower = metric_val.lower().strip()

    # Check invalid metric tokens (action verbs)
    if metric_lower in INVALID_METRIC_TOKENS:
        logger.debug(f"[INTENT] Metric ambiguity: invalid token '{metric_lower}'")
        return (
            ClarificationReason.INVALID_METRIC_TOKEN,
            f"'{metric_val}' is an action verb, not a measurement dimension",
            metric_val
        )

    # Check bare aggregation primitives with ranking
    if metric_lower in BARE_AGGREGATION_PRIMITIVES:
        if state.ranking in ("top_n", "bottom_n"):
            logger.debug(f"[INTENT] Metric ambiguity: bare aggregation '{metric_lower}' with ranking")
            return (
                ClarificationReason.BARE_AGGREGATION,
                f"'{metric_val}' needs a target — {metric_val} of what?",
                metric_val
            )

    # Default: missing metric
    return (
        ClarificationReason.MISSING_METRIC,
        "No complete metric was specified",
        metric_val
    )


def _make_clarification(state: IntentState, missing: str) -> tuple:
    """
    Generate clarification question.

    DEPRECATED: Use _make_clarification_with_context() for new code.
    Kept for backward compatibility.
    """
    q, opts, _ = _make_clarification_with_context(state, missing)
    return q, opts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_intent_state() -> IntentState:
    """Create fresh intent state."""
    return IntentState()


def create_intent_merger() -> IntentMerger:
    """Create intent merger."""
    return IntentMerger()


def get_query_for_nlsql(accumulated: IntentState, current: str) -> str:
    """
    Get query for NL-SQL.

    INVARIANT: NL-SQL receives ORIGINAL query, not fragments.

    RULE:
    - Turn 1: Use current (no prior context)
    - Turn 2+: Use original_query (managed by IntentMerger)

    NOTE: The BUG 3 fix in IntentMerger.merge() updates original_query when
    can_proceed() transitions to True. This ensures NL-SQL always receives
    the correct query text.
    """
    if accumulated.original_query and accumulated.turn_count > 1:
        logger.info(f"[NL-SQL] Using original: {accumulated.original_query[:50]}...")
        return accumulated.original_query
    return current


def format_clarification(decision: IntentDecision) -> str:
    """Format clarification for display."""
    if decision.proceed:
        return ""

    # v6.5: If context is present, the clarification text is already formatted
    # with "**Why I'm asking:**" explanation. Don't wrap in additional bold.
    if decision.context is not None:
        lines = [decision.clarification, ""]
    else:
        # Legacy path: wrap in bold for backward compatibility
        lines = [f"**{decision.clarification}**", ""]

    if decision.options:
        for i, opt in enumerate(decision.options, 1):
            lines.append(f"{i}. {opt}")

    return "\n".join(lines)


# =============================================================================
# RELATIONAL AMBIGUITY HANDLING (v6.1.1)
# =============================================================================

def generate_ambiguity_clarification(ambiguity: PendingAmbiguity) -> IntentDecision:
    """
    Generate a clarification question from a relational ambiguity.

    v6.5: Enhanced with explanation-aware context.

    This is called when RCL detects multiple FK paths.
    The question is derived DETERMINISTICALLY from the ambiguity structure.
    """
    column = ambiguity.column
    source = ambiguity.source_table.split(".")[-1]
    target = ambiguity.target_table.split(".")[-1]

    # v6.5: Build explanation-aware question
    # Explain WHY there's ambiguity and WHERE it comes from
    fk_names = [opt["fk_column"] for opt in ambiguity.options]

    question = (
        f"**Why I'm asking:** '{column}' exists on `{target}`, but `{source}` "
        f"connects to `{target}` via multiple relationships: {', '.join(fk_names)}.\n\n"
        f"Which relationship do you mean?"
    )

    # Build options from FK metadata
    options = []
    for opt in ambiguity.options:
        options.append(opt["description"])

    # v6.5: Build structured context
    context = ClarificationContext(
        reason=ClarificationReason.RELATIONAL_AMBIGUITY,
        explanation=f"Multiple FK paths from {source} to {target}",
        detected_value=column,
        schema_source=f"{source} -> {target}",
        entity_context={
            "source_table": source,
            "target_table": target,
            "fk_options": fk_names,
        }
    )

    logger.debug(
        f"[INTENT] Relational ambiguity: {source} -> {target} via {fk_names}, "
        f"column={column}"
    )
    logger.info(f"[CLARIFICATION_REQUESTED] Question generated with {len(options)} options")

    return IntentDecision(
        proceed=False,
        clarification=question,
        options=options,
        context=context
    )


def match_ambiguity_response(
    response: str,
    pending: PendingAmbiguity
) -> Optional[str]:
    """
    Match user response to an ambiguity option.

    Returns the fk_column if matched, None if unclear.

    Matching rules (deterministic):
    1. Exact match on fk_column name
    2. Numeric choice (1, 2, etc.)
    3. Keyword match (arrival, departure, etc.)
    """
    response_lower = response.lower().strip()
    options = pending.options

    # Rule 1: Numeric choice
    if response_lower.isdigit():
        idx = int(response_lower) - 1  # 1-indexed
        if 0 <= idx < len(options):
            chosen = options[idx]["fk_column"]
            logger.info(f"[INTENT] Matched by number: {response} -> {chosen}")
            return chosen

    # Rule 2: Exact FK column match
    for opt in options:
        if opt["fk_column"].lower() == response_lower:
            logger.info(f"[INTENT] Matched by exact FK: {response} -> {opt['fk_column']}")
            return opt["fk_column"]

    # Rule 3: Keyword matching (derived from FK name patterns)
    keyword_map = {
        "arrival": ["arrival", "arriving", "inbound", "destination", "to"],
        "departure": ["departure", "departing", "outbound", "origin", "from", "leaving"],
    }

    for opt in options:
        fk_lower = opt["fk_column"].lower()

        # Check which category this FK belongs to
        for category, keywords in keyword_map.items():
            if any(kw in fk_lower for kw in keywords):
                # Check if response matches this category
                if any(kw in response_lower for kw in keywords):
                    logger.info(f"[INTENT] Matched by keyword '{category}': {response} -> {opt['fk_column']}")
                    return opt["fk_column"]

    # Rule 4: Partial match on description
    for opt in options:
        desc_lower = opt.get("description", "").lower()
        # Check if significant words from response appear in description
        response_words = set(response_lower.split())
        desc_words = set(desc_lower.split())
        overlap = response_words & desc_words
        if len(overlap) >= 1 and len(overlap) > len(response_words) * 0.3:
            logger.info(f"[INTENT] Matched by description overlap: {response} -> {opt['fk_column']}")
            return opt["fk_column"]

    logger.warning(f"[INTENT] Could not match response: '{response}'")
    return None


def format_ambiguity_clarification(pending: PendingAmbiguity) -> str:
    """
    Format ambiguity clarification for display.

    v6.5: Enhanced with explanation of why clarification is needed.

    Returns a user-friendly message asking for clarification.
    """
    source = pending.source_table.split(".")[-1]
    target = pending.target_table.split(".")[-1]
    fk_names = [opt["fk_column"] for opt in pending.options]

    lines = [
        f"**Why I'm asking:** '{pending.column}' exists on `{target}`, "
        f"but `{source}` connects to `{target}` via multiple relationships.",
        "",
        f"The available paths are: {', '.join(fk_names)}",
        "",
        "Which relationship do you mean?",
        ""
    ]

    for i, opt in enumerate(pending.options, 1):
        lines.append(f"{i}. {opt['description']}")

    return "\n".join(lines)
