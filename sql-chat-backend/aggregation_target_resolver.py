"""
Aggregation Target Resolver — OptimaX NL→SQL Compiler Phase

Deterministic, schema-driven resolution of the aggregation target attribute
for bare aggregation metrics (avg, sum, min, max, total, minimum, maximum).

PURPOSE
-------
When the LLM extractor emits:

    entity_type = "passenger"
    metric      = "average"
    event       = None

the pipeline lacks the target column for the aggregation (i.e., "average of
WHAT?").  This module extracts that column reference from the original NL
query and resolves it to a qualified schema column.

PROBLEM SOLVED
--------------
Queries like:

    "average passenger age"
    "maximum booking price"
    "sum of flight distances"

emit a metric token (average / maximum / sum) from the LLM extractor.
The target attribute ("age", "price", "distances") is not captured into any
extractor field.  Without the target the accumulator correctly marks the
intent INCOMPLETE and asks for clarification.

This module bridges that extraction gap by:
  1. Detecting the aggregation keyword in the NL query.
  2. Extracting the trailing phrase (potential entity + attribute).
  3. Routing the phrase through ALSR for schema-grounded column resolution.
  4. Returning a ResolvedAttribute for binding as IntentState.metric_target.

PIPELINE POSITION
-----------------
  Extract → Entity Resolver → **Aggregation Target Resolver** →
  IntentMerger → Evaluate → Execute

DESIGN INVARIANTS
-----------------
  - Pure functions only (deterministic, no side effects)
  - No LLM calls
  - No hardcoded domain knowledge
  - No heuristic numeric-column guessing
  - Schema-driven via ALSR (resolve_attribute_phrase)
  - Returns None if attribute cannot be resolved — accumulator clarifies
  - Does NOT modify IntentState (caller applies the binding)
  - Does NOT import from intent_accumulator (avoids circular dependency)

ACCEPTANCE TESTS
----------------
  Case 1: "Average passenger age"   → metric=average, entity=passenger
          → metric_target = passenger.age          → proceed=True

  Case 2: "Maximum booking price"   → metric=max, entity=booking
          → metric_target = booking.price          → proceed=True

  Case 3: "Average passenger"       → no attribute phrase found
          → return None → accumulator clarifies

  Case 4: "Count passengers"        → metric=count → skipped entirely

  Case 5: "Count passengers per booking" → metric=count → skipped;
          grouping handled by Phase 3 (grouping_phrase_detector)
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# AGGREGATION METRICS THAT REQUIRE A TARGET COLUMN
# =============================================================================
# These bare aggregation primitives require a column reference to form valid
# SQL. COUNT(*) is always valid as a standalone entity count and is excluded.
# =============================================================================

AGGREGATION_METRICS: frozenset = frozenset({
    "avg", "average",
    "sum", "total",
    "min", "minimum",
    "max", "maximum",
})

# Canonical aggregation keyword pattern (regex alternation).
_AGG_KEYWORD_PATTERN = r"(?:average|avg|sum|total|min(?:imum)?|max(?:imum)?)"

# Pattern: <agg_keyword> [of] <phrase>
# Captures everything after the keyword (and optional "of") as group(1).
_AGG_THEN_PHRASE_RE = re.compile(
    r"\b" + _AGG_KEYWORD_PATTERN + r"\s+(?:of\s+)?(.+)",
    re.IGNORECASE,
)

# =============================================================================
# STOP WORDS — end the aggregation target phrase
# =============================================================================
# Trim these words (and everything after) from the extracted candidate phrase.
#
#   per / by / grouped / group  — grouping keywords (handled by Phase 3)
#   where / having / limit / order — SQL clause starters
#   from / in / with             — prepositions starting a new clause
# =============================================================================

_PHRASE_STOP_RE = re.compile(
    r"\s+(?:per|by|grouped|group|where|having|limit|order|from|in|with)\b.*$",
    re.IGNORECASE,
)


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class ResolvedAttribute:
    """
    A resolved aggregation target attribute.

    Attributes:
        table:     Bare table name (e.g., "passenger")
        column:    Matched column name (e.g., "age")
        qualified: Fully qualified reference (e.g., "passenger.age")
    """
    table: str
    column: str
    qualified: str


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _trim_phrase(phrase: str) -> str:
    """
    Remove trailing stop words and their arguments from a candidate phrase.

    Examples::

        _trim_phrase("passenger age limit 10")  →  "passenger age"
        _trim_phrase("booking price per month") →  "booking price"
        _trim_phrase("flight duration")         →  "flight duration"
    """
    return _PHRASE_STOP_RE.sub("", phrase).strip()


def _extract_candidate_phrase(query: str, metric: str) -> Optional[str]:
    """
    Extract the raw attribute phrase from the NL query.

    Finds the aggregation keyword and captures the phrase that follows it
    (up to the first stop word).

    Args:
        query:  Original NL query (pre-extraction, unmodified)
        metric: Metric token emitted by the LLM extractor (for logging)

    Returns:
        Candidate phrase string, or None if pattern not matched.

    Examples::

        _extract_candidate_phrase("average passenger age",   "average") → "passenger age"
        _extract_candidate_phrase("sum of booking amount",   "sum")     → "booking amount"
        _extract_candidate_phrase("maximum booking price",   "max")     → "booking price"
        _extract_candidate_phrase("min fare",                "min")     → "fare"
        _extract_candidate_phrase("average passenger",       "average") → "passenger"
    """
    m = _AGG_THEN_PHRASE_RE.search(query)
    if not m:
        logger.debug(
            f"[ATR] No aggregation pattern found for metric='{metric}' "
            f"in query: '{query[:80]}'"
        )
        return None

    raw = m.group(1).strip()
    phrase = _trim_phrase(raw)
    if not phrase:
        logger.debug(f"[ATR] Candidate phrase empty after stop-word trim: raw='{raw}'")
        return None

    logger.debug(f"[ATR] Extracted candidate phrase: '{phrase}' (metric='{metric}')")
    return phrase


# =============================================================================
# PUBLIC API
# =============================================================================

def resolve_aggregation_target(
    original_query: str,
    metric: str,
    entity: str,
    schema_reference: Dict[str, Any],
) -> Optional[ResolvedAttribute]:
    """
    Resolve the aggregation target attribute for a bare aggregation metric.

    Routes the detected attribute phrase through ALSR for schema-grounded
    column resolution.  Returns None when the attribute cannot be resolved;
    the accumulator will then issue a CLARIFY decision and ask the user.

    ALGORITHM
    ---------
    1. Guard: metric not in AGGREGATION_METRICS → return None immediately
       (COUNT and domain metrics need no target)
    2. Extract candidate phrase via aggregation keyword pattern
    3. Route phrase through ALSR (resolve_attribute_phrase)
    4. If ALSR resolved → return ResolvedAttribute
    5. Fallback: if phrase is a single token and ALSR returned passthrough,
       compose entity + " " + phrase and retry ALSR
    6. If still unresolved → return None (accumulator clarifies)

    GUARDRAILS
    ----------
    - Does NOT auto-select numeric columns heuristically
    - Does NOT assume there is only one numeric column per entity
    - Returns None on ambiguous ALSR result → accumulator handles ambiguity
    - No silent fallbacks — every unresolved path returns None explicitly

    Args:
        original_query:   Raw NL query string (pre-extraction, unmodified)
        metric:           Metric token from LLM extractor (e.g., "average")
        entity:           Resolved entity name (e.g., "passenger")
        schema_reference: Database schema dict with "tables" key

    Returns:
        ResolvedAttribute if the attribute is schema-grounded, None otherwise.

    Examples::

        resolve_aggregation_target("Average passenger age", "average",
                                   "passenger", schema)
        → ResolvedAttribute(table="passenger", column="age",
                            qualified="passenger.age")

        resolve_aggregation_target("Maximum booking price", "max",
                                   "booking", schema)
        → ResolvedAttribute(table="booking", column="price",
                            qualified="booking.price")

        resolve_aggregation_target("Average passenger", "average",
                                   "passenger", schema)
        → None   (no distinct attribute phrase; accumulator clarifies)

        resolve_aggregation_target("Count passengers", "count",
                                   "passenger", schema)
        → None   (COUNT skipped by guard)

    UNIT VERIFICATION BLOCK
    -----------------------
    # Case 1: full entity+attribute phrase
    # "Average passenger age" → candidate="passenger age"
    #   → ALSR("passenger age") → entity_part="passenger", attr_part="age"
    #   → resolved: passenger.age ✓

    # Case 2: keyword at start, attribute phrase
    # "Maximum booking price" → candidate="booking price"
    #   → ALSR("booking price") → entity_part="booking", attr_part="price"
    #   → resolved: booking.price ✓

    # Case 3: keyword + entity only, no attribute
    # "Average passenger" → candidate="passenger"
    #   → ALSR("passenger") → passthrough (single_token)
    #   → Fallback: compose "passenger passenger" → ALSR unresolved
    #   → return None ✓  (accumulator clarifies)

    # Case 4: count metric
    # "Count passengers" → guard: "count" not in AGGREGATION_METRICS
    #   → return None immediately ✓

    # Case 5: single-token attribute with entity known
    # "Min fare" (entity="booking") → candidate="fare"
    #   → ALSR("fare") → passthrough (single_token)
    #   → Fallback: compose "booking fare"
    #   → ALSR("booking fare") → entity_part="booking", attr_part="fare"
    #   → resolved: booking.fare ✓  (if booking.fare exists in schema)
    """
    metric_lower = (metric or "").lower().strip()

    # --- Guard 1: Only applies to non-count bare aggregation primitives ---
    if metric_lower not in AGGREGATION_METRICS:
        logger.debug(
            f"[ATR] Skipped: metric='{metric_lower}' is not a target-requiring "
            f"aggregation primitive (count or domain metric)"
        )
        return None

    # --- Guard 2: Must have resolved entity ---
    if not entity or entity in ("unknown", ""):
        logger.debug("[ATR] Skipped: entity is absent or unknown")
        return None

    # --- Guard 3: Must have schema ---
    if not schema_reference or "tables" not in schema_reference:
        logger.debug("[ATR] Skipped: no schema reference available")
        return None

    # -------------------------------------------------------------------------
    # Step 1: Extract candidate phrase from query
    # -------------------------------------------------------------------------
    candidate = _extract_candidate_phrase(original_query, metric_lower)
    if not candidate:
        logger.info(
            f"[ATR] No candidate attribute phrase detected for "
            f"metric='{metric_lower}', entity='{entity}'"
        )
        return None

    # -------------------------------------------------------------------------
    # Step 2: Route candidate through ALSR
    # -------------------------------------------------------------------------
    try:
        from semantic_attribute_resolver import resolve_attribute_phrase
    except ImportError:
        logger.warning(
            "[ATR] ALSR not available — semantic_attribute_resolver not installed"
        )
        return None

    result = resolve_attribute_phrase(candidate, schema_reference)

    if result.status == "resolved" and result.binding:
        b = result.binding
        logger.info(
            f"[ATR] Resolved: '{candidate}' → {b.qualified} "
            f"(metric='{metric_lower}', entity='{entity}', "
            f"strategy={b.strategy}, conf={b.confidence})"
        )
        return ResolvedAttribute(
            table=b.table,
            column=b.column,
            qualified=b.qualified,
        )

    # -------------------------------------------------------------------------
    # Step 3: Single-token fallback — compose entity + candidate and retry
    #
    # When the candidate is a bare single token (e.g., "age", "fare", "price"),
    # ALSR returns "passthrough" (single_token reason) because it requires ≥2
    # tokens to split into entity + attribute parts.
    #
    # Compose: entity + " " + candidate (e.g., "passenger age") and re-route
    # through ALSR.  This gives ALSR the entity context needed to resolve the
    # column.  Only attempted for genuine single-word candidates (no space).
    # -------------------------------------------------------------------------
    if result.status in ("passthrough", "unresolved") and " " not in candidate:
        composed = f"{entity} {candidate}"
        logger.debug(
            f"[ATR] Single-token fallback: composing '{entity}' + '{candidate}' "
            f"→ '{composed}'"
        )
        result2 = resolve_attribute_phrase(composed, schema_reference)

        if result2.status == "resolved" and result2.binding:
            b = result2.binding
            logger.info(
                f"[ATR] Resolved (fallback): '{composed}' → {b.qualified} "
                f"(metric='{metric_lower}', entity='{entity}', "
                f"strategy={b.strategy}, conf={b.confidence})"
            )
            return ResolvedAttribute(
                table=b.table,
                column=b.column,
                qualified=b.qualified,
            )

        logger.info(
            f"[ATR] Fallback composition '{composed}' also unresolved "
            f"(status={result2.status}, reason={result2.reason}) — "
            f"returning None; accumulator will clarify"
        )
    else:
        logger.info(
            f"[ATR] Candidate '{candidate}' unresolved "
            f"(status={result.status}, reason={result.reason}) — "
            f"returning None; accumulator will clarify"
        )

    # Both attempts failed → return None, accumulator clarifies
    return None
