"""
OptimaX v6.22+ — Grouping Phrase Detector
==========================================

Deterministic structural detection of NL grouping patterns.

PURPOSE
-------
Detect grouping expressions in natural language queries and return the raw
phrase for schema-grounded resolution by ALSR.  This module has no knowledge
of the database schema; it only applies syntactic parsing rules to the query
string.

PROBLEM SOLVED
--------------
Queries like:

    count flights per aircraft model limit 10

emit a single-word entity_type ("flights") from the LLM extractor.  The
tail phrase "aircraft model" is not captured.  ALSR Phase 1 / Phase 2 only
run when the LLM collapses a composite phrase into a single extraction field.
Neither phase fires here because entity_type = "flights" (no whitespace), and
metric / event fields do not contain "aircraft model".

This module bridges that extraction gap by scanning the original NL query for
structural grouping markers.

RECOGNIZED PATTERNS
-------------------
Tried in priority order (first match wins):

    grouped by <phrase>       most explicit two-word pattern
    group by <phrase>         SQL-style embedded in NL
    per <phrase>              most common natural grouping marker
    by <phrase>               guarded — only when analytic context is present

"Analytic context" means the query contains an aggregation verb (count, sum,
total, avg, average, min, max, minimum, maximum) or a ranking pattern
("top N" or "bottom N").  This guard prevents false positives on phrases like
"filtered by airport" or "sorted by date".

STOP WORDS
----------
The grouping phrase is trimmed at the first occurrence of a stop word:
    limit, order, having

Example:
    "aircraft model limit 10"  →  "aircraft model"

PIPELINE POSITION
-----------------
Called in query_pipeline.py after semantic_intent_extractor.extract() and
before IntentMerger.merge().  The extracted phrase is routed through ALSR
(resolve_attribute_phrase) for schema-grounded column resolution.

DESIGN INVARIANTS
-----------------
- Pure function: no side effects, no state, no I/O
- Deterministic: same input always produces the same output
- No LLM calls
- No embeddings
- No hardcoded domain knowledge
- Schema-agnostic: returns the phrase as a string; ALSR handles resolution
- Conservative: returns None when uncertain; never forces a proceed decision
"""

import re
from typing import Optional


# =============================================================================
# Stop-word trimmer
# =============================================================================
# Removes a stop word and everything after it from a phrase candidate.
# Matches "\s+ <stop-word> \b .*" at the end of the phrase.
#
# Stop words:
#   limit  — trailing "LIMIT N" appended by the user
#   order  — "ORDER BY ..." tail clause
#   having — "HAVING ..." tail clause
#
# Note: matching is case-insensitive; \s+ prevents partial-word matches on
# column names that happen to start with a stop word (e.g., "ordinal").
# The \b word boundary after the stop word provides additional protection.
# =============================================================================
_PHRASE_STOP_RE = re.compile(
    r'\s+(?:limit|order|having)\b.*$',
    re.IGNORECASE,
)


# =============================================================================
# Analytic context guard
# =============================================================================
# Aggregation verbs that indicate a query is performing a calculation.
# Used to guard the bare "by <phrase>" pattern against false positives.
_AGGREGATION_VERBS: frozenset = frozenset({
    "count", "sum", "total", "avg", "average",
    "min", "max", "minimum", "maximum",
})

# Ranking pattern: "top N" or "bottom N" where N is one or more digits.
_RANKING_RE = re.compile(r'\b(?:top|bottom)\s+\d+\b', re.IGNORECASE)


# =============================================================================
# Grouping patterns — ordered by specificity (most specific first)
# =============================================================================
# Each pattern captures the raw phrase in group(1).  The patterns are
# intentionally greedy (.+) so that trimming is handled by _trim_phrase()
# rather than by the regex stop condition.
# =============================================================================
_GROUPED_BY_RE = re.compile(r'\bgrouped\s+by\s+(.+)', re.IGNORECASE)
_GROUP_BY_RE   = re.compile(r'\bgroup\s+by\s+(.+)',   re.IGNORECASE)
_PER_RE        = re.compile(r'\bper\s+(.+)',           re.IGNORECASE)
_BY_RE         = re.compile(r'\bby\s+(.+)',            re.IGNORECASE)


# =============================================================================
# Internal helpers
# =============================================================================

def _trim_phrase(phrase: str) -> str:
    """
    Remove trailing stop words (limit, order, having) and their arguments.

    Trims everything from the first stop word onward.

    Examples:
        "aircraft model limit 10"  →  "aircraft model"
        "departure airport order"  →  "departure airport"
        "booking class"            →  "booking class"   (no stop word)
    """
    return _PHRASE_STOP_RE.sub('', phrase).strip()


def _has_analytic_context(query: str) -> bool:
    """
    Return True iff the query contains an aggregation verb or a ranking pattern.

    Used to guard the bare "by <phrase>" pattern against false positives in
    filter/ordering contexts (e.g., "filtered by airport", "sorted by price").

    Aggregation verbs: count, sum, total, avg, average, min, max,
                       minimum, maximum.
    Ranking patterns:  top N, bottom N  (N = integer).

    Tokenisation splits on non-word characters so that partial matches inside
    longer words are avoided (e.g., "recount" does not match "count").
    """
    lower = query.lower()
    tokens = re.split(r'\W+', lower)
    if any(t in _AGGREGATION_VERBS for t in tokens):
        return True
    if _RANKING_RE.search(query):
        return True
    return False


# =============================================================================
# Public API
# =============================================================================

def extract_grouping_phrase(original_query: str) -> Optional[str]:
    """
    Detect structural grouping patterns in a NL query and return the phrase.

    Patterns are tried in priority order.  The first matching pattern wins.

    Priority:
        1. ``grouped by <phrase>``  — most explicit, always matched
        2. ``group by <phrase>``    — SQL-style, always matched
        3. ``per <phrase>``         — natural language grouping, always matched
        4. ``by <phrase>``          — guarded: only when analytic context present

    Stop words (limit, order, having) are trimmed from the tail of the phrase
    before returning.

    Args:
        original_query: Raw NL query string (unmodified, pre-extraction).

    Returns:
        The extracted grouping phrase (whitespace-stripped), or None if no
        grouping pattern is detected.

    Examples::

        extract_grouping_phrase("count flights per aircraft model")
        → "aircraft model"

        extract_grouping_phrase("count passengers grouped by age")
        → "age"

        extract_grouping_phrase("count flights per aircraft model limit 10")
        → "aircraft model"

        extract_grouping_phrase("top 5 flights by aircraft model")
        → "aircraft model"

        extract_grouping_phrase("count flights")
        → None

        extract_grouping_phrase("show flights ordered by price")
        → None   (no analytic context; "by" guard prevents false positive)
    """
    if not original_query or not original_query.strip():
        return None

    q = original_query.strip()

    # ------------------------------------------------------------------
    # Unconditional patterns: grouped by / group by / per
    # ------------------------------------------------------------------
    for pattern in (_GROUPED_BY_RE, _GROUP_BY_RE, _PER_RE):
        m = pattern.search(q)
        if m:
            phrase = _trim_phrase(m.group(1))
            if phrase:
                return phrase

    # ------------------------------------------------------------------
    # Conditional "by <phrase>" — only when analytic context is present
    # ------------------------------------------------------------------
    if _has_analytic_context(q):
        m = _BY_RE.search(q)
        if m:
            phrase = _trim_phrase(m.group(1))
            if phrase:
                return phrase

    return None
