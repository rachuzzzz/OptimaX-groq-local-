"""
OptimaX v6.22+ — Execution Risk Classifier
============================================

Structural SQL execution risk classification.

PURPOSE
-------
Determines whether a normalized SQL query requires an EXPLAIN cost check
before execution. Replaces the brittle LIMIT-based bypass in the cost guard
with a principled, join-aware, grouping-aware, subquery-aware classification.

THE PROBLEM WITH LIMIT-BASED BYPASS
-------------------------------------
The previous guard logic was:

    if LIMIT <= 100:
        skip cost check

This is structurally wrong. LIMIT bounds the output cardinality, not the
internal computation cost. For a GROUP BY query over a 4-hop join:

    SELECT model, COUNT(*) FROM aircraft
    JOIN flight ... JOIN booking_leg ... JOIN boarding_pass ... JOIN passenger ...
    GROUP BY model
    LIMIT 10;

The database must:
  1. Execute all four JOINs, potentially materializing millions of intermediate rows.
  2. Group the result set.
  3. THEN apply LIMIT to the grouped output.

LIMIT 10 means we return 10 rows. It says nothing about the cost of steps 1 and 2.

CLASSIFICATION TAXONOMY
-----------------------
PURE_SINGLE_TABLE_LOOKUP
    No JOIN. No GROUP BY. No aggregate function.
    LIMIT directly bounds the output. Safe to skip EXPLAIN.

PURE_SINGLE_TABLE_AGGREGATE
    Single table. Aggregate function(s) (COUNT/SUM/AVG/MIN/MAX). No GROUP BY.
    Returns at most 1 row regardless of table size. Always safe.

GROUPED_SINGLE_TABLE_AGGREGATE
    Single table. GROUP BY present.
    Full-table scan required to compute groups. Cost unknown. EXPLAIN required.

MULTI_TABLE_JOIN
    One or more JOIN clauses. No GROUP BY.
    Join intermediate size unknown. EXPLAIN required.

MULTI_TABLE_GROUPED_AGGREGATE
    One or more JOIN clauses. GROUP BY present.
    High fan-out risk: join materialization + grouping. EXPLAIN always required.
    LIMIT does NOT change this classification.

COMPLEX_QUERY
    CTE (WITH), set operation (UNION/INTERSECT/EXCEPT), or subquery.
    Structural complexity too high for static classification.
    EXPLAIN always required.

TOKENIZER DESIGN
----------------
Classification is based on a token-stream analysis, not substring matching.

The tokenizer:
  1. Strips block comments (/* ... */) and line comments (-- ...).
  2. Strips single-quoted and dollar-quoted string literals to prevent SQL
     keywords embedded in string values (e.g., WHERE note = 'JOIN required')
     from being mistaken for structural clauses.
  3. Produces a flat list of (WORD | LPAREN | RPAREN, UPPERCASE_VALUE) tokens.

The analyzer:
  - Tracks parenthesis depth to distinguish top-level clauses from subqueries.
  - Only counts JOINs, GROUP BY, and set operations at depth 0 (top-level query).
  - Detects SELECT inside parentheses (depth > 0) as subqueries.

INVARIANT
---------
This classifier can only raise the required protection level, never lower it.
When uncertain, it falls back to COMPLEX_QUERY (always cost-checked).

PIPELINE POSITION
-----------------
Must be applied AFTER:
  - SQL sanitization
  - Schema normalization (schema prefix injection)
  - Canonical alias injection
  - RCL correction

And BEFORE:
  - Cost guard (EXPLAIN)
  - Database execution
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Risk Classification Taxonomy
# =============================================================================

class QueryRiskClass(Enum):
    """
    Structural execution risk class for a normalized SQL query.

    Determines whether the cost guard (EXPLAIN) must run before execution.
    """
    PURE_SINGLE_TABLE_LOOKUP       = "PURE_SINGLE_TABLE_LOOKUP"
    PURE_SINGLE_TABLE_AGGREGATE    = "PURE_SINGLE_TABLE_AGGREGATE"
    GROUPED_SINGLE_TABLE_AGGREGATE = "GROUPED_SINGLE_TABLE_AGGREGATE"
    MULTI_TABLE_JOIN               = "MULTI_TABLE_JOIN"
    MULTI_TABLE_GROUPED_AGGREGATE  = "MULTI_TABLE_GROUPED_AGGREGATE"
    COMPLEX_QUERY                  = "COMPLEX_QUERY"

    @property
    def requires_cost_check(self) -> bool:
        """
        True iff this risk class requires an EXPLAIN cost check before execution.

        Only pure single-table queries are structurally guaranteed to be cheap:

        PURE_SINGLE_TABLE_LOOKUP    — LIMIT directly bounds output rows.
        PURE_SINGLE_TABLE_AGGREGATE — returns at most 1 row (no GROUP BY).

        All other classes require EXPLAIN because LIMIT does not bound the
        internal computation cost (JOIN materialization, group computation, etc.).
        """
        return self not in (
            QueryRiskClass.PURE_SINGLE_TABLE_LOOKUP,
            QueryRiskClass.PURE_SINGLE_TABLE_AGGREGATE,
        )


@dataclass
class RiskClassification:
    """Result of structural SQL execution risk classification."""
    risk_class: QueryRiskClass
    has_joins: bool
    join_count: int
    has_group_by: bool
    has_aggregates: bool
    is_complex: bool            # True if CTE, set operation, or subquery present
    requires_cost_check: bool
    reasoning: str


# =============================================================================
# SQL Tokenizer
# =============================================================================
# Produces a flat list of (type, UPPERCASE_VALUE) tokens from a SQL string.
#
# Token types:
#   WORD   — SQL keyword or identifier (uppercased for comparison)
#   LPAREN — opening parenthesis '('
#   RPAREN — closing parenthesis ')'
#
# All other characters (operators, semicolons, commas, *, =, digits in
# numeric literals, etc.) are silently discarded. They carry no structural
# information relevant to risk classification.
#
# Schema-qualified identifiers (e.g. postgres_air.aircraft, T1.model) are
# treated as single WORD tokens because '.' is included in the identifier
# character class. This is correct: these tokens will never match any SQL
# keyword, so they cannot produce false positives.
# =============================================================================

_AGGREGATE_FUNCTIONS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})
_SET_OPERATIONS      = frozenset({"UNION", "INTERSECT", "EXCEPT"})
_CTE_KEYWORD         = "WITH"

# Identifier regex: leading alpha/underscore, then alphanumeric + underscore + $ + dot
# The dot allows schema-qualified names (postgres_air.aircraft) and alias refs (T1.model)
# to be parsed as single tokens. These can never match SQL keywords.
_IDENTIFIER_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_$\.]*')
_PAREN_RE      = re.compile(r'[()]')


def _strip_literals_and_comments(sql: str) -> str:
    """
    Remove comments and string literals from SQL before tokenization.

    This prevents SQL keywords embedded in string values or comments from
    being treated as structural clauses.

    Handles:
      - Block comments:              /* ... */
      - Line comments:               -- to end of line
      - Single-quoted strings:       'it''s a value'   ('' escape)
      - Dollar-quoted strings (PG):  $tag$...$tag$, $$...$$
    """
    # Block comments: /* ... */  (non-greedy, dotall)
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    # Line comments: -- to newline
    sql = re.sub(r'--[^\n]*', ' ', sql)
    # Dollar-quoted strings: $tag$...$tag$ or $$...$$
    sql = re.sub(r'\$[A-Za-z0-9_]*\$.*?\$[A-Za-z0-9_]*\$', "''", sql, flags=re.DOTALL)
    # Standard single-quoted strings — handles '' internal escapes
    sql = re.sub(r"'(?:[^'\\]|\\.)*'", "''", sql)
    return sql


def _tokenize(sql: str) -> List[Tuple[str, str]]:
    """
    Produce a (type, UPPER_VALUE) token list from normalized SQL.

    String literals and comments must have been stripped first.
    Returns only WORD, LPAREN, and RPAREN tokens.
    """
    cleaned = _strip_literals_and_comments(sql)
    tokens: List[Tuple[str, str]] = []

    # Merge identifier and paren matches into a single pass
    for match in re.finditer(r'[A-Za-z_][A-Za-z0-9_$\.]*|[()]', cleaned):
        raw = match.group(0)
        upper = raw.upper()
        if upper == '(':
            tokens.append(('LPAREN', upper))
        elif upper == ')':
            tokens.append(('RPAREN', upper))
        else:
            tokens.append(('WORD', upper))

    return tokens


# =============================================================================
# Structural Feature Extractor
# =============================================================================

def _extract_features(tokens: List[Tuple[str, str]]) -> dict:
    """
    Walk the token stream and extract the structural features needed for
    risk classification.

    Paren depth is tracked to separate top-level query structure from subquery
    content. Only top-level (depth == 0) tokens contribute to JOIN and GROUP BY
    counts. Subqueries (SELECT at depth > 0) are detected at any depth.

    Aggregate function detection is restricted to depth 0: an aggregate inside
    a subquery is internal to that subquery and does not change the outer
    query's classification (the outer query will be COMPLEX_QUERY anyway).
    """
    depth = 0

    has_cte      = False   # CTE: query starts with WITH
    has_set_op   = False   # UNION / INTERSECT / EXCEPT at top level
    has_subquery = False   # SELECT inside any parenthesized context
    join_count   = 0       # top-level JOIN clauses
    has_group_by = False   # top-level GROUP BY
    has_aggregates = False # top-level aggregate function call

    prev_word: Optional[str] = None

    n = len(tokens)

    # CTE detection: first WORD token must be WITH
    first_word = next(
        (tval for ttype, tval in tokens if ttype == 'WORD'),
        None,
    )
    if first_word == _CTE_KEYWORD:
        has_cte = True

    for i, (ttype, tval) in enumerate(tokens):

        # ── Depth tracking ──────────────────────────────────────────────────
        if ttype == 'LPAREN':
            depth += 1
            continue
        if ttype == 'RPAREN':
            depth -= 1
            if depth < 0:
                depth = 0  # guard against malformed SQL
            continue

        # ── Subquery detection: SELECT inside parentheses ────────────────────
        if depth > 0 and tval == 'SELECT':
            has_subquery = True
            # Continue processing — no need to inspect further for this token

        # ── Top-level structural analysis ────────────────────────────────────
        if depth == 0:
            # Set operations
            if tval in _SET_OPERATIONS:
                has_set_op = True

            # JOIN (any variant: INNER, LEFT, RIGHT, FULL, CROSS, NATURAL)
            # The variant keywords precede JOIN but do not appear instead of it.
            if tval == 'JOIN':
                join_count += 1

            # GROUP BY: two-token sequence GROUP → BY
            # prev_word tracks the immediately preceding WORD token.
            # This correctly handles arbitrary whitespace between GROUP and BY
            # (whitespace is not tokenized, so GROUP and BY are always adjacent
            # in the token stream when forming a GROUP BY clause).
            if tval == 'BY' and prev_word == 'GROUP':
                has_group_by = True

            # Aggregate function: FUNC_NAME immediately followed by LPAREN
            if tval in _AGGREGATE_FUNCTIONS:
                # Lookahead up to 2 positions for the opening paren.
                # The function name must be directly followed by '(' with
                # no intervening WORD tokens (which would mean a different
                # context such as a keyword that happens to match a function name).
                for j in range(i + 1, min(i + 3, n)):
                    next_ttype, next_tval = tokens[j]
                    if next_ttype == 'LPAREN':
                        has_aggregates = True
                        break
                    if next_ttype == 'WORD':
                        break  # intervening keyword — not a function call

        # Update prev_word for two-token sequence detection
        if ttype == 'WORD':
            prev_word = tval

    return {
        "has_cte":       has_cte,
        "has_set_op":    has_set_op,
        "has_subquery":  has_subquery,
        "join_count":    join_count,
        "has_joins":     join_count > 0,
        "has_group_by":  has_group_by,
        "has_aggregates": has_aggregates,
    }


# =============================================================================
# Classifier
# =============================================================================

class ExecutionRiskClassifier:
    """
    Structural SQL execution risk classifier.

    Classifies a normalized SQL query into one of six risk categories,
    determining whether the cost guard (EXPLAIN) must run before execution.

    The classification is derived from a token-stream analysis of the final
    normalized SQL string. String literals and comments are stripped before
    analysis to prevent false positives. Parenthesis depth is tracked to
    distinguish top-level query structure from subquery content.

    INVARIANT: This classifier can only raise the required protection level,
    never lower it. Any error during tokenization or classification falls
    back to COMPLEX_QUERY, which always requires a cost check.

    This class has no mutable state and is safe to use as a module-level
    singleton shared across threads.
    """

    def classify(self, sql: str) -> RiskClassification:
        """
        Classify a normalized SQL query by structural execution risk.

        Args:
            sql: Fully normalized SQL (sanitized, schema-prefixed,
                 alias-injected, RCL-corrected, LIMIT-enforced).

        Returns:
            RiskClassification with risk_class, structural feature flags,
            requires_cost_check, and a human-readable reasoning string.

        Never raises: any exception falls back to COMPLEX_QUERY.
        """
        if not sql or not sql.strip():
            return self._make(
                QueryRiskClass.COMPLEX_QUERY,
                False, 0, False, False, True,
                "Empty or null SQL — safe fallback to COMPLEX_QUERY",
            )

        try:
            tokens = _tokenize(sql)
            feats  = _extract_features(tokens)
            return self._classify_from_features(feats)

        except Exception as exc:
            logger.warning(
                f"[GUARD] ExecutionRiskClassifier error — safe fallback to COMPLEX_QUERY: {exc}"
            )
            return self._make(
                QueryRiskClass.COMPLEX_QUERY,
                False, 0, False, False, True,
                f"Tokenization error — safe fallback: {exc}",
            )

    def _classify_from_features(self, feats: dict) -> RiskClassification:
        """Apply the classification decision tree to extracted structural features."""

        is_complex   = feats["has_cte"] or feats["has_set_op"] or feats["has_subquery"]
        has_joins    = feats["has_joins"]
        join_count   = feats["join_count"]
        has_group_by = feats["has_group_by"]
        has_aggregates = feats["has_aggregates"]

        # ── Classification decision tree ─────────────────────────────────────
        # Ordered from most to least structurally complex.
        # The first matching condition wins.

        if is_complex:
            reasons = []
            if feats["has_cte"]:      reasons.append("CTE (WITH)")
            if feats["has_set_op"]:   reasons.append("set operation (UNION/INTERSECT/EXCEPT)")
            if feats["has_subquery"]: reasons.append("subquery")
            reasoning  = f"Structural complexity: {', '.join(reasons)}"
            risk_class = QueryRiskClass.COMPLEX_QUERY

        elif has_joins and has_group_by:
            reasoning = (
                f"{join_count} JOIN(s) + GROUP BY — "
                f"database must materialize the full join result before grouping; "
                f"LIMIT does not bound intermediate computation cost"
            )
            risk_class = QueryRiskClass.MULTI_TABLE_GROUPED_AGGREGATE

        elif has_joins:
            reasoning = (
                f"{join_count} JOIN(s), no GROUP BY — "
                f"join intermediate size unknown; EXPLAIN required"
            )
            risk_class = QueryRiskClass.MULTI_TABLE_JOIN

        elif has_group_by:
            reasoning = (
                "Single table, GROUP BY — "
                "full-table scan required to compute groups; cost unknown"
            )
            risk_class = QueryRiskClass.GROUPED_SINGLE_TABLE_AGGREGATE

        elif has_aggregates:
            reasoning = (
                "Single table, scalar aggregate(s), no GROUP BY — "
                "returns exactly 1 row; always safe"
            )
            risk_class = QueryRiskClass.PURE_SINGLE_TABLE_AGGREGATE

        else:
            reasoning = (
                "Single table, no aggregation, no GROUP BY — "
                "simple row lookup; LIMIT directly bounds output"
            )
            risk_class = QueryRiskClass.PURE_SINGLE_TABLE_LOOKUP

        return self._make(
            risk_class, has_joins, join_count,
            has_group_by, has_aggregates, is_complex, reasoning,
        )

    def _make(
        self,
        risk_class: QueryRiskClass,
        has_joins: bool,
        join_count: int,
        has_group_by: bool,
        has_aggregates: bool,
        is_complex: bool,
        reasoning: str,
    ) -> RiskClassification:
        return RiskClassification(
            risk_class=risk_class,
            has_joins=has_joins,
            join_count=join_count,
            has_group_by=has_group_by,
            has_aggregates=has_aggregates,
            is_complex=is_complex,
            requires_cost_check=risk_class.requires_cost_check,
            reasoning=reasoning,
        )
