"""
SQL Normalizer — Canonical Alias Injection + Column Qualification (v6.22)
=========================================================================

ARCHITECTURAL ROLE:
    Pre-RCL normalization phase that ensures every table reference in a SQL
    query is internally consistent *before* the Relational Corrector (RCL)
    operates.

PROBLEM SOLVED:
    When an NL-SQL engine generates SQL that already contains JOINs but uses
    bare table names as qualifiers (e.g. `booking_leg.flight_id`), and RCL
    subsequently adds aliases to those tables, the resulting SQL has two
    conflicting reference styles:

        - Old ON condition:  booking_leg.flight_id = flight.flight_id
        - New FROM/JOIN:     FROM booking_leg b JOIN flight f1 ON b.flight_id = f1.flight_id

    The alias validator correctly rejects this because BOOKING_LEG and FLIGHT
    are "referenced" but only B and F1 are "declared".

SOLUTION (Option A):
    Insert a Canonical Alias Injection stage BEFORE RCL in the pipeline:

        Sanitize → Normalize schema → [CANONICAL ALIAS INJECTION] → RCL → Alias Validation

    This stage performs four operations:
        1. Assign deterministic canonical aliases to every table in FROM/JOIN
           (tables that already have aliases keep their existing ones)
        2. Rewrite all bare `table_name.column` references (in ON, WHERE, etc.)
           to `canonical_alias.column`
        3. Qualify unambiguous unqualified column references in SELECT, WHERE,
           GROUP BY, HAVING, ORDER BY, and ON clauses with the correct alias
        4. Use JOIN structure context for ON clauses so that ambiguous columns
           (e.g. FK columns shared by two tables) are qualified to the correct
           side of each equality condition

    After normalization:
        - All column references are either fully-qualified or genuinely ambiguous
        - RCL sees no "missing" columns for tables that are already joined
        - When RCL does add a new JOIN, it operates on already-consistent SQL
        - The alias validator always sees a coherent alias set

GUARANTEES:
    - Deterministic: identical input → identical output
    - Idempotent: double-normalizing produces the same result
    - Non-blocking: any failure returns the original SQL unchanged
    - Schema-agnostic alias generation (driven by table names, not DB type)
    - No LLM calls, no external I/O, no global state
    - String literals are never modified (masked before qualification)

INVARIANTS PRESERVED:
    - Guardrail order unchanged (this sits *before* RCL, not after)
    - Alias validator contract unchanged (still enforces alias consistency)
    - RCL ambiguity loop unchanged (RCL still handles truly missing joins)
    - Cost guard unchanged
    - Complexity analyzer unchanged
    - SchemaGraph unchanged

ROOT CAUSE FIX (v6.22):
    tools.py stores schema keys as schema-qualified names ("public.booking_leg"),
    but _parse_table_declarations() strips schema prefixes, yielding bare names
    ("booking_leg").  create_canonical_alias_injector() previously passed the
    schema-qualified keys verbatim, so every self._schema.get(table, set()) call
    returned an empty set — making the entire qualification phase dead code.
    Fixed: strip schema prefix in the factory before building table_columns.
"""

import re
import logging
from typing import Dict, Set, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL keyword set — column names matching these are never qualified
# (mirrors SQLParser.SQL_KEYWORDS in relational_corrector.py)
# ---------------------------------------------------------------------------
_SQL_KEYWORDS: Set[str] = {
    'select', 'from', 'where', 'join', 'on', 'and', 'or', 'not',
    'as', 'left', 'right', 'inner', 'outer', 'full', 'cross', 'natural',
    'group', 'order', 'by', 'having', 'limit', 'offset', 'union',
    'distinct', 'all', 'asc', 'desc', 'nulls', 'first', 'last',
    'null', 'is', 'in', 'between', 'like', 'ilike', 'exists',
    'case', 'when', 'then', 'else', 'end', 'true', 'false',
    'count', 'sum', 'avg', 'min', 'max', 'array_agg', 'string_agg',
    'extract', 'epoch', 'cast', 'coalesce', 'nullif', 'greatest', 'least',
    'concat', 'substring', 'trim', 'upper', 'lower', 'length',
    'abs', 'ceil', 'floor', 'round', 'trunc',
    'now', 'current_date', 'current_time', 'current_timestamp',
    'date', 'time', 'timestamp', 'interval',
    'year', 'month', 'day', 'hour', 'minute', 'second',
}

# Structural keywords that cannot be aliases and terminate a table declaration
_STRUCTURAL_KW: str = (
    r'ON|WHERE|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL|JOIN|'
    r'GROUP|ORDER|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|FETCH|FOR'
)

# SQL single-quoted string literal.
# Handles the '' escape (SQL standard): 'it''s fine' matches as one literal.
# Pattern: opening ', zero-or-more non-quote chars, then zero-or-more groups
# of '' (escaped quote) followed by non-quote chars, then closing '.
_STRING_LITERAL_RE = re.compile(r"'[^']*(?:''[^']*)*'")

# Keyword lookahead used to avoid consuming SQL keywords as aliases
_KW_LOOKAHEAD: str = (
    r'(?:ON|WHERE|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL|JOIN|'
    r'GROUP|ORDER|BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|'
    r'FETCH|FOR|SELECT|SET|WITH|AS)\b'
)


# =============================================================================
# CANONICAL ALIAS INJECTOR
# =============================================================================

class CanonicalAliasInjector:
    """
    Pre-RCL SQL normalization: canonical alias injection and column qualification.

    Initialize once with schema column metadata (table → column set).
    Call normalize(sql) for each generated SQL before passing to RCL.

    Thread-safe: normalize() is stateless with respect to instance state.
    """

    # Aliases that cannot be assigned (they are SQL structural keywords)
    _ALIAS_KEYWORDS: Set[str] = {
        'ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT', 'INNER', 'OUTER',
        'CROSS', 'FULL', 'JOIN', 'GROUP', 'ORDER', 'HAVING', 'LIMIT',
        'OFFSET', 'AS', 'SELECT', 'FROM', 'SET', 'NATURAL', 'UNION',
        'ALL', 'DISTINCT', 'WITH', 'EXCEPT', 'INTERSECT', 'FETCH', 'FOR',
    }

    def __init__(self, table_columns: Dict[str, Set[str]]):
        """
        Args:
            table_columns: Mapping of table_name (any case) → set of column names.
                           Normalised to lowercase internally.
        """
        # Normalise everything to lowercase for consistent lookup
        self._schema: Dict[str, Set[str]] = {
            k.lower(): {c.lower() for c in v}
            for k, v in table_columns.items()
        }
        logger.info(
            f"[NORMALIZER] CanonicalAliasInjector ready "
            f"({len(self._schema)} tables in schema)"
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def normalize(self, sql: str) -> str:
        """
        Normalize SQL to ensure consistent alias usage before RCL.

        Pipeline:
            1. Skip unsupported forms (no FROM, CTE, UNION, subqueries)
            2. Parse table declarations → (table → existing_alias | None)
               [v6.21: negative lookahead prevents keyword consumption]
            3. Generate canonical aliases for unaliased tables
            4. Inject new aliases into FROM/JOIN declarations
            5. Rewrite `table_name.col` → `alias.col` for ALL tables
               [v6.21: full_alias_map, not just new_aliases]
            6. Qualify unambiguous unqualified columns in SELECT/WHERE/
               GROUP BY/HAVING/ORDER BY clauses (one column at a time).
               ON clauses are handled in a separate single contextual pass
               (_qualify_on_clauses_contextual) so that equality conditions
               with the same column name on both sides are split correctly:
                   ON col = col → ON left_alias.col = join_alias.col
               [v6.22: contextual ON qualification, literal masking, root-cause fix]

        Returns original SQL unchanged on any error (non-blocking).
        """
        if not sql or not sql.strip():
            return sql

        try:
            working = re.sub(r'\s+', ' ', sql).strip()

            # -- Guard: skip if no FROM clause ---------------------------------
            if not re.search(r'\bFROM\b', working, re.IGNORECASE):
                return sql

            # -- Guard: skip CTEs (too structurally complex) -------------------
            if re.search(r'^\s*WITH\b', working, re.IGNORECASE):
                logger.debug("[NORMALIZER] CTE detected — skipping normalization")
                return sql

            # -- Guard: skip UNION / INTERSECT / EXCEPT  -----------------------
            if re.search(r'\b(?:UNION|INTERSECT|EXCEPT)\b', working, re.IGNORECASE):
                logger.debug("[NORMALIZER] Set operation detected — skipping normalization")
                return sql

            # -- Guard: skip subquery-heavy SQL --------------------------------
            if re.search(r'\(\s*SELECT\b', working, re.IGNORECASE):
                logger.debug("[NORMALIZER] Subquery detected — skipping normalization")
                return sql

            # Step 1: Parse table declarations from FROM/JOIN ------------------
            table_decls = self._parse_table_declarations(working)
            if not table_decls:
                return sql

            # Step 2: Separate tables that have aliases from those that don't --
            existing_aliases: Dict[str, str] = {
                t: a for t, a in table_decls.items() if a is not None
            }
            need_alias: List[str] = [
                t for t, a in table_decls.items() if a is None
            ]

            # Step 3: Generate canonical aliases for unaliased tables ----------
            reserved = {a.lower() for a in existing_aliases.values()}
            new_aliases: Dict[str, str] = self._generate_alias_map(need_alias, reserved)

            # Full alias map: table_name → alias (existing + new)
            full_alias_map: Dict[str, str] = {**existing_aliases, **new_aliases}

            if not full_alias_map:
                return sql

            # Step 4: Inject new aliases into FROM/JOIN declarations -----------
            result = self._inject_aliases_into_from_join(working, new_aliases)

            # Step 5: Rewrite bare `table_name.col` → `alias.col`
            # Applies to ALL tables in full_alias_map (both pre-existing and new).
            # This ensures ON conditions with old-style table.column references
            # (e.g., from an LLM that generated table.col for already-aliased tables)
            # are consistently rewritten to alias.col.
            result = self._rewrite_table_references(result, full_alias_map)

            # Step 6: Qualify unambiguous unqualified columns ------------------
            primary_table = next(iter(table_decls), None)
            result = self._qualify_columns(result, full_alias_map, primary_table)

            if result != working:
                logger.info(
                    f"[NORMALIZER] SQL normalised. "
                    f"New aliases assigned: {new_aliases} | "
                    f"Pre-existing aliases: {existing_aliases}"
                )
            else:
                logger.debug("[NORMALIZER] SQL already consistent — no changes made")

            return result

        except Exception as exc:
            # Non-blocking: normalization failure must never abort execution.
            logger.warning(
                f"[NORMALIZER] Normalization failed (passing through original SQL): {exc}"
            )
            return sql

    # -------------------------------------------------------------------------
    # STEP 1: PARSE TABLE DECLARATIONS
    # -------------------------------------------------------------------------

    def _parse_table_declarations(self, sql: str) -> Dict[str, Optional[str]]:
        """
        Extract (table_name, alias | None) for every FROM/JOIN in the SQL.

        Returns an ordered dict where the first entry is the primary FROM table.
        Table names are lowercased; schema prefixes are stripped.

        v6.21 fix: the optional alias group now uses a negative lookahead to
        prevent SQL structural keywords (ON, JOIN, WHERE, …) from being consumed
        as the alias.  Without this guard, `FROM booking_leg JOIN ...` would
        capture `JOIN` as booking_leg's alias, consuming the keyword and making
        the subsequent `JOIN flight` clause invisible to finditer.
        """
        result: Dict[str, Optional[str]] = {}

        # Negative lookahead: the next identifier must NOT be a structural keyword.
        # This prevents the optional alias group from consuming keywords like JOIN,
        # ON, WHERE, GROUP, ORDER, etc.
        _kw = (
            r'ON|WHERE|LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL|JOIN|'
            r'GROUP|ORDER|BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|'
            r'FETCH|FOR|SELECT|SET|WITH|AS'
        )
        _kw_la = r'(?:' + _kw + r')\b'

        # Pattern: FROM/JOIN [schema.]table [[AS] alias]
        # The alias sub-group only fires when the next word is NOT a keyword.
        pattern = re.compile(
            r'(?:FROM|JOIN)\s+'
            r'(?:[A-Za-z_][A-Za-z0-9_]*\.)?'           # optional schema.
            r'([A-Za-z_][A-Za-z0-9_]*)'                 # table name (group 1)
            r'(?:\s+(?:AS\s+)?(?!' + _kw_la + r')'     # optional: space/AS, not a keyword
            r'([A-Za-z_][A-Za-z0-9_]*))?',              # alias (group 2)
            re.IGNORECASE
        )

        for match in pattern.finditer(sql):
            table = match.group(1).lower()
            raw_alias = match.group(2)
            alias: Optional[str] = raw_alias.lower() if raw_alias else None

            # First declaration wins (self-joins reuse the first alias)
            if table not in result:
                result[table] = alias

        return result

    # -------------------------------------------------------------------------
    # STEP 2-3: CANONICAL ALIAS GENERATION
    # -------------------------------------------------------------------------

    def _generate_alias_map(
        self,
        tables: List[str],
        reserved: Set[str],
    ) -> Dict[str, str]:
        """
        Produce a collision-free canonical alias for each table in *tables*.

        Algorithm:
            1. First letter of each underscore-separated word:
               booking_leg → bl, flight → f, airport → a
            2. On collision: first N characters of the base table name
               (N = 2, 3, … len)
            3. Last resort: base_alias + counter (bl1, bl2, …)

        Args:
            tables:   Tables that need new aliases.
            reserved: Aliases already claimed (won't be reused).
        """
        alias_map: Dict[str, str] = {}
        used: Set[str] = set(reserved)

        for table in sorted(tables):   # sorted → deterministic order
            base = self._table_to_base_alias(table)
            raw = table.split('.')[-1].lower()  # strip schema prefix

            if base not in used:
                alias_map[table] = base
                used.add(base)
            else:
                # Try progressively longer prefixes of the table name
                candidate = None
                for n in range(2, len(raw) + 1):
                    attempt = raw[:n]
                    if attempt not in used:
                        candidate = attempt
                        break

                if candidate:
                    alias_map[table] = candidate
                    used.add(candidate)
                else:
                    # Absolute fallback: base + counter
                    counter = 1
                    while f"{base}{counter}" in used:
                        counter += 1
                    alias_map[table] = f"{base}{counter}"
                    used.add(alias_map[table])

        return alias_map

    def _table_to_base_alias(self, table_name: str) -> str:
        """
        Generate a base alias from a table name.

        Takes the first letter of each underscore-separated word:
            booking_leg         → bl
            flight              → f
            passenger_itinerary → pi
            airport             → a
        """
        clean = table_name.split('.')[-1].lower()
        words = clean.split('_')
        return ''.join(w[0] for w in words if w)

    # -------------------------------------------------------------------------
    # STEP 4: INJECT ALIASES INTO FROM/JOIN
    # -------------------------------------------------------------------------

    def _inject_aliases_into_from_join(
        self,
        sql: str,
        new_aliases: Dict[str, str],
    ) -> str:
        """
        Add canonical aliases to FROM/JOIN declarations for unaliased tables.

        Transforms:
            FROM booking_leg JOIN flight ON ...
        into:
            FROM booking_leg bl JOIN flight fl ON ...

        Only touches tables in *new_aliases*; tables with pre-existing
        aliases are left completely unchanged.

        Detection of "no existing alias": the token immediately following the
        table name must be a structural SQL keyword (ON, WHERE, JOIN, …) or
        end-of-string, NOT an identifier (which would be an alias).
        """
        result = sql

        for table, alias in new_aliases.items():
            result = self._inject_alias_for_table(result, table, alias)

        return result

    def _inject_alias_for_table(self, sql: str, table: str, alias: str) -> str:
        """
        Inject *alias* after the *table* name in FROM/JOIN if no alias exists.

        The table is unaliased when it is immediately followed (after optional
        whitespace) by a structural SQL keyword or end of string — not by
        another identifier.
        """
        # Lookahead: structural keyword or end
        lookahead = (
            r'(?=\s+(?:' + _STRUCTURAL_KW + r')\b'  # structural keyword follows
            r'|\s*(?:;|$))'                           # or end of string/semicolon
        )

        pattern = re.compile(
            r'(?P<kw>FROM|JOIN)\s+'
            r'(?P<schema>(?:[A-Za-z_][A-Za-z0-9_]*\.)?)'  # optional schema.
            r'(?P<table>' + re.escape(table) + r')'
            + lookahead,
            re.IGNORECASE,
        )

        def _replace(m: re.Match) -> str:
            return (
                f"{m.group('kw')} "
                f"{m.group('schema')}"
                f"{m.group('table')} "
                f"{alias}"
            )

        return pattern.sub(_replace, sql)

    # -------------------------------------------------------------------------
    # STEP 5: REWRITE TABLE_NAME.COLUMN → ALIAS.COLUMN
    # -------------------------------------------------------------------------

    def _rewrite_table_references(
        self,
        sql: str,
        alias_map: Dict[str, str],
    ) -> str:
        """
        Replace `table_name.column` with `alias.column` for all tables in alias_map.

        v6.21: applies to ALL tables (both newly-aliased and pre-existing aliases).
        This ensures ON conditions with old-style `table.col` references — whether
        the table had a pre-existing alias or received one in this call — are all
        consistently rewritten to `alias.col`.

        Example:
            ON booking_leg.flight_id = flight.flight_id
            → ON bl.flight_id = f.flight_id

        Negative lookbehind `(?<!\.)` prevents rewriting `schema.table.col`
        (where `table` is preceded by a dot — part of a schema-qualified name).
        """
        result = sql

        for table, alias in alias_map.items():
            # Match the table name when:
            #   - NOT immediately preceded by a dot (avoids schema.table.col)
            #   - word-bounded on both sides
            #   - immediately followed by a dot (it IS being used as a qualifier)
            pattern = re.compile(
                r'(?<!\.)' +
                r'\b' + re.escape(table) + r'\b' +
                r'(?=\s*\.)',
                re.IGNORECASE,
            )
            result = pattern.sub(alias, result)

        return result

    # -------------------------------------------------------------------------
    # STEP 6: QUALIFY UNAMBIGUOUS UNQUALIFIED COLUMNS
    # -------------------------------------------------------------------------

    def _qualify_columns(
        self,
        sql: str,
        alias_map: Dict[str, str],
        primary_table: Optional[str],
    ) -> str:
        """
        Qualify unambiguous unqualified columns in SELECT/WHERE/GROUP BY/
        HAVING/ORDER BY/ON clauses with the correct table alias.

        v6.22 changes:
            - String literals are masked before qualification so their contents
              are never modified (e.g. WHERE status = 'flight_id' is safe).
            - ON clauses are handled in a single contextual pass
              (_qualify_on_clauses_contextual) rather than per-column, so that
              equality conditions like `col = col` (same bare name on both sides)
              are split correctly: left side → col_to_alias, right → join_alias.
            - Structural warning is emitted when col_to_alias is empty for a
              multi-table query (indicates schema key mismatch or missing schema).

        Rules:
            - If only one query table has the column → qualify with that alias.
            - If multiple tables have the column and one is the primary FROM
              table → qualify with the primary table alias (JOIN-equal, so the
              value is identical regardless of which side is chosen).
            - If ambiguous and no primary table preference → leave unqualified
              (will raise DB error; user or LLM must be more specific).

        Safety guards:
            - SQL keywords never qualify.
            - Column names that collide with table or alias names are skipped.
            - String literals are masked and restored unchanged.
        """
        query_tables: Set[str] = set(alias_map.keys())
        all_aliases: Set[str] = set(alias_map.values())

        # Build col → alias map considering ambiguity
        col_to_alias = self._build_col_to_alias(
            alias_map, primary_table, query_tables, all_aliases
        )

        if not col_to_alias:
            if len(query_tables) > 1:
                # This most likely means the schema key mismatch is still present
                # (factory returning schema-qualified keys vs. bare table names).
                logger.warning(
                    f"[NORMALIZER] Column qualification skipped — schema lookup "
                    f"returned no columns for tables: {sorted(query_tables)}. "
                    f"Verify that CanonicalAliasInjector schema uses bare table "
                    f"names (not schema-qualified names like public.booking_leg)."
                )
            return sql

        # Mask string literals so their content is never modified.
        masked, literals = self._mask_literals(sql)
        result = masked

        # Apply qualification to non-ON clauses, one column at a time.
        for col, col_alias in col_to_alias.items():
            pattern = re.compile(
                r'(?<!\.)\b' + re.escape(col) + r'\b(?!\s*\.)',
                re.IGNORECASE,
            )
            replacement = f"{col_alias}.{col}"

            result = self._apply_in_select(result, pattern, replacement)
            result = self._apply_in_where(result, pattern, replacement)
            result = self._apply_in_group_by(result, pattern, replacement)
            result = self._apply_in_having(result, pattern, replacement)
            result = self._apply_in_order_by(result, pattern, replacement)
            # ON clauses are handled below in a single contextual pass.

        # ON clauses: single pass with JOIN structure context.
        # This handles the tautology case (ON col = col → ON bl.col = f.col).
        result = self._qualify_on_clauses_contextual(result, col_to_alias, alias_map)

        return self._unmask_literals(result, literals)

    def _build_col_to_alias(
        self,
        alias_map: Dict[str, str],
        primary_table: Optional[str],
        query_tables: Set[str],
        all_aliases: Set[str],
    ) -> Dict[str, str]:
        """
        Determine which unqualified columns can be unambiguously qualified.

        Returns a mapping { column_name → alias } for columns where the
        owning table is unambiguous (or resolved via primary-table preference).
        """
        col_to_tables: Dict[str, List[str]] = {}

        for table in query_tables:
            for col in self._schema.get(table, set()):
                if col in _SQL_KEYWORDS:
                    continue
                # Guard: if a column name matches a table name or alias,
                # skip it — substituting it globally risks rewriting
                # FROM/JOIN declarations (e.g. FROM flight → FROM fl.flight)
                if col in query_tables or col in all_aliases:
                    continue
                col_to_tables.setdefault(col, []).append(table)

        col_to_alias: Dict[str, str] = {}
        for col, tables_with_col in col_to_tables.items():
            if len(tables_with_col) == 1:
                # Unambiguous: only one table has this column
                col_to_alias[col] = alias_map[tables_with_col[0]]
            elif primary_table and primary_table in tables_with_col:
                # Ambiguous but primary table has it:
                # qualify with primary alias (JOIN equality guarantees same value)
                col_to_alias[col] = alias_map[primary_table]
            # else: genuinely ambiguous — leave unqualified

        return col_to_alias

    # -------------------------------------------------------------------------
    # STRING LITERAL MASKING
    # -------------------------------------------------------------------------

    def _mask_literals(self, sql: str) -> Tuple[str, List[str]]:
        """
        Replace all SQL single-quoted string literals with stable placeholders.

        Returns (masked_sql, list_of_original_literals).  Placeholders have the
        form __STRL0000__ so they are unambiguous identifiers that cannot clash
        with real SQL tokens.

        Handles the '' escape sequence inside strings (standard SQL escaping).
        """
        literals: List[str] = []

        def _store(m: re.Match) -> str:
            idx = len(literals)
            literals.append(m.group(0))
            return f"__STRL{idx:04d}__"

        masked = _STRING_LITERAL_RE.sub(_store, sql)
        return masked, literals

    def _unmask_literals(self, sql: str, literals: List[str]) -> str:
        """Restore string literals previously masked by _mask_literals."""
        for i, lit in enumerate(literals):
            sql = sql.replace(f"__STRL{i:04d}__", lit)
        return sql

    # -------------------------------------------------------------------------
    # CLAUSE-RESTRICTED REPLACEMENT HELPERS
    # -------------------------------------------------------------------------

    def _apply_in_select(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """Apply replacement inside the SELECT clause (between SELECT and FROM)."""
        return self._apply_in_clause(
            sql, pattern, replacement,
            start_re=r'\bSELECT\b',
            end_re=r'\bFROM\b',
        )

    def _apply_in_where(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """Apply replacement inside the WHERE clause."""
        return self._apply_in_clause(
            sql, pattern, replacement,
            start_re=r'\bWHERE\b',
            end_re=r'\b(?:GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|;|$)\b',
        )

    def _apply_in_group_by(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """Apply replacement inside the GROUP BY clause."""
        return self._apply_in_clause(
            sql, pattern, replacement,
            start_re=r'\bGROUP\s+BY\b',
            end_re=r'\b(?:HAVING|ORDER\s+BY|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|;|$)\b',
        )

    def _apply_in_having(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """Apply replacement inside the HAVING clause."""
        return self._apply_in_clause(
            sql, pattern, replacement,
            start_re=r'\bHAVING\b',
            end_re=r'\b(?:ORDER\s+BY|LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|;|$)\b',
        )

    def _apply_in_order_by(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """Apply replacement inside the ORDER BY clause."""
        return self._apply_in_clause(
            sql, pattern, replacement,
            start_re=r'\bORDER\s+BY\b',
            end_re=r'\b(?:LIMIT|OFFSET|UNION|EXCEPT|INTERSECT|;|$)\b',
        )

    def _apply_in_on_clauses(
        self, sql: str, pattern: re.Pattern, replacement: str
    ) -> str:
        """
        Apply replacement inside every ON clause condition.

        Each ON expression runs from the ON keyword to the next structural
        boundary (the next JOIN of any form, WHERE, GROUP BY, ORDER BY, HAVING,
        LIMIT, OFFSET, set operator, or end of string).

        The FROM/JOIN declaration tokens themselves are NOT touched — those are
        before the ON keyword and outside the expression we process.

        After step 5 (_rewrite_table_references) all `table.col` patterns are
        already `alias.col`, so this step only fires for the rare case where the
        LLM generated a bare column name directly inside an ON condition.
        """
        # Keyword sequence that ends an ON expression
        _on_end_re = re.compile(
            r'\b(?:'
            r'(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL)\s+JOIN'
            r'|JOIN'
            r'|WHERE'
            r'|GROUP\s+BY'
            r'|ORDER\s+BY'
            r'|HAVING'
            r'|LIMIT'
            r'|OFFSET'
            r'|UNION'
            r'|EXCEPT'
            r'|INTERSECT'
            r')\b',
            re.IGNORECASE,
        )
        _on_kw_re = re.compile(r'\bON\b', re.IGNORECASE)

        parts: List[str] = []
        scan_pos = 0

        for on_m in _on_kw_re.finditer(sql):
            # Emit everything up to and including the ON keyword unchanged.
            parts.append(sql[scan_pos:on_m.end()])

            expr_start = on_m.end()
            end_m = _on_end_re.search(sql, expr_start)
            expr_end = end_m.start() if end_m else len(sql)

            # Apply replacement only to the ON expression body.
            on_expr = sql[expr_start:expr_end]
            parts.append(pattern.sub(replacement, on_expr))

            scan_pos = expr_end

        # Emit any remaining SQL after the last ON expression.
        parts.append(sql[scan_pos:])
        return ''.join(parts)

    # -------------------------------------------------------------------------
    # ON CLAUSE: CONTEXTUAL SINGLE-PASS QUALIFICATION (v6.22)
    # -------------------------------------------------------------------------

    def _qualify_on_clauses_contextual(
        self,
        sql: str,
        col_to_alias: Dict[str, str],
        alias_map: Dict[str, str],
    ) -> str:
        """
        Single-pass ON clause qualification using JOIN structure context.

        For each ``JOIN table alias ON …`` segment:
          1. Extract the join target alias from the JOIN declaration.
          2. Pass the ON expression to _qualify_on_expression, supplying the
             join alias so that equality conditions can be split correctly.

        This replaces the per-column _apply_in_on_clauses calls and solves the
        tautology problem:

            ON flight_id = flight_id
            → ON bl.flight_id = f.flight_id   (not bl.flight_id = bl.flight_id)

        Falls back to standard col_to_alias qualification when no join alias
        can be identified (e.g. CROSS JOIN or JOIN without explicit alias).
        """
        all_aliases: Set[str] = set(alias_map.values())

        # Matches: [LEFT|RIGHT|…] JOIN [schema.]table [AS] alias ON
        # Group 1 captures the alias (may be None if table has no alias).
        _join_on_re = re.compile(
            r'(?:(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL)\s+)?'
            r'JOIN\s+'
            r'(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*'   # table
            r'(?:\s+(?:AS\s+)?(?!' + _KW_LOOKAHEAD + r')'
            r'([A-Za-z_][A-Za-z0-9_]*))?'                              # group 1: alias
            r'\s+ON\b',
            re.IGNORECASE,
        )

        # End-of-ON-expression boundary (same as _apply_in_on_clauses)
        _on_end_re = re.compile(
            r'\b(?:'
            r'(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL|NATURAL)\s+JOIN'
            r'|JOIN'
            r'|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|OFFSET'
            r'|UNION|EXCEPT|INTERSECT'
            r')\b',
            re.IGNORECASE,
        )

        parts: List[str] = []
        scan_pos = 0

        for join_m in _join_on_re.finditer(sql):
            raw_alias = join_m.group(1)
            # Only use as context if this alias is registered in alias_map
            join_alias: Optional[str] = (
                raw_alias.lower()
                if raw_alias and raw_alias.lower() in all_aliases
                else None
            )

            on_start = join_m.end()
            end_m = _on_end_re.search(sql, on_start)
            on_end = end_m.start() if end_m else len(sql)

            # Emit everything up to and including the JOIN … ON header
            parts.append(sql[scan_pos:on_start])

            on_expr = sql[on_start:on_end]
            parts.append(
                self._qualify_on_expression(on_expr, join_alias, col_to_alias)
            )

            scan_pos = on_end

        parts.append(sql[scan_pos:])
        return ''.join(parts)

    def _qualify_on_expression(
        self,
        on_expr: str,
        join_alias: Optional[str],
        col_to_alias: Dict[str, str],
    ) -> str:
        """
        Qualify bare column references within a single ON expression.

        Splits the expression by AND/OR connectors and delegates each atomic
        condition to _qualify_on_condition.
        """
        if not on_expr.strip():
            return on_expr

        # Split by AND/OR while preserving the connectors.
        _connector_re = re.compile(r'(\bAND\b|\bOR\b)', re.IGNORECASE)
        parts = _connector_re.split(on_expr)

        qualified: List[str] = []
        for part in parts:
            if re.match(r'^\s*(?:AND|OR)\s*$', part, re.IGNORECASE):
                qualified.append(part)
            else:
                qualified.append(
                    self._qualify_on_condition(part, join_alias, col_to_alias)
                )
        return ''.join(qualified)

    def _qualify_on_condition(
        self,
        cond: str,
        join_alias: Optional[str],
        col_to_alias: Dict[str, str],
    ) -> str:
        """
        Qualify a single ON condition (e.g. ``col = col`` or ``col1 = col2``).

        Special case — same bare column on both sides with a known join alias:

            flight_id = flight_id
            → bl.flight_id = f.flight_id

        Left operand  → col_to_alias[col]  (FROM-chain / source table)
        Right operand → join_alias          (JOIN target table)

        This prevents the tautology ``ON bl.flight_id = bl.flight_id`` that
        the per-column approach would have produced.

        For all other conditions: standard col_to_alias substitution.
        If a column cannot be resolved (not in col_to_alias), it is left bare
        and a structural warning is emitted so the caller can diagnose the issue.
        """
        # Detect: bare_col = bare_col  (identical unqualified names on both sides)
        _same_col_eq_re = re.compile(
            r'^(?P<pre>\s*)'
            r'(?<!\.)\b(?P<col>[A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\.)'
            r'(?P<op>\s*=\s*)'
            r'(?<!\.)\b(?P=col)\b(?!\s*\.)'
            r'(?P<suf>\s*)$',
            re.IGNORECASE | re.DOTALL,
        )
        m = _same_col_eq_re.match(cond)
        if m and join_alias:
            col = m.group('col').lower()
            if col in col_to_alias:
                left_alias = col_to_alias[col]
                right_alias = join_alias
                if left_alias != right_alias:
                    # Correct split: left → FROM-chain table, right → JOIN target
                    return (
                        f"{m.group('pre')}{left_alias}.{col}"
                        f"{m.group('op')}{right_alias}.{col}"
                        f"{m.group('suf')}"
                    )
                # If both sides resolve to the same alias the ambiguity can't
                # be resolved here — fall through to standard qualification.
            else:
                logger.warning(
                    f"[NORMALIZER] Structural ambiguity: column '{col}' appears "
                    f"unqualified on both sides of ON condition but is not in "
                    f"col_to_alias — cannot qualify deterministically."
                )
                return cond  # Return as-is; DB will surface the error.

        # General case: apply col_to_alias to each bare column occurrence.
        result = cond
        for col, alias in col_to_alias.items():
            pattern = re.compile(
                r'(?<!\.)\b' + re.escape(col) + r'\b(?!\s*\.)',
                re.IGNORECASE,
            )
            result = pattern.sub(f'{alias}.{col}', result)
        return result

    def _apply_in_clause(
        self,
        sql: str,
        pattern: re.Pattern,
        replacement: str,
        start_re: str,
        end_re: str,
    ) -> str:
        """
        Apply *pattern* → *replacement* only within the SQL clause delimited
        by *start_re* (exclusive) and *end_re* (exclusive).

        If the clause is not present in the SQL, returns sql unchanged.
        """
        start_match = re.search(start_re, sql, re.IGNORECASE)
        if not start_match:
            return sql

        content_start = start_match.end()

        end_match = re.search(end_re, sql[content_start:], re.IGNORECASE | re.DOTALL)
        content_end = (
            content_start + end_match.start()
            if end_match
            else len(sql)
        )

        clause_text = sql[content_start:content_end]
        modified = pattern.sub(replacement, clause_text)

        if modified == clause_text:
            return sql

        return sql[:content_start] + modified + sql[content_end:]


# =============================================================================
# FACTORY
# =============================================================================

def create_canonical_alias_injector(schema: Dict) -> Optional[CanonicalAliasInjector]:
    """
    Build a CanonicalAliasInjector from the standard schema dict.

    Accepts the same schema format used by create_relational_corrector():
        {
            "tables": {
                "table_name": {
                    "columns": [{"name": "col", "type": "..."}, ...],
                    ...
                }
            }
        }

    Returns None if schema is missing or contains no tables.
    """
    if not schema or "tables" not in schema:
        logger.warning("[NORMALIZER] No schema provided — canonical alias injection disabled")
        return None

    table_columns: Dict[str, Set[str]] = {}
    for table_name, table_info in schema["tables"].items():
        # Strip schema prefix so keys match the bare names extracted by
        # _parse_table_declarations().
        # e.g. "public.booking_leg" → "booking_leg"
        # Without this, every self._schema.get(table, set()) call returns an
        # empty set because the DB stores "public.booking_leg" but the SQL
        # parser yields "booking_leg".
        bare_name = table_name.split('.')[-1].lower()
        cols = table_info.get("columns", [])
        table_columns[bare_name] = {c["name"].lower() for c in cols}

    if not table_columns:
        logger.warning("[NORMALIZER] Empty schema — canonical alias injection disabled")
        return None

    return CanonicalAliasInjector(table_columns)
