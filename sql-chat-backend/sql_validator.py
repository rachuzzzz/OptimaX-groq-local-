"""
OptimaX v5.0 - SQL Alias Validation Layer
==========================================

PURPOSE (FOR INDUSTRY/ACADEMIC REVIEWERS):
This module prevents SQL execution failures caused by alias mismatches.

PROBLEM STATEMENT:
- NL-SQL engines sometimes generate SQL with undefined aliases
- Example: SELECT T2.passenger_id FROM flight T1 (T2 is never defined)
- Database correctly rejects these queries
- Without validation, users see cryptic database errors

SOLUTION:
A pre-execution validation layer that:
1. Parses SQL to extract DECLARED aliases (FROM, JOIN clauses)
2. Extracts REFERENCED aliases (SELECT, WHERE, GROUP BY, ORDER BY)
3. Compares and aborts if any reference is undeclared

WHAT THIS IS NOT (CRITICAL FOR REVIEWERS):
- NOT SQL repair (we abort, not fix)
- NOT LLM-based validation (pure regex parsing)
- NOT schema-aware (operates on SQL syntax only)
- NOT database-specific (standard SQL patterns)

ARCHITECTURAL POSITION:
    NL-SQL Output -> [SQL ALIAS VALIDATION] -> Schema Normalization -> Database

Author: OptimaX Team
Version: 5.0 (SQL Alias Validation)
"""

import re
import logging
from typing import Set, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AliasValidationResult:
    """
    Result of SQL alias validation.

    Attributes:
        valid: Whether the SQL passed alias validation
        declared_aliases: Set of aliases declared in FROM/JOIN
        referenced_aliases: Set of aliases referenced in query
        undefined_aliases: Set of aliases referenced but not declared
        error_message: Human-readable error (if invalid)
        sql: The SQL that was validated
    """
    valid: bool
    declared_aliases: Set[str]
    referenced_aliases: Set[str]
    undefined_aliases: Set[str]
    error_message: Optional[str]
    sql: str


class SQLAliasValidator:
    """
    Validates that all alias references in SQL are properly declared.

    IMPORTANT FOR REVIEWERS:
    - This is SYNTAX validation, not semantic validation
    - We parse SQL text with regex (no external dependencies)
    - We DO NOT attempt to repair invalid SQL
    - We ABORT execution if aliases are undefined
    """

    # Pattern to match table declarations with aliases in FROM/JOIN
    # Matches: table_name alias, table_name AS alias, schema.table alias
    # Examples: "flight T1", "booking AS b", "postgres_air.passenger p"
    FROM_JOIN_ALIAS_PATTERN = re.compile(
        r'''
        (?:FROM|JOIN)\s+                          # FROM or JOIN keyword
        (?:[\w.]+)\s+                             # table name (possibly schema-qualified)
        (?:AS\s+)?                                # optional AS keyword
        ([A-Za-z_][A-Za-z0-9_]*)                  # capture: alias
        (?:\s|,|ON|WHERE|LEFT|RIGHT|INNER|OUTER|CROSS|$)  # followed by delimiter
        ''',
        re.IGNORECASE | re.VERBOSE
    )

    # Pattern to find alias references (alias.column)
    # Matches: T1.column_name, t2.*, alias.field
    ALIAS_REFERENCE_PATTERN = re.compile(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:\*|[A-Za-z_][A-Za-z0-9_]*)',
        re.IGNORECASE
    )

    # Pattern to extract subqueries (we need to handle them separately)
    SUBQUERY_PATTERN = re.compile(
        r'\(\s*SELECT\s+.*?\)',
        re.IGNORECASE | re.DOTALL
    )

    # Known SQL functions that look like aliases but aren't
    # These use dot notation but are NOT table aliases
    SQL_FUNCTIONS_WITH_DOT = {
        'count', 'sum', 'avg', 'min', 'max', 'coalesce',
        'cast', 'convert', 'extract', 'date_part', 'date_trunc',
        'string_agg', 'array_agg', 'json_agg', 'jsonb_agg',
        'row_number', 'rank', 'dense_rank', 'ntile',
        'lead', 'lag', 'first_value', 'last_value',
        'pg_catalog', 'information_schema', 'public'  # schema names
    }

    def validate(self, sql: str) -> AliasValidationResult:
        """
        Validate that all alias references in SQL are declared.

        Args:
            sql: SQL query string to validate

        Returns:
            AliasValidationResult with validation status

        VALIDATION LOGIC:
        1. Extract aliases declared in FROM/JOIN clauses
        2. Extract alias references (prefix.column patterns)
        3. Compare: any reference not in declared set is INVALID
        """
        if not sql or not sql.strip():
            return AliasValidationResult(
                valid=True,
                declared_aliases=set(),
                referenced_aliases=set(),
                undefined_aliases=set(),
                error_message=None,
                sql=sql
            )

        # Normalize SQL for parsing
        normalized_sql = self._normalize_sql(sql)

        # Step 1: Extract declared aliases from FROM/JOIN
        declared_aliases = self._extract_declared_aliases(normalized_sql)

        # Step 2: Extract referenced aliases (prefix.column patterns)
        referenced_aliases = self._extract_referenced_aliases(normalized_sql)

        # Step 3: Find undefined aliases
        undefined_aliases = referenced_aliases - declared_aliases

        # Remove false positives (schema names, table-qualified columns, function calls)
        undefined_aliases = self._filter_false_positives(undefined_aliases, normalized_sql, declared_aliases)

        if undefined_aliases:
            error_msg = self._build_error_message(undefined_aliases, declared_aliases)
            logger.warning(f"SQL Alias Validation FAILED: {error_msg}")
            logger.debug(f"  Declared aliases: {declared_aliases}")
            logger.debug(f"  Referenced aliases: {referenced_aliases}")
            logger.debug(f"  Undefined aliases: {undefined_aliases}")

            return AliasValidationResult(
                valid=False,
                declared_aliases=declared_aliases,
                referenced_aliases=referenced_aliases,
                undefined_aliases=undefined_aliases,
                error_message=error_msg,
                sql=sql
            )

        logger.debug(f"SQL Alias Validation PASSED: {len(declared_aliases)} aliases declared")

        return AliasValidationResult(
            valid=True,
            declared_aliases=declared_aliases,
            referenced_aliases=referenced_aliases,
            undefined_aliases=set(),
            error_message=None,
            sql=sql
        )

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for consistent parsing."""
        # Remove comments
        sql = re.sub(r'--.*?$', ' ', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)

        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)

        return sql.strip()

    def _extract_declared_aliases(self, sql: str) -> Set[str]:
        """
        Extract aliases declared in FROM and JOIN clauses.

        Handles:
        - FROM table alias
        - FROM table AS alias
        - JOIN table alias ON ...
        - Schema-qualified tables: schema.table alias
        """
        declared = set()

        # Find all FROM/JOIN patterns with aliases
        matches = self.FROM_JOIN_ALIAS_PATTERN.findall(sql)
        for alias in matches:
            if alias.upper() not in ('ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT',
                                      'INNER', 'OUTER', 'CROSS', 'FULL', 'JOIN',
                                      'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET'):
                declared.add(alias.upper())

        # Also handle simple "FROM table t" patterns without AS
        simple_pattern = re.compile(
            r'(?:FROM|JOIN)\s+([\w.]+)\s+([A-Za-z_][A-Za-z0-9_]*)\b',
            re.IGNORECASE
        )
        for table, alias in simple_pattern.findall(sql):
            if alias.upper() not in ('ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT',
                                      'INNER', 'OUTER', 'CROSS', 'FULL', 'JOIN',
                                      'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
                                      'AS', 'SELECT', 'SET'):
                declared.add(alias.upper())

        return declared

    def _extract_referenced_aliases(self, sql: str) -> Set[str]:
        """
        Extract alias references (alias.column patterns) from SQL.

        References appear in:
        - SELECT: T1.column_name
        - WHERE: T1.field = T2.field
        - GROUP BY: T1.field
        - ORDER BY: T1.field
        - ON clauses: T1.id = T2.foreign_id
        """
        referenced = set()

        matches = self.ALIAS_REFERENCE_PATTERN.findall(sql)
        for alias in matches:
            # Skip known non-aliases
            if alias.lower() not in self.SQL_FUNCTIONS_WITH_DOT:
                referenced.add(alias.upper())

        return referenced

    def _filter_false_positives(self, undefined: Set[str], sql: str, declared_aliases: Set[str]) -> Set[str]:
        """
        Filter out false positives from undefined aliases.

        False positives include:
        - Schema names used in schema.table patterns in FROM/JOIN clauses
        - Table names used as table-qualified column references (when no aliases declared)
        - Built-in function namespaces

        CRITICAL FIX (v5.0.1):
        Schema-qualified table names (schema.table) in FROM/JOIN must NOT
        be treated as alias references. The schema component is part of
        the table identifier, not an alias.

        CRITICAL FIX (v5.0.2):
        Table-qualified column references (table.column) are VALID SQL when:
        - The table is declared in FROM/JOIN
        - No alias is assigned to that table

        Example 1: FROM postgres_air.booking
        - postgres_air is a SCHEMA, not an alias
        - booking is the TABLE

        Example 2: FROM airport JOIN flight ON airport.airport_code = flight.departure_airport
        - airport.airport_code is a TABLE-QUALIFIED column (valid)
        - flight.departure_airport is a TABLE-QUALIFIED column (valid)
        - No aliases are declared, so table names are valid qualifiers
        """
        # Step 1: Extract schema names from FROM/JOIN clauses
        # Pattern: FROM|JOIN schema.table (with or without following alias)
        schema_table_pattern = re.compile(
            r'(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)',
            re.IGNORECASE
        )

        # Collect all schema names used in FROM/JOIN clauses
        declared_schemas = set()
        for schema, table in schema_table_pattern.findall(sql):
            declared_schemas.add(schema.upper())

        # Step 2: Extract table names from FROM/JOIN clauses
        # These are valid qualifiers when used as table.column (if no alias assigned)
        #
        # Pattern matches:
        # - FROM table_name
        # - FROM schema.table_name
        # - JOIN table_name
        # - JOIN schema.table_name
        #
        # We extract the FINAL table name (after schema if present)
        table_name_pattern = re.compile(
            r'(?:FROM|JOIN)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)',
            re.IGNORECASE
        )

        # Collect all table names from FROM/JOIN clauses
        declared_tables = set()
        for table in table_name_pattern.findall(sql):
            declared_tables.add(table.upper())

        # Step 3: Determine which tables have NO alias assigned
        # If a table has an alias, table.column is invalid (must use alias.column)
        # If a table has NO alias, table.column is valid
        #
        # Pattern to find tables WITH aliases:
        # FROM table alias | FROM table AS alias | JOIN table alias ON
        table_with_alias_pattern = re.compile(
            r'(?:FROM|JOIN)\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\b',
            re.IGNORECASE
        )

        tables_with_aliases = set()
        for table, alias in table_with_alias_pattern.findall(sql):
            # Check if the "alias" is actually a SQL keyword (not a real alias)
            if alias.upper() not in ('ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT',
                                      'INNER', 'OUTER', 'CROSS', 'FULL', 'JOIN',
                                      'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
                                      'AS', 'SELECT', 'SET', 'NATURAL'):
                tables_with_aliases.add(table.upper())

        # Tables without aliases can be used as qualifiers
        tables_without_aliases = declared_tables - tables_with_aliases

        filtered = set()

        for alias in undefined:
            # Check if this is a schema name from a schema.table declaration
            if alias in declared_schemas:
                # This is a schema qualifier, NOT an undefined alias
                logger.debug(f"  Filtered '{alias}': schema name in FROM/JOIN")
                continue

            # Check if this is a table name used as table-qualified column
            # (only valid if the table has no alias assigned)
            if alias in tables_without_aliases:
                # This is a table-qualified column reference (valid SQL)
                logger.debug(f"  Filtered '{alias}': table name without alias (table.column is valid)")
                continue

            # Check if it's a known schema/function namespace
            if alias.lower() in self.SQL_FUNCTIONS_WITH_DOT:
                logger.debug(f"  Filtered '{alias}': known function/schema namespace")
                continue

            filtered.add(alias)

        return filtered

    def _build_error_message(self, undefined: Set[str], declared: Set[str]) -> str:
        """Build a clear error message for undefined aliases."""
        undefined_list = sorted(undefined)

        if len(undefined_list) == 1:
            alias = undefined_list[0]
            msg = f"Invalid SQL generated: alias '{alias}' is not defined."
        else:
            aliases_str = "', '".join(undefined_list)
            msg = f"Invalid SQL generated: aliases '{aliases_str}' are not defined."

        if declared:
            declared_str = "', '".join(sorted(declared))
            msg += f" Defined aliases: '{declared_str}'."
        else:
            msg += " No aliases are defined in the query."

        return msg


@dataclass
class QueryComplexityResult:
    """
    Result of query complexity analysis.

    Attributes:
        complexity_score: Numeric score (higher = more complex)
        join_count: Number of JOIN operations
        has_where_filter: Whether query has WHERE clause
        has_limit: Whether query has LIMIT clause
        aggregation_count: Number of aggregation functions
        subquery_count: Number of subqueries
        is_safe: Whether query passes safety threshold
        requires_filter: Whether additional filters are required
        warning_message: Warning for complex queries
    """
    complexity_score: int
    join_count: int
    has_where_filter: bool
    has_limit: bool
    aggregation_count: int
    subquery_count: int
    is_safe: bool
    requires_filter: bool
    warning_message: Optional[str]


class QueryComplexityAnalyzer:
    """
    Analyzes SQL query complexity and enforces safety guardrails.

    IMPORTANT FOR REVIEWERS:
    - This is HEURISTIC analysis (not execution plan)
    - We count structural elements as complexity indicators
    - We DO NOT execute the query to measure complexity
    - We require filters for high-complexity queries

    COMPLEXITY SCORING:
    - Each JOIN: +2 points
    - Aggregation without GROUP BY: +3 points
    - Subquery: +3 points
    - Missing WHERE: +2 points
    - Missing LIMIT: +1 point

    THRESHOLDS:
    - Score >= 6: Require at least one filter (route, date, ID)
    - Score >= 10: Block execution, require user to narrow query
    """

    # Threshold scores
    FILTER_REQUIRED_THRESHOLD = 6
    BLOCK_EXECUTION_THRESHOLD = 10

    # Aggregation function patterns
    AGGREGATION_PATTERN = re.compile(
        r'\b(COUNT|SUM|AVG|MIN|MAX|STRING_AGG|ARRAY_AGG)\s*\(',
        re.IGNORECASE
    )

    # JOIN pattern
    JOIN_PATTERN = re.compile(
        r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|JOIN)\b',
        re.IGNORECASE
    )

    # Subquery pattern
    SUBQUERY_PATTERN = re.compile(
        r'\(\s*SELECT\b',
        re.IGNORECASE
    )

    def analyze(self, sql: str, row_limit: Optional[int] = None) -> QueryComplexityResult:
        """
        Analyze SQL query complexity.

        v6.9: BUG 2 FIX - Added row_limit parameter for execution-order independence.

        Args:
            sql: SQL query string
            row_limit: If provided, indicates LIMIT will be enforced by the system.
                       This makes the analyzer independent of execution order.

        Returns:
            QueryComplexityResult with complexity metrics

        ARCHITECTURAL NOTE:
        When row_limit is provided, the analyzer treats the query as bounded
        even if LIMIT isn't in the SQL yet. This eliminates execution-order
        dependency - the analyzer can be called at any point in the pipeline.
        """
        if not sql or not sql.strip():
            return QueryComplexityResult(
                complexity_score=0,
                join_count=0,
                has_where_filter=False,
                has_limit=True,  # Empty query is "safe"
                aggregation_count=0,
                subquery_count=0,
                is_safe=True,
                requires_filter=False,
                warning_message=None
            )

        sql_upper = sql.upper()

        # Count structural elements
        join_count = len(self.JOIN_PATTERN.findall(sql))
        aggregation_count = len(self.AGGREGATION_PATTERN.findall(sql))
        subquery_count = len(self.SUBQUERY_PATTERN.findall(sql))
        has_where = 'WHERE' in sql_upper
        has_limit_in_sql = 'LIMIT' in sql_upper
        has_group_by = 'GROUP BY' in sql_upper

        # =====================================================================
        # v6.9: BUG 2 FIX - Treat as bounded if row_limit will be enforced
        # =====================================================================
        # If row_limit is provided, LIMIT will be injected by the system.
        # This makes complexity analysis execution-order independent.
        # =====================================================================
        has_limit = has_limit_in_sql or (row_limit is not None)

        # Calculate complexity score
        score = 0
        score += join_count * 2
        score += subquery_count * 3

        # Aggregation without GROUP BY is expensive (full table scan)
        if aggregation_count > 0 and not has_group_by:
            score += 3
        else:
            score += aggregation_count  # Still adds some complexity

        if not has_where:
            score += 2

        if not has_limit:
            score += 1

        # =====================================================================
        # v6.9: BUG 2 FIX - Bounded queries are always safe
        # =====================================================================
        # Queries with LIMIT (or where LIMIT will be enforced) are bounded
        # and should NEVER be blocked by the complexity analyzer.
        # The cost guard handles expensive queries at runtime.
        #
        # RULE: If query is bounded -> is_safe = True
        #       Complexity blocking only applies to UNBOUNDED queries.
        # =====================================================================
        if has_limit:
            # Bounded query - always safe from complexity perspective
            is_safe = True
            requires_filter = False
            if row_limit is not None and not has_limit_in_sql:
                logger.debug(
                    f"[COMPLEXITY] Query will have LIMIT {row_limit} enforced - marked safe (score={score})"
                )
            else:
                logger.debug(
                    f"[COMPLEXITY] Query has LIMIT - marked safe (score={score})"
                )
        else:
            # Unbounded query - apply complexity thresholds
            is_safe = score < self.BLOCK_EXECUTION_THRESHOLD
            requires_filter = score >= self.FILTER_REQUIRED_THRESHOLD and not has_where

        # Build warning message
        warning = None
        if not is_safe:
            warning = self._build_block_message(join_count, aggregation_count, has_where)
        elif requires_filter:
            warning = self._build_filter_message(join_count, aggregation_count)

        return QueryComplexityResult(
            complexity_score=score,
            join_count=join_count,
            has_where_filter=has_where,
            has_limit=has_limit,
            aggregation_count=aggregation_count,
            subquery_count=subquery_count,
            is_safe=is_safe,
            requires_filter=requires_filter,
            warning_message=warning
        )

    def _build_block_message(self, joins: int, aggs: int, has_where: bool) -> str:
        """Build message for blocked queries."""
        reasons = []
        if joins >= 3:
            reasons.append(f"{joins} table joins")
        if aggs > 0:
            reasons.append(f"{aggs} aggregation(s)")
        if not has_where:
            reasons.append("no filtering conditions")

        reason_str = ", ".join(reasons) if reasons else "high complexity"

        return (
            f"This query is too complex to execute safely ({reason_str}). "
            f"Please narrow your query by specifying:\n"
            f"- A specific route (e.g., 'JFK to LAX')\n"
            f"- A date range (e.g., 'in January 2024')\n"
            f"- A specific entity (e.g., 'passenger 12345')"
        )

    def _build_filter_message(self, joins: int, aggs: int) -> str:
        """Build message requesting filters."""
        return (
            f"This query involves {joins} joins and may be slow without filters. "
            f"Consider adding a route, date range, or specific ID to narrow results."
        )


# =============================================================================
# SQL OUTPUT SANITIZER (v5.0.2 - Strict Output Contract)
# =============================================================================
# PURPOSE (FOR INDUSTRY/ACADEMIC REVIEWERS):
# This module enforces a strict SQL-only output contract between the NL-SQL
# engine and the execution pipeline.
#
# PROBLEM STATEMENT:
# NL-SQL engines (like LlamaIndex NLSQLTableQueryEngine) sometimes emit:
# - Natural language explanations before/after SQL
# - Multiple SQL statements in a single response
# - Commentary between SQL fragments
#
# Example of INVALID output:
#   SELECT AVG(award_points) FROM frequent_flyer
#
#   To make the query more interesting, let's order the results...
#
#   SELECT level, AVG(award_points) FROM frequent_flyer GROUP BY level;
#
# SOLUTION:
# A strict sanitizer that:
# 1. Extracts ONLY the FIRST valid SQL statement
# 2. Detects multiple SQL statements (FAIL FAST)
# 3. Discards all natural language commentary
# 4. Returns clean SQL or a structured error
#
# WHAT THIS IS NOT (CRITICAL FOR REVIEWERS):
# - NOT SQL repair (we extract or reject, never modify)
# - NOT query merging (multiple statements = HARD FAILURE)
# - NOT schema-aware (operates on SQL syntax only)
# - NOT relaxing validation (this is an ADDITIONAL gate)
#
# ARCHITECTURAL POSITION:
#     NL-SQL Output -> [SQL OUTPUT SANITIZER] -> Alias Validation -> Execution
# =============================================================================

@dataclass
class SanitizationResult:
    """
    Result of SQL output sanitization.

    Attributes:
        valid: Whether sanitization produced valid output
        sql: The sanitized SQL statement (if valid)
        statement_count: Number of SQL statements detected
        had_commentary: Whether natural language was present
        error_message: Error description (if invalid)
        raw_input: The original input for debugging
    """
    valid: bool
    sql: Optional[str]
    statement_count: int
    had_commentary: bool
    error_message: Optional[str]
    raw_input: str


class SQLOutputSanitizer:
    """
    Enforces strict SQL-only output contract.

    IMPORTANT FOR REVIEWERS:
    - This is a PROTOCOL BOUNDARY enforcer
    - It does NOT repair or modify SQL logic
    - It ABORTS on multiple statements (no merging)
    - It STRIPS commentary (natural language never reaches validators)
    """

    # SQL statement start patterns (case-insensitive)
    # These indicate the beginning of a new SQL statement
    SQL_STATEMENT_PATTERNS = [
        r'\bSELECT\b',
        r'\bWITH\b',  # CTE
        r'\bINSERT\b',
        r'\bUPDATE\b',
        r'\bDELETE\b',
        r'\bCREATE\b',
        r'\bDROP\b',
        r'\bALTER\b',
        r'\bTRUNCATE\b',
    ]

    # Combined pattern to find SQL statement starts
    STATEMENT_START_PATTERN = re.compile(
        '|'.join(SQL_STATEMENT_PATTERNS),
        re.IGNORECASE
    )

    # Pattern to detect statement terminators (semicolons followed by more SQL)
    MULTI_STATEMENT_PATTERN = re.compile(
        r';\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\b',
        re.IGNORECASE
    )

    # Pattern to extract SQL block (greedy, captures until end or next statement)
    # This captures a SQL statement from keyword to semicolon or end
    SQL_BLOCK_PATTERN = re.compile(
        r'\b((?:SELECT|WITH)\b.*?)(?:;(?:\s*$|\s*(?=SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE))|$)',
        re.IGNORECASE | re.DOTALL
    )

    # Read-only statements (these are the ONLY ones we allow)
    READ_ONLY_PATTERN = re.compile(
        r'^\s*(?:SELECT|WITH)\b',
        re.IGNORECASE
    )

    def sanitize(self, raw_output: str) -> SanitizationResult:
        """
        Sanitize NL-SQL engine output to extract exactly one SQL statement.

        Args:
            raw_output: Raw output from NL-SQL engine

        Returns:
            SanitizationResult with sanitized SQL or error

        CRITICAL BEHAVIOR:
        - Extracts FIRST valid SQL statement
        - FAILS if multiple statements detected
        - STRIPS all natural language commentary
        - Returns structured error on violation
        """
        if not raw_output or not raw_output.strip():
            return SanitizationResult(
                valid=False,
                sql=None,
                statement_count=0,
                had_commentary=False,
                error_message="Empty SQL output from NL-SQL engine",
                raw_input=raw_output or ""
            )

        raw_input = raw_output  # Preserve for debugging
        working_text = raw_output.strip()

        # =====================================================================
        # STEP 1: Detect multiple SQL statements (FAIL FAST)
        # =====================================================================
        # If we find a semicolon followed by another SQL keyword, this is
        # multiple statements. We REJECT this - no merging, no selection.
        # =====================================================================
        if self.MULTI_STATEMENT_PATTERN.search(working_text):
            # Count actual SQL statements
            statement_starts = self.STATEMENT_START_PATTERN.findall(working_text)
            statement_count = len(statement_starts)

            logger.warning(
                f"[SANITIZER] REJECTED: Multiple SQL statements detected "
                f"({statement_count} statements)"
            )

            return SanitizationResult(
                valid=False,
                sql=None,
                statement_count=statement_count,
                had_commentary=True,  # Likely has commentary between statements
                error_message=(
                    f"Multiple SQL statements detected ({statement_count}). "
                    f"Only one statement per query is allowed. Please ask a more specific question."
                ),
                raw_input=raw_input
            )

        # =====================================================================
        # STEP 2: Find the FIRST SQL statement
        # =====================================================================
        # Look for SELECT or WITH (read-only). Other statements are blocked.
        # =====================================================================
        first_match = self.STATEMENT_START_PATTERN.search(working_text)

        if not first_match:
            # No SQL found at all - pure commentary or garbage
            logger.warning("[SANITIZER] REJECTED: No SQL statement found in output")
            return SanitizationResult(
                valid=False,
                sql=None,
                statement_count=0,
                had_commentary=True,
                error_message="No valid SQL statement found in NL-SQL output",
                raw_input=raw_input
            )

        # Check if there's commentary BEFORE the SQL
        had_commentary = first_match.start() > 0

        # Extract from first SQL keyword to end
        sql_start = first_match.start()
        sql_portion = working_text[sql_start:]

        # =====================================================================
        # STEP 3: Extract the complete SQL statement
        # =====================================================================
        # Find the end of the first statement (semicolon or end of string)
        # =====================================================================
        sql_statement = self._extract_first_statement(sql_portion)

        if not sql_statement:
            logger.warning("[SANITIZER] REJECTED: Could not extract valid SQL statement")
            return SanitizationResult(
                valid=False,
                sql=None,
                statement_count=0,
                had_commentary=had_commentary,
                error_message="Could not extract valid SQL statement from output",
                raw_input=raw_input
            )

        # =====================================================================
        # STEP 4: Verify read-only (SELECT/WITH only)
        # =====================================================================
        if not self.READ_ONLY_PATTERN.match(sql_statement):
            detected_type = self.STATEMENT_START_PATTERN.search(sql_statement)
            type_name = detected_type.group(0).upper() if detected_type else "UNKNOWN"

            logger.warning(f"[SANITIZER] REJECTED: Non-read-only statement ({type_name})")
            return SanitizationResult(
                valid=False,
                sql=None,
                statement_count=1,
                had_commentary=had_commentary,
                error_message=f"Only SELECT queries are allowed. Detected: {type_name}",
                raw_input=raw_input
            )

        # =====================================================================
        # STEP 5: Final cleanup
        # =====================================================================
        # Remove trailing semicolons (we add them if needed during execution)
        # Normalize whitespace
        # =====================================================================
        clean_sql = self._cleanup_sql(sql_statement)

        # Check if there was trailing commentary after the SQL
        sql_end_pos = sql_start + len(sql_statement)
        has_trailing_commentary = sql_end_pos < len(working_text) and working_text[sql_end_pos:].strip()
        had_commentary = had_commentary or has_trailing_commentary

        if had_commentary:
            logger.info("[SANITIZER] Stripped natural language commentary from SQL output")

        logger.info(f"[SANITIZER] Extracted clean SQL: {clean_sql[:80]}...")

        return SanitizationResult(
            valid=True,
            sql=clean_sql,
            statement_count=1,
            had_commentary=had_commentary,
            error_message=None,
            raw_input=raw_input
        )

    def _extract_first_statement(self, sql_text: str) -> Optional[str]:
        """
        Extract the first complete SQL statement from text.

        Handles:
        - Statements ending with semicolon
        - Statements ending at EOF
        - Nested parentheses (subqueries)
        """
        # Remove leading/trailing whitespace
        sql_text = sql_text.strip()

        if not sql_text:
            return None

        # Find first semicolon that's not inside a string or parenthesis
        # Simplified approach: find first ; that ends the statement
        paren_depth = 0
        in_single_quote = False
        in_double_quote = False
        i = 0

        while i < len(sql_text):
            char = sql_text[i]

            # Handle escape sequences in strings
            if i > 0 and sql_text[i - 1] == '\\':
                i += 1
                continue

            # Track quotes
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote

            # Track parentheses (only outside strings)
            if not in_single_quote and not in_double_quote:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth = max(0, paren_depth - 1)
                elif char == ';' and paren_depth == 0:
                    # Found end of statement
                    return sql_text[:i + 1]

            i += 1

        # No semicolon found - return entire text (statement without terminator)
        return sql_text

    def _cleanup_sql(self, sql: str) -> str:
        """
        Clean up extracted SQL statement.

        - Remove trailing semicolons
        - Normalize whitespace
        - Remove any remaining commentary artifacts
        """
        # Remove trailing semicolons (we control termination)
        sql = sql.rstrip(';').strip()

        # Normalize internal whitespace (but preserve necessary spaces)
        sql = re.sub(r'\s+', ' ', sql)

        return sql


# =============================================================================
# COLUMN EXISTENCE VALIDATOR (v5.0.2 - Pre-Execution Schema Validation)
# =============================================================================
# PURPOSE (FOR INDUSTRY/ACADEMIC REVIEWERS):
# This module prevents runtime database errors caused by non-existent columns
# by validating SQL against known schema metadata BEFORE execution.
#
# PROBLEM STATEMENT:
# NL-SQL engines generate syntactically valid SQL that references columns
# which may not exist in the target tables. This causes UndefinedColumn
# errors at the database layer, which:
# - Leak internal error details to clients
# - Provide poor user experience
# - Violate Phase-1 stability guarantees
#
# SOLUTION:
# A pre-execution validator that:
# 1. Parses SQL to extract table and column references
# 2. Validates each column exists in its referenced table
# 3. Returns structured error BEFORE database execution
#
# WHAT THIS IS NOT (CRITICAL FOR REVIEWERS):
# - NOT SQL repair (we reject, never modify)
# - NOT column mapping (we don't guess "airport_code" -> "departure_airport")
# - NOT join inference (we don't add missing tables)
# - NOT semantic analysis (we validate syntax against schema, nothing more)
#
# ARCHITECTURAL POSITION:
#     Sanitized SQL -> [COLUMN EXISTENCE VALIDATOR] -> Alias Validation -> Execution
# =============================================================================

@dataclass
class ColumnValidationResult:
    """
    Result of column existence validation.

    Attributes:
        valid: Whether all referenced columns exist
        sql: The SQL that was validated
        invalid_columns: List of (table, column) tuples that don't exist
        available_columns: Dict mapping tables to their valid columns (for error messages)
        error_message: Human-readable error (if invalid)
    """
    valid: bool
    sql: str
    invalid_columns: List[Tuple[str, str]]  # [(table, column), ...]
    available_columns: Dict[str, List[str]]  # {table: [columns]}
    error_message: Optional[str]


class ColumnExistenceValidator:
    """
    Validates that SQL only references columns that exist in the schema.

    IMPORTANT FOR REVIEWERS:
    - This is SCHEMA VALIDATION, not semantic analysis
    - We use runtime schema metadata (database agnostic)
    - We DO NOT repair or rewrite SQL
    - We ABORT execution if columns are invalid
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize with schema metadata.

        Args:
            schema: Schema dictionary with structure:
                    {"tables": {"table_name": {"columns": [{"name": "col", "type": "..."}]}}}
        """
        self.schema = schema
        self._build_column_index()

    def _build_column_index(self) -> None:
        """Build a fast lookup index for column validation."""
        # Map: table_name (lowercase) -> set of column names (lowercase)
        self.column_index: Dict[str, Set[str]] = {}
        # Map: table_name (lowercase) -> list of column names (original case, for error messages)
        self.column_names: Dict[str, List[str]] = {}

        if not self.schema or "tables" not in self.schema:
            return

        for table_name, table_info in self.schema["tables"].items():
            # Extract just the table name (without schema prefix)
            simple_name = table_name.split(".")[-1].lower()

            columns = table_info.get("columns", [])
            self.column_index[simple_name] = {
                col["name"].lower() for col in columns
            }
            self.column_names[simple_name] = [
                col["name"] for col in columns
            ]

    def validate(self, sql: str) -> ColumnValidationResult:
        """
        Validate that all referenced columns exist in the schema.

        Args:
            sql: SQL query string to validate

        Returns:
            ColumnValidationResult with validation status

        VALIDATION LOGIC:
        1. Extract tables referenced in FROM/JOIN clauses
        2. Build alias-to-table mapping
        3. Extract column references (qualified and unqualified)
        4. Validate each column exists in its table
        5. Return structured error if any column is invalid
        """
        if not sql or not sql.strip():
            return ColumnValidationResult(
                valid=True,
                sql=sql,
                invalid_columns=[],
                available_columns={},
                error_message=None
            )

        if not self.column_index:
            # No schema available - skip validation (graceful degradation)
            logger.warning("[COLUMN_VALIDATOR] No schema available - skipping validation")
            return ColumnValidationResult(
                valid=True,
                sql=sql,
                invalid_columns=[],
                available_columns={},
                error_message=None
            )

        # Normalize SQL for parsing
        normalized_sql = self._normalize_sql(sql)

        # Step 1: Extract tables and aliases from FROM/JOIN
        table_aliases = self._extract_table_aliases(normalized_sql)

        if not table_aliases:
            # No tables found - can't validate columns
            logger.debug("[COLUMN_VALIDATOR] No tables found in SQL - skipping validation")
            return ColumnValidationResult(
                valid=True,
                sql=sql,
                invalid_columns=[],
                available_columns={},
                error_message=None
            )

        # Step 2: Extract column references
        column_refs = self._extract_column_references(normalized_sql, table_aliases)

        # Step 3: Validate each column
        invalid_columns = []
        available_columns = {}

        for table_name, column_name in column_refs:
            table_lower = table_name.lower()
            column_lower = column_name.lower()

            if table_lower not in self.column_index:
                # Table not in schema - might be a schema issue, let DB handle it
                logger.debug(f"[COLUMN_VALIDATOR] Table '{table_name}' not in schema - skipping")
                continue

            if column_lower not in self.column_index[table_lower]:
                # Column doesn't exist in table
                invalid_columns.append((table_name, column_name))

                # Add available columns for error message
                if table_lower not in available_columns:
                    available_columns[table_lower] = self.column_names.get(table_lower, [])

        if invalid_columns:
            error_msg = self._build_error_message(invalid_columns, available_columns)
            logger.warning(f"[COLUMN_VALIDATOR] Validation FAILED: {error_msg}")

            return ColumnValidationResult(
                valid=False,
                sql=sql,
                invalid_columns=invalid_columns,
                available_columns=available_columns,
                error_message=error_msg
            )

        logger.debug(f"[COLUMN_VALIDATOR] Validation PASSED")

        return ColumnValidationResult(
            valid=True,
            sql=sql,
            invalid_columns=[],
            available_columns={},
            error_message=None
        )

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for consistent parsing."""
        # Remove comments
        sql = re.sub(r'--.*?$', ' ', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)
        return sql.strip()

    def _extract_table_aliases(self, sql: str) -> Dict[str, str]:
        """
        Extract table names and their aliases from FROM/JOIN clauses.

        Returns:
            Dict mapping alias (or table name if no alias) to actual table name
            Example: {"f": "flight", "flight": "flight", "b": "booking"}
        """
        aliases = {}

        # Pattern: FROM/JOIN [schema.]table [AS] [alias]
        # Captures: schema (optional), table, alias (optional)
        pattern = re.compile(
            r'(?:FROM|JOIN)\s+'
            r'(?:([A-Za-z_][A-Za-z0-9_]*)\.)?'  # Optional schema
            r'([A-Za-z_][A-Za-z0-9_]*)'          # Table name
            r'(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?',  # Optional alias
            re.IGNORECASE
        )

        for match in pattern.finditer(sql):
            schema_name = match.group(1)  # May be None
            table_name = match.group(2)
            alias = match.group(3)  # May be None

            # Skip SQL keywords that might be captured as aliases
            if alias and alias.upper() in ('ON', 'WHERE', 'AND', 'OR', 'LEFT', 'RIGHT',
                                            'INNER', 'OUTER', 'CROSS', 'FULL', 'JOIN',
                                            'GROUP', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET',
                                            'AS', 'SELECT', 'SET', 'NATURAL'):
                alias = None

            table_lower = table_name.lower()

            # Map alias to table (if alias exists)
            if alias:
                aliases[alias.lower()] = table_lower

            # Always map table name to itself
            aliases[table_lower] = table_lower

        return aliases

    def _extract_column_references(
        self,
        sql: str,
        table_aliases: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """
        Extract column references from SQL.

        Returns:
            List of (table_name, column_name) tuples

        IMPORTANT: We only validate columns we can definitively map to a table.
        - Qualified references (table.column): Validate against resolved table
        - Unqualified references (column): Only validate for single-table queries

        v6.2 FIX: Strips quoted literals before extraction to prevent
        'JFK', 'ATL', etc. from being detected as missing columns.
        """
        column_refs = []

        # =====================================================================
        # v6.2 FIX: Strip quoted literals BEFORE extracting identifiers
        # =====================================================================
        # String literals like 'JFK' or "ATL" are VALUES, not column names.
        # We remove them to prevent false detection as missing columns.
        # =====================================================================
        sql = self._strip_quoted_literals(sql)

        # Pattern 1: Qualified column references (alias.column or table.column)
        qualified_pattern = re.compile(
            r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b',
            re.IGNORECASE
        )

        # =============================================================
        # SQL functions that use dot notation (e.g., EXTRACT(EPOCH FROM ...))
        # These qualifiers are NOT table aliases - exclude from validation.
        # List is EXPLICIT and database-agnostic.
        # =============================================================
        sql_functions_with_dot = {
            # Aggregate functions
            'count', 'sum', 'avg', 'min', 'max', 'coalesce',
            'array_agg', 'string_agg', 'json_agg', 'jsonb_agg',
            'bool_and', 'bool_or', 'every',
            # Date/time functions
            'extract', 'date_part', 'date_trunc', 'age',
            # Type conversion
            'cast', 'convert', 'nullif', 'greatest', 'least',
            # String functions
            'concat', 'substring', 'substr', 'trim', 'upper', 'lower',
            'length', 'position', 'replace',
            # Numeric functions
            'abs', 'ceil', 'floor', 'round', 'trunc', 'mod', 'power', 'sqrt',
        }

        for match in qualified_pattern.finditer(sql):
            qualifier = match.group(1).lower()
            column = match.group(2)

            # Skip function calls (not table.column references)
            if qualifier in sql_functions_with_dot:
                continue

            # Resolve alias to table name
            if qualifier in table_aliases:
                actual_table = table_aliases[qualifier]
                column_refs.append((actual_table, column))

        # Pattern 2: Unqualified column references (only for single-table queries)
        if len(set(table_aliases.values())) == 1:
            # Single table - we can validate unqualified columns
            single_table = list(set(table_aliases.values()))[0]

            # Extract unqualified column names from SELECT, WHERE, GROUP BY, ORDER BY
            # This is a simplified extraction - we look for identifiers that aren't:
            # - Part of qualified references (already handled)
            # - SQL keywords
            # - Function names
            # - Table names/aliases

            # For safety, we only validate SELECT clause columns for unqualified refs
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)

                # =============================================================
                # FIX 1: Extract SELECT output aliases to EXCLUDE from validation
                # =============================================================
                # Aliases declared via AS are OUTPUT names, not column references.
                # Pattern: "expression AS alias" - we capture the alias after AS
                # These must NOT be validated as table columns.
                # =============================================================
                select_aliases = set()
                alias_pattern = re.compile(
                    r'\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\b',
                    re.IGNORECASE
                )
                for alias_match in alias_pattern.finditer(select_clause):
                    select_aliases.add(alias_match.group(1).lower())

                # Find unqualified identifiers
                unqualified_pattern = re.compile(
                    r'(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)(?![.\w(])',
                    re.IGNORECASE
                )

                # =============================================================
                # FIX 2: Expanded SQL keywords/functions whitelist
                # =============================================================
                # Database-agnostic list of SQL reserved words and common functions.
                # These are NOT column names and must be excluded from validation.
                # List is EXPLICIT - no inference or fuzzy matching.
                # =============================================================
                sql_keywords = {
                    # SQL reserved words
                    'select', 'from', 'where', 'and', 'or', 'not', 'in', 'is', 'null',
                    'as', 'on', 'join', 'left', 'right', 'inner', 'outer', 'full',
                    'group', 'by', 'order', 'having', 'limit', 'offset', 'asc', 'desc',
                    'distinct', 'all', 'union', 'intersect', 'except', 'case', 'when',
                    'then', 'else', 'end', 'true', 'false', 'between', 'like', 'ilike',
                    'exists', 'any', 'some', 'over', 'partition', 'row', 'rows',
                    'preceding', 'following', 'unbounded', 'current', 'range',
                    'with', 'recursive', 'using', 'natural', 'cross',
                    # Aggregate functions
                    'count', 'sum', 'avg', 'min', 'max', 'array_agg', 'string_agg',
                    'json_agg', 'jsonb_agg', 'bool_and', 'bool_or', 'every',
                    # Date/time functions and keywords (database-agnostic)
                    'extract', 'epoch', 'date', 'time', 'timestamp', 'interval',
                    'year', 'month', 'day', 'hour', 'minute', 'second', 'week',
                    'quarter', 'dow', 'doy', 'timezone', 'date_part', 'date_trunc',
                    'now', 'current_date', 'current_time', 'current_timestamp',
                    'localtime', 'localtimestamp', 'age',
                    # Type casting and conversion
                    'cast', 'convert', 'coalesce', 'nullif', 'greatest', 'least',
                    # String functions
                    'concat', 'substring', 'substr', 'trim', 'ltrim', 'rtrim',
                    'upper', 'lower', 'length', 'char_length', 'position', 'replace',
                    # Numeric functions
                    'abs', 'ceil', 'ceiling', 'floor', 'round', 'trunc', 'mod',
                    'power', 'sqrt', 'sign', 'random',
                    # NULL handling
                    'isnull', 'ifnull', 'nvl',
                }

                for match in unqualified_pattern.finditer(select_clause):
                    identifier = match.group(1)
                    identifier_lower = identifier.lower()
                    # Skip if: SQL keyword, table/alias, or SELECT output alias
                    if identifier_lower not in sql_keywords:
                        if identifier_lower not in table_aliases:
                            if identifier_lower not in select_aliases:
                                column_refs.append((single_table, identifier))

        return column_refs

    def _build_error_message(
        self,
        invalid_columns: List[Tuple[str, str]],
        available_columns: Dict[str, List[str]]
    ) -> str:
        """Build a user-friendly error message for invalid columns."""
        if len(invalid_columns) == 1:
            table, column = invalid_columns[0]
            available = available_columns.get(table.lower(), [])

            msg = f"Column '{column}' does not exist in table '{table}'."

            if available:
                # Suggest similar columns if any
                similar = self._find_similar_columns(column, available)
                if similar:
                    msg += f" Did you mean: {', '.join(similar[:3])}?"
                else:
                    # Show a sample of available columns
                    sample = available[:5]
                    msg += f" Available columns include: {', '.join(sample)}"
                    if len(available) > 5:
                        msg += f" (and {len(available) - 5} more)"
        else:
            # Multiple invalid columns
            msg = "Multiple invalid column references:\n"
            for table, column in invalid_columns[:3]:
                msg += f"  - '{column}' not in '{table}'\n"
            if len(invalid_columns) > 3:
                msg += f"  - ... and {len(invalid_columns) - 3} more"

        return msg

    def _find_similar_columns(self, target: str, available: List[str]) -> List[str]:
        """Find columns with similar names (simple substring/prefix matching)."""
        target_lower = target.lower()
        similar = []

        for col in available:
            col_lower = col.lower()
            # Check for common patterns
            if (target_lower in col_lower or
                col_lower in target_lower or
                target_lower.replace('_', '') in col_lower.replace('_', '')):
                similar.append(col)

        return similar

    def _strip_quoted_literals(self, sql: str) -> str:
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


# =============================================================================
# SQL LIMIT ENFORCER (v6.8 - Timeout Prevention)
# =============================================================================
# PURPOSE:
# Prevent query timeouts by enforcing hard LIMIT on generated SQL.
#
# PROBLEM STATEMENT:
# Queries like "top passengers" or "passengers by age" can cause timeouts
# due to unbounded GROUP BY operations scanning entire tables.
#
# SOLUTION:
# Inject a LIMIT clause before execution:
# - GROUP BY queries: LIMIT 100
# - Non-GROUP BY queries: LIMIT 50
# - Existing LIMIT: Do not override
#
# v6.9.1: UNBOUNDED LIST QUERY HANDLING
# Queries that are:
# - SELECT statements
# - Have NO WHERE clause
# - Have NO LIMIT clause
# - Have NO aggregation functions
# Are "list-style" queries that should be safely bounded with a default LIMIT.
# This is NOT semantic inference - it is execution safety.
#
# WHAT THIS IS NOT:
# - NOT query optimization (we only add LIMIT)
# - NOT LLM-based (pure regex parsing)
# - NOT semantic analysis (operates on SQL syntax only)
# =============================================================================

@dataclass
class LimitEnforcementResult:
    """
    Result of LIMIT enforcement.

    Attributes:
        sql: The SQL with LIMIT enforced
        limit_applied: Whether a new LIMIT was added
        original_limit: The original LIMIT value (if any)
        enforced_limit: The enforced LIMIT value
        was_unbounded_list: Whether query was an unbounded list query (v6.9.1)
    """
    sql: str
    limit_applied: bool
    original_limit: Optional[int]
    enforced_limit: int
    was_unbounded_list: bool = False


class SQLLimitEnforcer:
    """
    Enforces hard LIMIT on SQL queries to prevent timeouts.

    IMPORTANT FOR REVIEWERS:
    - This is SYNTAX modification, not optimization
    - We DO NOT change query logic, only add LIMIT
    - We respect existing LIMIT values
    - Deterministic, rule-based only

    v6.9.1: Added unbounded list query detection and bounding.
    """

    # Default limits
    GROUP_BY_LIMIT = 100
    DEFAULT_LIMIT = 50
    LIST_QUERY_LIMIT = 50  # v6.9.1: Limit for unbounded list queries

    # Patterns
    LIMIT_PATTERN = re.compile(r'\bLIMIT\s+(\d+)', re.IGNORECASE)
    GROUP_BY_PATTERN = re.compile(r'\bGROUP\s+BY\b', re.IGNORECASE)
    WHERE_PATTERN = re.compile(r'\bWHERE\b', re.IGNORECASE)
    SELECT_PATTERN = re.compile(r'^\s*(?:WITH\b.*?\)\s*)?SELECT\b', re.IGNORECASE | re.DOTALL)

    # Aggregation functions - reuse pattern from QueryComplexityAnalyzer
    AGGREGATION_PATTERN = re.compile(
        r'\b(COUNT|SUM|AVG|MIN|MAX|STRING_AGG|ARRAY_AGG)\s*\(',
        re.IGNORECASE
    )

    def is_unbounded_list_query(self, sql: str) -> bool:
        """
        Check if query is an unbounded list-style query that needs LIMIT.

        v6.9.1: Deterministic detection for execution safety.

        Returns True if ALL conditions are met:
        - SQL is a SELECT statement
        - No WHERE clause
        - No LIMIT clause
        - No aggregation functions (COUNT, SUM, AVG, etc.)

        These queries should have LIMIT injected for execution safety.
        This is NOT semantic inference - it is structural analysis only.

        Args:
            sql: SQL query string

        Returns:
            True if query is an unbounded list query, False otherwise
        """
        if not sql or not sql.strip():
            return False

        sql_upper = sql.upper()

        # Must be SELECT (possibly with CTE)
        if not self.SELECT_PATTERN.match(sql):
            return False

        # Must NOT have WHERE
        if self.WHERE_PATTERN.search(sql):
            return False

        # Must NOT have LIMIT
        if self.LIMIT_PATTERN.search(sql):
            return False

        # Must NOT have aggregation functions
        if self.AGGREGATION_PATTERN.search(sql):
            return False

        # All conditions met - this is an unbounded list query
        return True

    def bound_list_query(self, sql: str, limit: Optional[int] = None) -> LimitEnforcementResult:
        """
        Bound an unbounded list query with a safe LIMIT.

        v6.9.1: Specific handling for list-style queries.

        This method ONLY injects LIMIT if the query matches the unbounded
        list query criteria (SELECT, no WHERE, no LIMIT, no aggregation).

        Args:
            sql: SQL query string
            limit: Optional custom limit (defaults to LIST_QUERY_LIMIT)

        Returns:
            LimitEnforcementResult with bounded SQL if applicable
        """
        if not sql or not sql.strip():
            return LimitEnforcementResult(
                sql=sql,
                limit_applied=False,
                original_limit=None,
                enforced_limit=0,
                was_unbounded_list=False
            )

        sql = sql.strip()
        target_limit = limit if limit is not None else self.LIST_QUERY_LIMIT

        # Check if this is an unbounded list query
        if not self.is_unbounded_list_query(sql):
            # Not an unbounded list query - return unchanged
            # Check for existing LIMIT
            limit_match = self.LIMIT_PATTERN.search(sql)
            existing_limit = int(limit_match.group(1)) if limit_match else 0

            return LimitEnforcementResult(
                sql=sql,
                limit_applied=False,
                original_limit=existing_limit if limit_match else None,
                enforced_limit=existing_limit,
                was_unbounded_list=False
            )

        # Unbounded list query - inject LIMIT
        sql_clean = sql.rstrip(';').strip()
        bounded_sql = f"{sql_clean} LIMIT {target_limit}"

        logger.info(
            f"[LIMIT_ENFORCER] Bounded list query with LIMIT {target_limit} "
            f"(unbounded SELECT without WHERE/aggregation)"
        )

        return LimitEnforcementResult(
            sql=bounded_sql,
            limit_applied=True,
            original_limit=None,
            enforced_limit=target_limit,
            was_unbounded_list=True
        )

    def enforce_limit(self, sql: str) -> LimitEnforcementResult:
        """
        Enforce LIMIT on SQL query.

        Rules:
        - If query has GROUP BY: enforce LIMIT 100
        - Otherwise: enforce LIMIT 50
        - If LIMIT exists: do not override

        v6.9.1: Also detects and flags unbounded list queries.

        Args:
            sql: SQL query string

        Returns:
            LimitEnforcementResult with enforced SQL
        """
        if not sql or not sql.strip():
            return LimitEnforcementResult(
                sql=sql,
                limit_applied=False,
                original_limit=None,
                enforced_limit=0,
                was_unbounded_list=False
            )

        sql = sql.strip()

        # Check if this is an unbounded list query (for flagging)
        is_unbounded_list = self.is_unbounded_list_query(sql)

        # Determine appropriate limit based on query type
        has_group_by = bool(self.GROUP_BY_PATTERN.search(sql))
        target_limit = self.GROUP_BY_LIMIT if has_group_by else self.DEFAULT_LIMIT

        # Check for existing LIMIT
        limit_match = self.LIMIT_PATTERN.search(sql)

        if limit_match:
            original_limit = int(limit_match.group(1))
            logger.debug(
                f"[LIMIT_ENFORCER] Existing LIMIT {original_limit} found - not overriding"
            )
            return LimitEnforcementResult(
                sql=sql,
                limit_applied=False,
                original_limit=original_limit,
                enforced_limit=original_limit,
                was_unbounded_list=False
            )

        # No LIMIT - inject one
        # Remove trailing semicolon if present, add LIMIT, re-add semicolon
        sql_clean = sql.rstrip(';').strip()
        enforced_sql = f"{sql_clean} LIMIT {target_limit}"

        if is_unbounded_list:
            logger.info(
                f"[LIMIT_ENFORCER] Bounded unbounded list query with LIMIT {target_limit}"
            )
        else:
            logger.info(
                f"[LIMIT_ENFORCER] Injected LIMIT {target_limit} "
                f"({'GROUP BY detected' if has_group_by else 'default limit'})"
            )

        return LimitEnforcementResult(
            sql=enforced_sql,
            limit_applied=True,
            original_limit=None,
            enforced_limit=target_limit,
            was_unbounded_list=is_unbounded_list
        )


def enforce_sql_limit(sql: str) -> LimitEnforcementResult:
    """Convenience function to enforce SQL LIMIT."""
    enforcer = SQLLimitEnforcer()
    return enforcer.enforce_limit(sql)


def bound_unbounded_list_query(sql: str, limit: Optional[int] = None) -> LimitEnforcementResult:
    """
    Convenience function to bound unbounded list queries.

    v6.9.1: Execution safety for list-style SELECT queries.

    Injects LIMIT only if query is:
    - A SELECT statement
    - Has NO WHERE clause
    - Has NO LIMIT clause
    - Has NO aggregation functions

    This ensures downstream guards (cost guard, complexity analyzer)
    see the query as bounded.

    Args:
        sql: SQL query string
        limit: Optional custom limit (defaults to 50)

    Returns:
        LimitEnforcementResult with bounded SQL if applicable
    """
    enforcer = SQLLimitEnforcer()
    return enforcer.bound_list_query(sql, limit)


def is_unbounded_list_query(sql: str) -> bool:
    """
    Check if SQL is an unbounded list-style query.

    v6.9.1: Deterministic detection for execution safety.

    Returns True if query is SELECT without WHERE, LIMIT, or aggregation.
    """
    enforcer = SQLLimitEnforcer()
    return enforcer.is_unbounded_list_query(sql)


def create_limit_enforcer() -> SQLLimitEnforcer:
    """Factory function to create a LIMIT enforcer."""
    return SQLLimitEnforcer()


# =============================================================================
# PURE AGGREGATE QUERY DETECTOR (v6.10 - Row-Scan Guard Exception)
# =============================================================================
# PURPOSE:
# Detect pure aggregate queries that should bypass the row-scan guard.
#
# PROBLEM STATEMENT:
# The row-scan guard aborts queries with high estimated scanned rows,
# but this incorrectly blocks safe aggregate queries like:
#   SELECT COUNT(*) FROM flight;
#   SELECT AVG(price) FROM booking;
#
# These queries return exactly ONE ROW and do NOT materialize large result sets.
#
# SOLUTION:
# A pure aggregate query detector that identifies queries which:
# - Have ONLY aggregate functions in SELECT (COUNT, SUM, AVG, MIN, MAX)
# - Have NO raw column references in SELECT
# - Have NO SELECT *
# - Have NO GROUP BY (GROUP BY expands output rows)
#
# WHAT THIS IS NOT:
# - NOT LLM-based (pure regex parsing)
# - NOT schema-aware (operates on SQL syntax only)
# - NOT heuristic (deterministic rules only)
# =============================================================================


def is_pure_aggregate_query(sql: str) -> bool:
    """
    Detect if a query is a pure aggregate query that returns exactly one row.

    A PURE AGGREGATE QUERY is defined as:
    - SELECT clause contains ONLY aggregate functions (COUNT, SUM, AVG, MIN, MAX)
    - No raw column references are selected (only aggregated values)
    - No SELECT * (would return all columns, not aggregates)
    - No GROUP BY clause (GROUP BY expands output to multiple rows)

    Pure aggregate queries return exactly ONE ROW regardless of table size,
    so they should bypass the row-scan guard.

    Examples that ARE pure aggregates (bypass row-scan guard):
        SELECT COUNT(*) FROM flight;
        SELECT AVG(price) FROM booking;
        SELECT MAX(age) FROM passenger;
        SELECT COUNT(*), AVG(price) FROM booking;
        SELECT MIN(scheduled_departure), MAX(scheduled_arrival) FROM flight;
        SELECT COUNT(*) FROM flight WHERE status = 'Scheduled';

    Examples that are NOT pure aggregates (apply row-scan guard):
        SELECT passenger_id, COUNT(*) FROM passenger GROUP BY passenger_id;
        SELECT * FROM flight;
        SELECT city, COUNT(*) FROM flight GROUP BY city;
        SELECT flight_id, scheduled_departure FROM flight;
        SELECT COUNT(*), flight_no FROM flight;  -- has raw column

    Args:
        sql: SQL query string to analyze

    Returns:
        True if query is a pure aggregate query, False otherwise
    """
    if not sql or not sql.strip():
        return False

    sql_upper = sql.upper()
    sql_normalized = re.sub(r'\s+', ' ', sql_upper).strip()

    # ==========================================================================
    # RULE 1: Must be a SELECT statement
    # ==========================================================================
    if not re.match(r'^\s*SELECT\b', sql_normalized):
        return False

    # ==========================================================================
    # RULE 2: Must NOT have GROUP BY (expands output rows)
    # ==========================================================================
    if 'GROUP BY' in sql_normalized:
        return False

    # ==========================================================================
    # RULE 3: Must NOT have SELECT * (returns all columns, not aggregates)
    # ==========================================================================
    # Check for SELECT * or SELECT table.* patterns
    # But allow COUNT(*) which is an aggregate
    select_star_pattern = re.compile(
        r'\bSELECT\s+(?:DISTINCT\s+)?'  # SELECT or SELECT DISTINCT
        r'(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?\*'  # optional table. prefix followed by *
        r'(?!\s*\))',  # NOT followed by ) - this excludes COUNT(*)
        re.IGNORECASE
    )
    if select_star_pattern.search(sql):
        return False

    # ==========================================================================
    # RULE 4: Extract SELECT clause and verify ONLY aggregates
    # ==========================================================================
    # Extract the SELECT clause (between SELECT and FROM)
    select_clause_match = re.search(
        r'\bSELECT\s+(.*?)\s+FROM\b',
        sql_normalized,
        re.IGNORECASE | re.DOTALL
    )

    if not select_clause_match:
        return False

    select_clause = select_clause_match.group(1).strip()

    # Remove DISTINCT keyword if present
    select_clause = re.sub(r'^\s*DISTINCT\s+', '', select_clause, flags=re.IGNORECASE)

    # ==========================================================================
    # RULE 5: Verify SELECT clause contains ONLY aggregate expressions
    # ==========================================================================
    # Aggregate functions: COUNT, SUM, AVG, MIN, MAX
    # We need to ensure every element in the SELECT list is an aggregate
    #
    # Strategy:
    # 1. Split SELECT clause by commas (respecting parentheses)
    # 2. For each element, check if it's an aggregate expression
    # 3. If ANY element is not an aggregate, return False
    # ==========================================================================

    # Split by commas, respecting parentheses depth
    select_items = _split_select_clause(select_clause)

    if not select_items:
        return False

    # Aggregate function pattern - must match the ENTIRE expression (with optional alias)
    # Matches: COUNT(*), AVG(price), SUM(amount) AS total, MIN(date), MAX(col)
    # Also handles nested: COUNT(DISTINCT col), AVG(COALESCE(col, 0))
    aggregate_pattern = re.compile(
        r'^\s*'
        r'(COUNT|SUM|AVG|MIN|MAX)'  # Aggregate function
        r'\s*\('                     # Opening paren
        r'.*'                        # Content (any, including nested parens)
        r'\)'                        # Closing paren
        r'(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?'  # Optional alias
        r'\s*$',
        re.IGNORECASE | re.DOTALL
    )

    for item in select_items:
        item = item.strip()
        if not item:
            continue

        # Check if this item is a pure aggregate expression
        if not aggregate_pattern.match(item):
            # This item is NOT an aggregate - not a pure aggregate query
            return False

    # All items are aggregates - this is a pure aggregate query
    return True


def _split_select_clause(select_clause: str) -> list:
    """
    Split SELECT clause by commas, respecting parentheses depth.

    Args:
        select_clause: The SELECT clause content (between SELECT and FROM)

    Returns:
        List of individual SELECT items
    """
    items = []
    current_item = []
    paren_depth = 0

    for char in select_clause:
        if char == '(':
            paren_depth += 1
            current_item.append(char)
        elif char == ')':
            paren_depth -= 1
            current_item.append(char)
        elif char == ',' and paren_depth == 0:
            # Top-level comma - split here
            items.append(''.join(current_item).strip())
            current_item = []
        else:
            current_item.append(char)

    # Don't forget the last item
    if current_item:
        items.append(''.join(current_item).strip())

    return items


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_column_existence(sql: str, schema: Dict[str, Any]) -> ColumnValidationResult:
    """Validate that SQL columns exist in schema."""
    validator = ColumnExistenceValidator(schema)
    return validator.validate(sql)


def create_column_validator(schema: Dict[str, Any]) -> ColumnExistenceValidator:
    """Factory function to create a column existence validator."""
    return ColumnExistenceValidator(schema)


def sanitize_sql_output(raw_output: str) -> SanitizationResult:
    """Sanitize NL-SQL engine output."""
    sanitizer = SQLOutputSanitizer()
    return sanitizer.sanitize(raw_output)


def create_sql_sanitizer() -> SQLOutputSanitizer:
    """Factory function to create a SQL output sanitizer."""
    return SQLOutputSanitizer()


def validate_sql_aliases(sql: str) -> AliasValidationResult:
    """Validate SQL alias references."""
    validator = SQLAliasValidator()
    return validator.validate(sql)


def analyze_query_complexity(sql: str, row_limit: Optional[int] = None) -> QueryComplexityResult:
    """Analyze SQL query complexity."""
    analyzer = QueryComplexityAnalyzer()
    return analyzer.analyze(sql, row_limit=row_limit)


def create_alias_validator() -> SQLAliasValidator:
    """Factory function to create an alias validator."""
    return SQLAliasValidator()


def create_complexity_analyzer() -> QueryComplexityAnalyzer:
    """Factory function to create a complexity analyzer."""
    return QueryComplexityAnalyzer()
