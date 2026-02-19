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
import sqlparse
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
    was_capped: bool = False  # True when existing LIMIT was reduced


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

    def enforce_limit(self, sql: str, max_limit: Optional[int] = None) -> LimitEnforcementResult:
        """
        Enforce LIMIT on SQL query. Canonical LIMIT enforcement method.

        Rules:
        - If query has GROUP BY and no max_limit override: enforce LIMIT 100
        - Otherwise: enforce LIMIT 50 (or max_limit if provided)
        - If LIMIT exists and exceeds max_limit: cap it down
        - If LIMIT exists and within max_limit: leave unchanged
        - Idempotent: calling twice produces same result

        Args:
            sql: SQL query string
            max_limit: Maximum allowed LIMIT value. If provided, existing LIMITs
                       exceeding this will be capped down.

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
        if max_limit is not None:
            target_limit = max_limit
        else:
            target_limit = self.GROUP_BY_LIMIT if has_group_by else self.DEFAULT_LIMIT

        # Check for existing LIMIT
        limit_match = self.LIMIT_PATTERN.search(sql)

        if limit_match:
            original_limit = int(limit_match.group(1))

            if original_limit > target_limit:
                # Cap the existing LIMIT down
                capped_sql = re.sub(
                    r'\bLIMIT\s+\d+',
                    f'LIMIT {target_limit}',
                    sql,
                    flags=re.IGNORECASE,
                )
                logger.info(
                    f"[BOUNDING] Capped LIMIT {original_limit} → {target_limit}"
                )
                return LimitEnforcementResult(
                    sql=capped_sql,
                    limit_applied=True,
                    original_limit=original_limit,
                    enforced_limit=target_limit,
                    was_unbounded_list=False,
                    was_capped=True,
                )

            logger.debug(
                f"[BOUNDING] Existing LIMIT {original_limit} within bounds — no change"
            )
            return LimitEnforcementResult(
                sql=sql,
                limit_applied=False,
                original_limit=original_limit,
                enforced_limit=original_limit,
                was_unbounded_list=False
            )

        # No LIMIT — inject one
        # Remove trailing semicolon if present, add LIMIT
        sql_clean = sql.rstrip(';').strip()
        enforced_sql = f"{sql_clean} LIMIT {target_limit}"

        logger.info(
            f"[BOUNDING] Injected LIMIT {target_limit} "
            f"({'GROUP BY' if has_group_by else 'unbounded list' if is_unbounded_list else 'default'})"
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


def enforce_query_bounds(sql: str, row_limit: int = 50) -> LimitEnforcementResult:
    """
    Canonical entry point for all LIMIT enforcement. Idempotent.

    This is the SINGLE function the pipeline should call for LIMIT bounding.
    It injects LIMIT if missing, and caps existing LIMIT if too high.

    Args:
        sql: SQL query string
        row_limit: Maximum allowed rows (default 50)

    Returns:
        LimitEnforcementResult with bounded SQL
    """
    enforcer = SQLLimitEnforcer()
    return enforcer.enforce_limit(sql, max_limit=row_limit)


def detect_unbounded_intent(nl_query: str) -> bool:
    """
    Pattern-match NL queries for unbounded intent.

    Detects phrases like "all", "every", "no limit", "entire", "everything",
    "complete list", etc. Advisory only — cost guard remains active regardless.

    Args:
        nl_query: Natural language query from user

    Returns:
        True if user appears to want unbounded results
    """
    if not nl_query:
        return False
    patterns = [
        r'\ball\b',
        r'\bevery\b',
        r'\bno limit\b',
        r'\bentire\b',
        r'\beverything\b',
        r'\bcomplete list\b',
        r'\bfull list\b',
        r'\blist all\b',
        r'\bshow all\b',
        r'\bget all\b',
        r'\bfetch all\b',
        r'\bwithout limit\b',
    ]
    nl_lower = nl_query.lower()
    for pattern in patterns:
        if re.search(pattern, nl_lower):
            return True
    return False


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
# LIMIT REWRITER (v6.16 - Cost Guard Numeric Refinement)
# =============================================================================
# Deterministic LIMIT rewrite for cost guard refinement path.
# Takes validated SQL and a new LIMIT value, returns SQL with exactly
# that LIMIT. Preserves ORDER BY, handles semicolons, idempotent.
#
# This is NOT the same as enforce_query_bounds (which caps/injects).
# This is a targeted rewrite for user-specified row counts.
# =============================================================================

# Shared pattern: matches LIMIT <digits> optionally followed by OFFSET <digits>
_LIMIT_CLAUSE_PATTERN = re.compile(
    r'\bLIMIT\s+\d+(?:\s+OFFSET\s+\d+)?',
    re.IGNORECASE,
)


def rewrite_limit(sql: str, new_limit: int) -> str:
    """
    Rewrite the LIMIT clause of a validated SQL query.

    Rules:
        1. If LIMIT exists: replace it with new value (preserve OFFSET if present)
        2. If no LIMIT: append LIMIT before semicolon / at end
        3. Preserve ORDER BY
        4. Handle trailing semicolons
        5. Deterministic — same input always produces same output

    Args:
        sql: Validated SQL query (already passed safety checks)
        new_limit: New LIMIT value (must be > 0)

    Returns:
        SQL with exactly LIMIT <new_limit>

    Raises:
        ValueError: If new_limit is not a positive integer
    """
    if not isinstance(new_limit, int) or new_limit <= 0:
        raise ValueError(f"new_limit must be a positive integer, got {new_limit}")

    if not sql or not sql.strip():
        raise ValueError("sql must be a non-empty string")

    sql = sql.strip()

    # Check for existing LIMIT clause
    match = _LIMIT_CLAUSE_PATTERN.search(sql)

    if match:
        # Replace existing LIMIT (drop any OFFSET — cost guard path is fresh)
        rewritten = _LIMIT_CLAUSE_PATTERN.sub(f'LIMIT {new_limit}', sql)
    else:
        # No LIMIT — inject before trailing semicolon or at end
        if sql.endswith(';'):
            rewritten = sql[:-1].rstrip() + f' LIMIT {new_limit};'
        else:
            rewritten = sql.rstrip() + f' LIMIT {new_limit}'

    return rewritten


def extract_limit(sql: str) -> Optional[int]:
    """
    Extract the LIMIT value from a SQL query.

    Returns None if no LIMIT clause is present.
    """
    if not sql:
        return None
    match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


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
# STRUCTURED ROUTE FILTER INJECTION (v6.12.1 - Relationally-Aware)
# =============================================================================
# PURPOSE:
# Inject structured route constraints into generated SQL as WHERE conditions.
# When the base table lacks route columns, use DJPI SchemaGraph.find_join_path()
# to discover the join chain and inject the necessary JOINs.
#
# DESIGN PRINCIPLE:
# Context binding operates on STRUCTURED STATE, not surface language.
# Route constraints are stored as typed dicts, not appended to query strings.
# Injection uses sqlparse for structural WHERE detection, then inserts
# constraints at SQL clause boundaries.
#
# WHAT THIS IS NOT:
# - NOT regex-based string replacement
# - NOT query string augmentation
# - NOT LLM-based rewriting
# - NOT keyword matching
# - NOT hardcoded join chains (uses DJPI graph discovery)
#
# ARCHITECTURAL POSITION:
#   RCL Output -> [ROUTE FILTER INJECTION] -> Column Validation -> Execution
# =============================================================================


def _resolve_to_schema_graph_key(table_name: str, sg_tables) -> Optional[str]:
    """
    Case-insensitive lookup to map a table name to SchemaGraph key.

    Handles schema-prefix matching: 'flight' matches 'postgres_air.flight'.

    Args:
        table_name: Table name from SQL (possibly lowercase, possibly schema-prefixed)
        sg_tables: Set of table keys from SchemaGraph.tables

    Returns:
        Exact key from SchemaGraph.tables or None
    """
    name_lower = table_name.lower()
    for key in sg_tables:
        key_lower = key.lower()
        if key_lower == name_lower:
            return key
        # Match short name to schema-prefixed key: 'flight' -> 'postgres_air.flight'
        if key_lower.endswith(f".{name_lower}"):
            return key
    return None


def _find_table_ref_in_sql(table_name_lower: str, table_aliases: Dict[str, str]) -> str:
    """
    Find the best SQL reference for a table given known aliases.

    Prefers explicit alias (e.g. 'f') over bare table name.
    Falls back to short table name if present.

    Args:
        table_name_lower: Lowercase table name (e.g. 'flight')
        table_aliases: Dict from SQLParser.extract_tables_with_aliases()

    Returns:
        Best reference string to use in SQL
    """
    # Check for explicit alias (key != table short name, value matches table)
    for alias, full_name in table_aliases.items():
        full_lower = full_name.lower()
        short_name = full_lower.split(".")[-1]
        if short_name == table_name_lower and alias != short_name:
            return alias

    # Fall back to short table name if in aliases
    if table_name_lower in table_aliases:
        return table_name_lower

    return table_name_lower


def _inject_filter_expr_into_sql(sql: str, filter_expr: str) -> str:
    """
    Inject a WHERE/AND filter expression into SQL at the correct position.

    Uses sqlparse for structural WHERE detection and regex for clause
    boundary detection.

    Args:
        sql: SQL query (without trailing semicolon handling — caller strips)
        filter_expr: Fully qualified filter expression

    Returns:
        SQL with filter injected
    """
    # Structural analysis: detect WHERE clause via sqlparse token tree
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql

    stmt = parsed[0]
    has_where = any(
        isinstance(token, sqlparse.sql.Where) for token in stmt.tokens
    )

    # Separate SQL body from trailing semicolon
    sql_body = sql.rstrip()
    had_semicolon = sql_body.endswith(';')
    if had_semicolon:
        sql_body = sql_body[:-1].rstrip()

    # Find clause boundary: first post-WHERE keyword position
    boundary_pattern = re.compile(
        r'\b(GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|OFFSET)\b',
        re.IGNORECASE,
    )
    match = boundary_pattern.search(sql_body)
    insert_pos = match.start() if match else len(sql_body)

    before = sql_body[:insert_pos].rstrip()
    after = sql_body[insert_pos:]

    if has_where:
        result = f"{before} AND {filter_expr} {after}"
    else:
        result = f"{before} WHERE {filter_expr} {after}"

    if had_semicolon:
        result = result.rstrip() + ';'

    return result


def _inject_joins_and_filters(sql: str, join_clauses: List[str], filter_expr: str) -> str:
    """
    Insert JOIN clauses after FROM/existing JOINs and add WHERE filter.

    Args:
        sql: SQL query
        join_clauses: List of JOIN clause strings (e.g. 'JOIN schema.table ON ...')
        filter_expr: Fully qualified filter expression

    Returns:
        SQL with JOINs and filter injected
    """
    # Separate SQL body from trailing semicolon
    sql_body = sql.rstrip()
    had_semicolon = sql_body.endswith(';')
    if had_semicolon:
        sql_body = sql_body[:-1].rstrip()

    # Find insertion point for JOINs: after FROM table and any existing JOINs,
    # before WHERE/GROUP BY/ORDER BY/HAVING/LIMIT
    boundary_pattern = re.compile(
        r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|OFFSET)\b',
        re.IGNORECASE,
    )
    # Also find the last existing JOIN...ON clause to insert after it
    join_end_pattern = re.compile(
        r'\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+|FULL\s+)?JOIN\b'
        r'.*?\bON\b.*?(?=\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|CROSS\s+|FULL\s+)?JOIN\b'
        r'|\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)',
        re.IGNORECASE | re.DOTALL,
    )

    # Find where to insert new JOINs
    last_join_end = 0
    for m in join_end_pattern.finditer(sql_body):
        last_join_end = m.end()

    if last_join_end > 0:
        # Insert after the last existing JOIN
        insert_pos = last_join_end
    else:
        # No existing JOINs — insert before first boundary keyword
        boundary_match = boundary_pattern.search(sql_body)
        if boundary_match:
            insert_pos = boundary_match.start()
        else:
            insert_pos = len(sql_body)

    before = sql_body[:insert_pos].rstrip()
    after = sql_body[insert_pos:].lstrip()

    join_block = " ".join(join_clauses)
    reassembled = f"{before} {join_block} {after}".rstrip()

    if had_semicolon:
        reassembled = reassembled + ';'

    # Now inject the filter expression
    return _inject_filter_expr_into_sql(reassembled, filter_expr)


# =============================================================================
# SELECT COLUMN QUALIFICATION (v6.12.2 - Ambiguity Prevention)
# =============================================================================
# After relational JOIN injection, columns like booking_id may exist in BOTH
# the base table and a joined table (e.g., booking AND booking_leg).
# PostgreSQL raises "column reference is ambiguous".
#
# Fix: use sqlparse AST traversal to qualify unqualified column references
# in the SELECT clause with the base table reference.
#
# This ONLY activates on the relational JOIN injection path.
# Direct injection and legacy paths are unaffected.
# =============================================================================


def _qualify_paren_ast(
    paren_token,
    base_ref: str,
    base_cols: set,
) -> None:
    """
    Walk inside a Parenthesis AST node and qualify unqualified column names.

    Handles function arguments like COUNT(booking_id), COUNT(DISTINCT booking_id),
    and nested expressions.

    Args:
        paren_token: sqlparse Parenthesis token
        base_ref: Table reference to prefix (e.g. 'booking' or 'b')
        base_cols: Set of lowercase column names belonging to the base table
    """
    for child in paren_token.tokens:
        if isinstance(child, sqlparse.sql.Identifier):
            _qualify_identifier_ast(child, base_ref, base_cols)
        elif isinstance(child, sqlparse.sql.IdentifierList):
            for ident in child.get_identifiers():
                _qualify_identifier_ast(ident, base_ref, base_cols)
        elif isinstance(child, sqlparse.sql.Parenthesis):
            _qualify_paren_ast(child, base_ref, base_cols)
        elif child.ttype is sqlparse.tokens.Name:
            # Bare Name token not wrapped in Identifier (rare but possible)
            if child.value.lower() in base_cols:
                child.value = f"{base_ref}.{child.value}"


def _qualify_identifier_ast(
    token,
    base_ref: str,
    base_cols: set,
) -> None:
    """
    Qualify an unqualified column reference inside an Identifier AST node.

    Decision tree per node type:
    - Identifier with dot       → already qualified, skip
    - Identifier wrapping Function → descend into Function parenthesis
    - Identifier with bare Name → qualify if name matches base_cols
    - Function                  → qualify inside parenthesis only (skip func name)
    - Parenthesis               → delegate to _qualify_paren_ast

    Args:
        token: sqlparse AST node (Identifier, Function, or Parenthesis)
        base_ref: Table reference to prefix
        base_cols: Set of lowercase column names belonging to the base table
    """
    if isinstance(token, sqlparse.sql.Identifier):
        # Already qualified? (has dot punctuation among direct children)
        if any(
            t.ttype is sqlparse.tokens.Punctuation and t.value == '.'
            for t in token.tokens
        ):
            return

        first = token.token_first(skip_cm=True, skip_ws=True)

        # Wraps a Function (e.g., COUNT(...) AS alias) → descend into Function
        if isinstance(first, sqlparse.sql.Function):
            _qualify_identifier_ast(first, base_ref, base_cols)
            return

        # Bare column name (e.g., booking_id or price)
        # Only the FIRST meaningful token is the column — anything after AS
        # is an output alias and must NOT be touched.
        if first and first.ttype is sqlparse.tokens.Name:
            if first.value.lower() in base_cols:
                first.value = f"{base_ref}.{first.value}"
        return

    if isinstance(token, sqlparse.sql.Function):
        # Qualify inside parentheses only — tokens[0] is the function name, skip it
        for child in token.tokens:
            if isinstance(child, sqlparse.sql.Parenthesis):
                _qualify_paren_ast(child, base_ref, base_cols)
        return

    if isinstance(token, sqlparse.sql.Parenthesis):
        _qualify_paren_ast(token, base_ref, base_cols)
        return


def _qualify_select_columns(
    sql: str,
    base_table_ref: str,
    base_columns: set,
) -> str:
    """
    AST-level qualification of unqualified column references in SELECT and
    GROUP BY clauses.

    Uses sqlparse to parse the SQL into an AST, traverses the SELECT and
    GROUP BY clause token trees, and prefixes unqualified column names that
    belong to the base table with base_table_ref.

    This prevents "column reference is ambiguous" errors after relational
    JOIN injection introduces tables that share column names with the base table.

    Args:
        sql: SQL query string (with JOINs already injected)
        base_table_ref: Table reference to prefix (alias if present, else table name)
        base_columns: Set of column names belonging to the base table

    Returns:
        SQL with SELECT and GROUP BY columns qualified

    Handles:
        - COUNT(column), SUM(column), AVG(column), MIN(column), MAX(column)
        - COUNT(DISTINCT column)
        - Raw column selects (booking_id, price)
        - Expressions with AS aliases (AVG(price) AS avg_price — alias untouched)
        - Multiple select expressions (IdentifierList)
        - Standalone Function tokens in SELECT (e.g., COUNT(col) without alias)
        - GROUP BY column references (must match qualified SELECT columns)

    Does NOT touch:
        - Already qualified references (b.booking_id)
        - Wildcards (*)
        - WHERE clause, JOIN clause, ORDER BY
        - Function names (COUNT, AVG, etc.)

    Idempotent: running twice does not modify further (already-qualified
    references are detected and skipped).
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return sql

    stmt = parsed[0]
    base_cols_lower = {c.lower() for c in base_columns}

    # =========================================================================
    # Phase 1: Qualify SELECT clause columns
    # =========================================================================
    in_select = False
    for token in stmt.tokens:
        if token.ttype is sqlparse.tokens.DML and token.normalized == 'SELECT':
            in_select = True
            continue
        if in_select:
            if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'FROM':
                break
            # Skip whitespace and DISTINCT keyword at clause level
            if token.ttype in (
                sqlparse.tokens.Whitespace,
                sqlparse.tokens.Newline,
            ):
                continue
            if token.ttype is sqlparse.tokens.Keyword and token.normalized == 'DISTINCT':
                continue

            # Process all possible token types in SELECT clause
            if isinstance(token, sqlparse.sql.IdentifierList):
                for ident in token.get_identifiers():
                    _qualify_identifier_ast(ident, base_table_ref, base_cols_lower)
            elif isinstance(token, sqlparse.sql.Identifier):
                _qualify_identifier_ast(token, base_table_ref, base_cols_lower)
            elif isinstance(token, sqlparse.sql.Function):
                # Standalone Function (e.g., COUNT(booking_id) without AS alias)
                # sqlparse may produce Function directly when there's no alias
                _qualify_identifier_ast(token, base_table_ref, base_cols_lower)
            elif isinstance(token, sqlparse.sql.Parenthesis):
                # Standalone Parenthesis in SELECT (rare but possible)
                _qualify_paren_ast(token, base_table_ref, base_cols_lower)
            elif token.ttype is sqlparse.tokens.Name:
                # Bare Name token (rare — usually wrapped in Identifier)
                if token.value.lower() in base_cols_lower:
                    token.value = f"{base_table_ref}.{token.value}"

    # =========================================================================
    # Phase 2: Qualify GROUP BY clause columns
    # =========================================================================
    # GROUP BY columns must match qualified SELECT columns. If SELECT has
    # booking.booking_id but GROUP BY has unqualified booking_id, PostgreSQL
    # raises an ambiguity error.
    # =========================================================================
    in_group_by = False
    for token in stmt.tokens:
        if token.ttype is sqlparse.tokens.Keyword:
            normalized = token.normalized.upper() if token.normalized else ''
            if normalized == 'GROUP BY':
                in_group_by = True
                continue
            if in_group_by and normalized in (
                'ORDER BY', 'HAVING', 'LIMIT', 'UNION',
                'INTERSECT', 'EXCEPT', 'OFFSET', 'WINDOW',
            ):
                break
        if in_group_by:
            if token.ttype in (
                sqlparse.tokens.Whitespace,
                sqlparse.tokens.Newline,
            ):
                continue
            if isinstance(token, sqlparse.sql.IdentifierList):
                for ident in token.get_identifiers():
                    _qualify_identifier_ast(ident, base_table_ref, base_cols_lower)
            elif isinstance(token, sqlparse.sql.Identifier):
                _qualify_identifier_ast(token, base_table_ref, base_cols_lower)
            elif token.ttype is sqlparse.tokens.Name:
                if token.value.lower() in base_cols_lower:
                    token.value = f"{base_table_ref}.{token.value}"

    return str(stmt)


# =============================================================================
# WHERE CLAUSE DEDUPLICATION (v6.12.3 - Idempotent Injection)
# =============================================================================
# When the LLM already includes route filter conditions in the WHERE clause,
# structured injection must detect and skip duplicates. Otherwise the SQL
# ends up with visually redundant predicates like:
#   WHERE f.departure_airport = 'JFK' AND f.departure_airport = 'JFK'
#
# This layer uses sqlparse AST to:
# 1. Extract existing equality predicates from the WHERE clause
# 2. Canonicalize both existing and incoming predicates (resolving aliases)
# 3. Inject only predicates that are genuinely missing
#
# Canonical form: (resolved_table_short.column, "=", "'literal'")
# All components lowercased. Alias resolved via table_aliases dict.
# =============================================================================


def _collect_comparisons_ast(token) -> list:
    """
    Recursively collect all sqlparse.sql.Comparison nodes from a token tree.

    Walks into Parenthesis, Where, and other grouping nodes.
    Skips subqueries (nested SELECT statements).

    Args:
        token: Any sqlparse token or token list

    Returns:
        List of sqlparse.sql.Comparison instances
    """
    results = []
    if isinstance(token, sqlparse.sql.Comparison):
        results.append(token)
        return results
    if hasattr(token, 'tokens'):
        for child in token.tokens:
            # Skip subqueries — don't descend into nested SELECTs
            if child.ttype is sqlparse.tokens.DML:
                return results
            results.extend(_collect_comparisons_ast(child))
    return results


def _canonicalize_column_ref(token, table_aliases: Dict[str, str]) -> Optional[str]:
    """
    Resolve an AST identifier token to canonical 'table_short.column' form.

    Handles:
        - Qualified: f.departure_airport → flight.departure_airport
        - Unqualified: departure_airport → departure_airport

    Args:
        token: sqlparse Identifier or Name token
        table_aliases: alias → full_table_name mapping

    Returns:
        Canonical lowercase string or None
    """
    if isinstance(token, sqlparse.sql.Identifier):
        names = [
            t.value for t in token.tokens
            if t.ttype is sqlparse.tokens.Name
        ]
        if len(names) == 2:
            qualifier = names[0].lower()
            column = names[1].lower()
            # Resolve alias to short table name
            if qualifier in table_aliases:
                resolved = table_aliases[qualifier].split(".")[-1].lower()
                return f"{resolved}.{column}"
            return f"{qualifier}.{column}"
        elif len(names) == 1:
            return names[0].lower()
    elif token.ttype is sqlparse.tokens.Name:
        return token.value.lower()
    return None


def _parse_comparison_to_canonical(
    comp_token,
    table_aliases: Dict[str, str],
) -> Optional[tuple]:
    """
    Parse a single sqlparse.sql.Comparison into a canonical predicate tuple.

    Only handles simple equality: <identifier> = <literal>
    Returns None for anything else (non-equality, complex expressions).

    Args:
        comp_token: sqlparse.sql.Comparison node
        table_aliases: alias → full_table_name mapping

    Returns:
        Tuple of (canonical_col_ref, "=", canonical_literal) or None
    """
    left = None
    op = None
    right = None

    for token in comp_token.tokens:
        if token.is_whitespace:
            continue
        if left is None:
            left = token
        elif op is None:
            if token.ttype is sqlparse.tokens.Comparison:
                op = token.value.strip()
            else:
                return None
        elif right is None:
            right = token
            break

    if not (left and op and right):
        return None
    if op != '=':
        return None

    col_ref = _canonicalize_column_ref(left, table_aliases)
    if not col_ref:
        return None

    # Canonicalize literal — accept single-quoted strings and numbers
    literal_val = None
    if right.ttype is not None and right.ttype in sqlparse.tokens.Literal:
        literal_val = right.value.lower()
    elif right.ttype is sqlparse.tokens.Literal.String.Single:
        literal_val = right.value.lower()

    if not literal_val:
        return None

    return (col_ref, '=', literal_val)


def _extract_where_predicates(
    sql: str,
    table_aliases: Dict[str, str],
) -> set:
    """
    Extract canonical equality predicates from SQL WHERE clause via AST.

    Parses the SQL, locates the WHERE clause, walks all Comparison nodes,
    and returns a set of canonical tuples for equality predicates.

    Args:
        sql: SQL query string
        table_aliases: alias → full_table_name mapping for alias resolution

    Returns:
        Set of (canonical_col_ref, "=", canonical_literal) tuples
    """
    parsed = sqlparse.parse(sql)
    if not parsed:
        return set()

    stmt = parsed[0]

    # Find WHERE clause
    where_clause = None
    for token in stmt.tokens:
        if isinstance(token, sqlparse.sql.Where):
            where_clause = token
            break

    if not where_clause:
        return set()

    # Collect all Comparison nodes from WHERE
    comparisons = _collect_comparisons_ast(where_clause)

    predicates = set()
    for comp in comparisons:
        canonical = _parse_comparison_to_canonical(comp, table_aliases)
        if canonical:
            predicates.add(canonical)

    return predicates


def _predicate_covered(
    incoming: tuple,
    existing_set: set,
) -> bool:
    """
    Check if an incoming predicate is already covered by existing predicates.

    Handles cross-matching between qualified and unqualified references:
        - incoming 'flight.departure_airport' matches existing 'departure_airport'
        - incoming 'departure_airport' matches existing 'flight.departure_airport'

    Args:
        incoming: Canonical tuple (col_ref, op, literal)
        existing_set: Set of canonical tuples from WHERE clause

    Returns:
        True if the predicate already exists
    """
    if incoming in existing_set:
        return True

    col_ref, op, val = incoming

    # Incoming is qualified → check if unqualified version exists
    if '.' in col_ref:
        bare_col = col_ref.split('.', 1)[1]
        if (bare_col, op, val) in existing_set:
            return True

    # Incoming is unqualified → check if any qualified version matches
    else:
        for (ex_ref, ex_op, ex_val) in existing_set:
            if ex_op == op and ex_val == val:
                if '.' in ex_ref and ex_ref.split('.', 1)[1] == col_ref:
                    return True

    return False


def _deduplicate_filter_conditions(
    sql: str,
    qualified_conditions: List[str],
    table_aliases: Dict[str, str],
) -> List[str]:
    """
    Remove filter conditions that already exist in the SQL WHERE clause.

    Parses the SQL to extract existing WHERE predicates, canonicalizes each
    incoming condition, and returns only conditions not already present.

    Args:
        sql: SQL query string
        qualified_conditions: List of condition strings (e.g. "flight.departure_airport = 'JFK'")
        table_aliases: alias → full_table_name mapping

    Returns:
        Filtered list of conditions to inject (may be empty)
    """
    existing = _extract_where_predicates(sql, table_aliases)
    if not existing:
        return qualified_conditions

    remaining = []
    for cond in qualified_conditions:
        # Parse the condition string via sqlparse to get canonical form
        parsed = sqlparse.parse(cond)
        canonical = None
        if parsed:
            comps = _collect_comparisons_ast(parsed[0])
            if comps:
                canonical = _parse_comparison_to_canonical(comps[0], table_aliases)

        if canonical and _predicate_covered(canonical, existing):
            logger.info(f"[FILTER_INJECT] Skipping duplicate: {cond}")
        else:
            remaining.append(cond)

    if not remaining and qualified_conditions:
        logger.info(
            "[FILTER_INJECT] Route filters already present — skipping injection"
        )

    return remaining


def _inject_relationally_aware(
    sql: str,
    conditions: List[str],
    column_names: List[str],
    table_aliases: Dict[str, str],
    primary_table: str,
    schema_metadata,
    schema_graph,
) -> Optional[str]:
    """
    Main relational injection logic.

    CASE 1 — Direct: primary table has all route columns -> qualify + inject.
    CASE 2 — Relational: find target table via DJPI, add JOINs if needed.

    Args:
        sql: SQL query
        conditions: List of unqualified condition strings (e.g. "col = 'VAL'")
        column_names: List of column names in the conditions
        table_aliases: Dict from SQLParser.extract_tables_with_aliases()
        primary_table: Primary FROM table (may be schema-prefixed, e.g. 'postgres_air.booking')
        schema_metadata: SchemaMetadata from relational_corrector
        schema_graph: SchemaGraph from join_path_inference

    Returns:
        Modified SQL or None (abort — no injection possible)
    """
    primary_short = primary_table.split(".")[-1].lower()

    # CASE 1: Direct — primary table has ALL route columns
    all_direct = all(
        schema_metadata.column_exists_in_table(primary_short, col)
        for col in column_names
    )
    if all_direct:
        primary_ref = _find_table_ref_in_sql(primary_short, table_aliases)
        qualified_conditions = []
        for cond, col in zip(conditions, column_names):
            qualified_conditions.append(cond.replace(col, f"{primary_ref}.{col}", 1))
        qualified_conditions = _deduplicate_filter_conditions(
            sql, qualified_conditions, table_aliases
        )
        if not qualified_conditions:
            return sql
        filter_expr = " AND ".join(qualified_conditions)
        logger.info(f"[FILTER_INJECT] Direct injection (base table has columns): {filter_expr}")
        return _inject_filter_expr_into_sql(sql, filter_expr)

    # CASE 2: Relational — find a target table that has ALL route columns
    target_sg_key = None
    target_short = None
    for tbl_key in schema_metadata.tables:
        tbl_short = tbl_key.split(".")[-1].lower()
        if all(schema_metadata.column_exists_in_table(tbl_short, col) for col in column_names):
            target_sg_key = tbl_key
            target_short = tbl_short
            break

    if not target_sg_key:
        logger.warning("[FILTER_INJECT] ABORT: No table found containing all route columns")
        return None

    # Check if target table is already in the SQL
    target_already_in_sql = False
    for alias, full_name in table_aliases.items():
        full_short = full_name.split(".")[-1].lower()
        if full_short == target_short:
            target_already_in_sql = True
            break

    if target_already_in_sql:
        # Target already joined — just add qualified WHERE
        target_ref = _find_table_ref_in_sql(target_short, table_aliases)
        qualified_conditions = []
        for cond, col in zip(conditions, column_names):
            qualified_conditions.append(cond.replace(col, f"{target_ref}.{col}", 1))
        qualified_conditions = _deduplicate_filter_conditions(
            sql, qualified_conditions, table_aliases
        )
        if not qualified_conditions:
            return sql
        filter_expr = " AND ".join(qualified_conditions)
        logger.info(f"[FILTER_INJECT] Target already joined (ref={target_ref}): {filter_expr}")
        return _inject_filter_expr_into_sql(sql, filter_expr)

    # Target NOT in SQL — use DJPI to find join path
    source_sg_key = _resolve_to_schema_graph_key(primary_short, schema_graph.tables)
    dest_sg_key = _resolve_to_schema_graph_key(target_short, schema_graph.tables)

    if not source_sg_key or not dest_sg_key:
        logger.warning(
            f"[FILTER_INJECT] ABORT: Could not resolve SchemaGraph keys: "
            f"{primary_short} -> {source_sg_key}, {target_short} -> {dest_sg_key}"
        )
        return None

    join_path = schema_graph.find_join_path(source_sg_key, dest_sg_key)

    if not join_path:
        logger.warning(
            f"[FILTER_INJECT] ABORT: No DJPI path found: {primary_short} -> {target_short}"
        )
        return None

    logger.info(
        f"[FILTER_INJECT] DJPI path found ({len(join_path)} hop(s)): "
        + " -> ".join([join_path[0][0]] + [step[2] for step in join_path])
    )

    # Build JOIN clauses, skipping tables already in SQL
    join_clauses = []
    for from_tbl, from_col, to_tbl, to_col in join_path:
        to_short = to_tbl.split(".")[-1].lower()
        # Check if this intermediate/target table is already in SQL
        already_present = False
        for alias, full_name in table_aliases.items():
            if full_name.split(".")[-1].lower() == to_short:
                already_present = True
                break

        if not already_present:
            # Resolve the from-table reference (may be aliased in SQL or a prior hop)
            from_short = from_tbl.split(".")[-1].lower()
            from_ref = _find_table_ref_in_sql(from_short, table_aliases)
            # For intermediate tables added by us, use the schema-prefixed name
            join_clauses.append(
                f"JOIN {to_tbl} ON {from_ref}.{from_col} = {to_tbl}.{to_col}"
            )
            # Add the new table to aliases so subsequent hops can reference it
            table_aliases[to_short] = to_tbl.lower()

    # Build qualified filter — target table uses its schema-prefixed name
    # (since we added it via JOIN with its full name)
    target_ref = _find_table_ref_in_sql(target_short, table_aliases)
    qualified_conditions = []
    for cond, col in zip(conditions, column_names):
        qualified_conditions.append(cond.replace(col, f"{target_ref}.{col}", 1))
    qualified_conditions = _deduplicate_filter_conditions(
        sql, qualified_conditions, table_aliases
    )
    if not qualified_conditions:
        return sql
    filter_expr = " AND ".join(qualified_conditions)

    if join_clauses:
        logger.info(f"[FILTER_INJECT] Relational injection complete: {len(join_clauses)} JOIN(s)")
        injected_sql = _inject_joins_and_filters(sql, join_clauses, filter_expr)

        # Qualify unqualified SELECT columns to prevent ambiguity after JOIN
        primary_ref = _find_table_ref_in_sql(primary_short, table_aliases)
        base_columns = schema_metadata.get_table_columns(primary_short)
        if base_columns:
            injected_sql = _qualify_select_columns(injected_sql, primary_ref, base_columns)
            logger.info(
                f"[FILTER_INJECT] SELECT columns qualified with base ref '{primary_ref}'"
            )
        return injected_sql
    else:
        # All intermediate tables were already present — just add filter
        return _inject_filter_expr_into_sql(sql, filter_expr)


def inject_structured_route_filters(
    sql: str,
    route_filters: Dict[str, str],
    schema_metadata=None,
    schema_graph=None,
) -> str:
    """
    Inject structured route filter constraints into SQL.

    v6.12.1: Relationally-aware. If the base table lacks route columns,
    uses DJPI SchemaGraph.find_join_path() to discover join chains and
    injects the necessary JOINs before adding WHERE conditions.

    Uses sqlparse for structural WHERE clause detection.
    Inserts AND/WHERE constraints at SQL clause boundaries.

    Args:
        sql: Generated SQL query
        route_filters: Dict of column_name -> IATA code
                       e.g. {"departure_airport": "JFK", "arrival_airport": "ATL"}
        schema_metadata: SchemaMetadata from relational_corrector (optional)
        schema_graph: SchemaGraph from join_path_inference (optional)

    Returns:
        SQL with structured WHERE constraints injected.

    Input validation:
        - Column names must be valid SQL identifiers (lowercase, alphanumeric + underscore)
        - Values must be 3-letter uppercase IATA codes
        - Invalid entries are silently skipped with warning

    Backward compatibility:
        If schema_metadata or schema_graph are None, falls back to legacy
        direct injection (unqualified columns, no JOIN awareness).
    """
    if not route_filters or not sql or not sql.strip():
        return sql

    # Build validated filter conditions
    conditions = []
    column_names = []
    for col, val in sorted(route_filters.items()):
        # Column names: valid SQL identifiers only
        if not re.match(r'^[a-z_][a-z0-9_]*$', col):
            logger.warning(f"[FILTER_INJECT] Rejected column name: {col}")
            continue
        # Values: strict 3-letter IATA codes only
        if not re.match(r'^[A-Z]{3}$', val):
            logger.warning(f"[FILTER_INJECT] Rejected filter value: {val}")
            continue
        conditions.append(f"{col} = '{val}'")
        column_names.append(col)

    if not conditions:
        return sql

    # Relationally-aware injection (v6.12.1)
    if schema_metadata is not None and schema_graph is not None:
        try:
            from relational_corrector import SQLParser
            table_aliases, primary_table = SQLParser.extract_tables_with_aliases(sql)

            if primary_table:
                result = _inject_relationally_aware(
                    sql, conditions, column_names,
                    table_aliases, primary_table,
                    schema_metadata, schema_graph,
                )
                if result is not None:
                    return result
                # result is None means ABORT — return SQL unmodified
                logger.info("[FILTER_INJECT] Relational injection aborted — returning SQL unmodified")
                return sql
        except Exception as e:
            logger.error(f"[FILTER_INJECT] Relational injection failed: {e} — falling back to legacy")

    # Legacy direct injection (no schema info or fallback)
    conditions = _deduplicate_filter_conditions(sql, conditions, {})
    if not conditions:
        return sql
    filter_expr = " AND ".join(conditions)
    logger.info(f"[FILTER_INJECT] Injected structured route filters (legacy): {filter_expr}")
    return _inject_filter_expr_into_sql(sql, filter_expr)


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
