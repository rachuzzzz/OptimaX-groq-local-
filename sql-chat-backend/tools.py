"""
OptimaX Tools - SQL Tools + One-Shot Visualization Classifier
==============================================================

Tools:
1. SQL Query Execution (via ReActAgent)
2. One-shot Visualization Classification (separate LLM call, no agent)

NEW in v4.2: classify_visualization_intent()
- No agent, no tools, no retries
- Pure LLM classification for chart type suggestions
- Eliminates timeouts and infinite loops

Author: OptimaX Team
Version: 4.2 (Split Visualization Architecture)
"""

import logging
from typing import Dict, Any, List
from sqlalchemy import create_engine, text, inspect
import sqlparse
import re
import json
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)

# Global variables to store last tool execution results
_last_sql_result = None
_last_chart_config = None


class DatabaseManager:
    """Manages database connection and schema information"""

    def __init__(self, database_url: str):
        """Initialize database connection"""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.active_schema = None  # Will be auto-detected
        self.schema = self._load_schema()
        self.cached_schema = None  # Free tier optimization: cache schema text
        logger.info(f"Database connected successfully (Schema: {self.active_schema or 'public'})")

    def _load_schema(self) -> Dict[str, Any]:
        """Load database schema information (lazy loading for faster startup)"""
        try:
            inspector = inspect(self.engine)
            schema_info = {"tables": {}}

            # Auto-detect available schemas (exclude system schemas)
            available_schemas = inspector.get_schema_names()
            system_schemas = ['information_schema', 'pg_catalog', 'pg_toast', 'pg_temp_1',
                            'pg_toast_temp_1', 'pg_statistic', 'mysql', 'sys', 'performance_schema']
            user_schemas = [s for s in available_schemas if s not in system_schemas]

            # Use the first user schema found, or 'public' as fallback
            if user_schemas:
                self.active_schema = user_schemas[0]
                logger.info(f"Detected user schema: {self.active_schema}")
            else:
                # Try 'public' schema (default for PostgreSQL/MySQL)
                self.active_schema = 'public'
                logger.info("Using default 'public' schema")

            # Load tables from the active schema
            try:
                tables = inspector.get_table_names(schema=self.active_schema)
            except:
                # Fallback: try without schema (for databases that don't use schemas)
                tables = inspector.get_table_names()
                self.active_schema = None  # No schema prefix needed
                logger.info("Database doesn't use schemas - loading tables without prefix")

            # Load all tables
            for table in tables:
                if self.active_schema:
                    columns = inspector.get_columns(table, schema=self.active_schema)
                    full_table_name = f"{self.active_schema}.{table}"
                else:
                    columns = inspector.get_columns(table)
                    full_table_name = table

                schema_info["tables"][full_table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                        }
                        for col in columns
                    ]
                }

            schema_name = self.active_schema or 'default'
            logger.info(f"Schema loaded: {len(schema_info['tables'])} tables from '{schema_name}' schema")
            return schema_info

        except Exception as e:
            logger.error(f"Failed to load schema: {str(e)}")
            return {"tables": {}}

    def get_schema_text(self) -> str:
        """Get human-readable schema description (cached for free tier)"""
        # Free tier optimization: return cached schema if available
        if self.cached_schema:
            return self.cached_schema

        table_count = len(self.schema["tables"])
        lines = [f"Database Schema ({table_count} tables):\n"]

        for table_name, table_info in self.schema["tables"].items():
            lines.append(f"\nTable: {table_name}")
            lines.append("Columns:")
            for col in table_info["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                lines.append(f"  - {col['name']}: {col['type']} ({nullable})")

        self.cached_schema = "\n".join(lines)
        return self.cached_schema

    def get_schema_for_llm(self) -> str:
        """Get formatted schema description optimized for LLM system prompts"""
        lines = []

        # Count tables and get basic stats
        table_count = len(self.schema["tables"])

        if table_count == 0:
            return "No tables found in database."

        lines.append(f"DATABASE SCHEMA ({table_count} tables):\n")

        for table_name, table_info in self.schema["tables"].items():
            # Create compact column list
            column_names = [col["name"] for col in table_info["columns"]]

            # Group columns by type for better readability
            key_columns = []
            time_columns = []
            text_columns = []
            numeric_columns = []
            other_columns = []

            for col in table_info["columns"]:
                col_name = col["name"]
                col_type = str(col["type"]).lower()

                # Categorize columns
                if "id" in col_name.lower() or "key" in col_name.lower():
                    key_columns.append(col_name)
                elif "timestamp" in col_type or "date" in col_type or "time" in col_name.lower():
                    time_columns.append(col_name)
                elif "varchar" in col_type or "text" in col_type or "char" in col_type:
                    text_columns.append(col_name)
                elif "int" in col_type or "numeric" in col_type or "decimal" in col_type or "float" in col_type:
                    numeric_columns.append(col_name)
                else:
                    other_columns.append(col_name)

            # Format as compact single line
            lines.append(f"- {table_name}:")
            if key_columns:
                lines.append(f"  Keys: {', '.join(key_columns)}")
            if time_columns:
                lines.append(f"  Time: {', '.join(time_columns)}")
            if text_columns:
                lines.append(f"  Text: {', '.join(text_columns)}")
            if numeric_columns:
                lines.append(f"  Numeric: {', '.join(numeric_columns)}")
            if other_columns:
                lines.append(f"  Other: {', '.join(other_columns)}")
            lines.append("")  # Blank line between tables

        return "\n".join(lines)

    def get_schema_summary(self) -> dict:
        """Get schema summary with statistics"""
        summary = {
            "table_count": len(self.schema["tables"]),
            "tables": []
        }

        for table_name, table_info in self.schema["tables"].items():
            summary["tables"].append({
                "name": table_name,
                "column_count": len(table_info["columns"]),
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"]
                    }
                    for col in table_info["columns"]
                ]
            })

        return summary

    def execute_query(self, sql: str, row_limit: int = 100) -> Dict[str, Any]:
        """
        Execute SQL query with safety validation

        Returns:
            Dict with success, data, columns, row_count, or error
        """
        # Validate SQL
        is_valid, validated_sql = self._validate_sql(sql, row_limit)
        if not is_valid:
            return {
                "success": False,
                "error": f"SQL validation failed: {validated_sql}",
            }

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(validated_sql))
                rows = result.fetchall()
                columns = list(result.keys())
                data = [dict(row._mapping) for row in rows]

                logger.info(f"Query executed: {len(data)} rows returned")

                return {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "sql": validated_sql,
                }

        except Exception as e:
            err_msg = str(e)

            # Helpful hints for common errors
            if "relation" in err_msg.lower() and "does not exist" in err_msg.lower():
                logger.error("Query failed - table not found. Check schema prefix.")
                if self.active_schema:
                    hint = f"Make sure to use {self.active_schema} schema prefix (e.g., {self.active_schema}.table_name)."
                else:
                    hint = "Check that the table name is correct and exists in the database."

                return {
                    "success": False,
                    "error": f"{err_msg}. {hint}",
                }

            logger.error(f"Query execution failed: {err_msg}")
            return {
                "success": False,
                "error": err_msg,
            }

    def _validate_sql(self, sql_query: str, row_limit: int) -> tuple:
        """Validate SQL for safety (read-only)"""
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Invalid SQL syntax"

            # Block dangerous operations
            dangerous_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "ALTER",
                "INSERT",
                "CREATE",
                "TRUNCATE",
            ]
            sql_upper = sql_query.upper()

            for keyword in dangerous_keywords:
                if re.search(rf"\b{keyword}\b", sql_upper):
                    return False, f"Dangerous operation not allowed: {keyword}"

            if not sql_upper.strip().startswith("SELECT"):
                return False, "Only SELECT statements are allowed"

            # Enforce LIMIT
            if "LIMIT" not in sql_upper:
                sql_query = sql_query.rstrip().rstrip(";") + f" LIMIT {row_limit};"
            else:
                limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
                if limit_match:
                    existing_limit = int(limit_match.group(1))
                    if existing_limit > row_limit:
                        sql_query = re.sub(
                            r"LIMIT\s+\d+",
                            f"LIMIT {row_limit}",
                            sql_query,
                            flags=re.IGNORECASE,
                        )

            if not sql_query.rstrip().endswith(";"):
                sql_query = sql_query.rstrip() + ";"

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"


# ChartRecommender removed - LLM now reasons autonomously about chart types
# No hardcoded logic - the LLM analyzes data and makes intelligent decisions


# Tool creation functions for LlamaIndex Agent


def create_sql_tool(db_manager: DatabaseManager) -> FunctionTool:
    """Create SQL execution tool"""

    def execute_sql(sql_query: str) -> str:
        """
        Execute a SQL query on the database.

        Args:
            sql_query: The SQL SELECT query to execute

        Returns:
            JSON string with query results or error
        """
        global _last_sql_result

        result = db_manager.execute_query(sql_query)

        # Store result globally for extraction
        _last_sql_result = result.copy() if result.get("success") else None

        return json.dumps(result, default=str)

    # Build dynamic description based on detected schema
    schema_name = db_manager.active_schema

    if schema_name:
        schema_instruction = f"""CRITICAL: ALL tables are in the {schema_name} schema. You MUST use schema prefix:
- ✅ CORRECT: SELECT * FROM {schema_name}.table_name
- ❌ WRONG: SELECT * FROM table_name

ALWAYS use {schema_name}.table_name (schema prefix required!)"""

        # Get first table as example (if available)
        example_table = list(db_manager.schema['tables'].keys())[0] if db_manager.schema['tables'] else f"{schema_name}.example_table"
        example_sql = f"""Example SQL:
SELECT * FROM {example_table} LIMIT 10;"""
    else:
        schema_instruction = "Note: This database does not use schema prefixes. Use table names directly."
        example_table = list(db_manager.schema['tables'].keys())[0] if db_manager.schema['tables'] else "example_table"
        example_sql = f"""Example SQL:
SELECT * FROM {example_table} LIMIT 10;"""

    description = f"""Execute a SQL query on the connected database.

Use this tool when the user asks for data analysis, statistics, or filtering.

{schema_instruction}

Important:
- Prefer aggregation (COUNT, AVG, SUM, GROUP BY) over raw rows
- Order results meaningfully (DESC for most, ASC for least)
- Use LIMIT 10-50 for top/bottom queries
- For date/time extraction: Use EXTRACT(MONTH FROM column_name), EXTRACT(YEAR FROM column_name)
- For text searches: Use ILIKE for case-insensitive matching

{example_sql}"""

    return FunctionTool.from_defaults(
        fn=execute_sql,
        name="execute_sql",
        description=description,
    )


def create_schema_tool(db_manager: DatabaseManager) -> FunctionTool:
    """Create schema information tool"""

    def get_schema() -> str:
        """
        Get database schema information.

        Returns:
            Schema details including tables and columns
        """
        return db_manager.get_schema_text()

    return FunctionTool.from_defaults(
        fn=get_schema,
        name="get_schema",
        description="Get database schema information including available tables and columns.",
    )


# Chart tool REMOVED per OptimaX Visualization Workflow specification
# LLM must ONLY use [CHART_SUGGESTION: ...] tags in response text
# System enforces deterministic rendering after user selection
# NO LLM involvement in chart configuration or rendering


def get_last_sql_result():
    """Get the last SQL execution result"""
    global _last_sql_result
    return _last_sql_result


def clear_last_sql_result():
    """Clear the last SQL result"""
    global _last_sql_result
    _last_sql_result = None


def get_last_chart_config():
    """Get the last chart configuration"""
    global _last_chart_config
    return _last_chart_config


def clear_last_chart_config():
    """Clear the last chart configuration"""
    global _last_chart_config
    _last_chart_config = None


def classify_visualization_intent(
    llm,
    data_summary: str,
    column_types: Dict[str, str],
    row_count: int
) -> Dict[str, Any]:
    """
    One-shot LLM call to classify visualization intent and suggest chart types.

    No agent, no tools, no retries - just pure classification.

    Args:
        llm: Groq LLM instance
        data_summary: Brief summary of the data (e.g., "accident counts by state")
        column_types: Dict of column names to types (e.g., {"state": "string", "count": "number"})
        row_count: Number of rows in the dataset

    Returns:
        Dict with analysis_type and recommended_charts
    """
    prompt = f"""You are a data visualization expert. Classify the following dataset and suggest appropriate chart types.

DATASET SUMMARY:
- Description: {data_summary}
- Columns: {json.dumps(column_types)}
- Row count: {row_count}

CLASSIFICATION TASK:
Determine the analysis type and suggest 2-3 suitable chart types.

ANALYSIS TYPES:
- comparison: Comparing categories (states, cities, weather types)
- time_series: Trends over time (monthly, yearly patterns)
- proportion: Part-to-whole relationships (≤8 categories)
- correlation: Relationship between two numeric variables
- distribution: Statistical spread of values

CHART OPTIONS:
- bar, horizontal_bar, line, pie, doughnut, area, scatter, table

RESPONSE FORMAT (JSON only):
{{
    "analysis_type": "comparison|time_series|proportion|correlation|distribution",
    "recommended_charts": [
        {{"type": "bar", "label": "Bar Chart", "recommended": true}},
        {{"type": "horizontal_bar", "label": "Horizontal Bar", "recommended": false}},
        {{"type": "table", "label": "Data Table", "recommended": false}}
    ]
}}

Respond with ONLY the JSON object, nothing else."""

    try:
        # One-shot LLM call - no agent, no tools
        response = llm.complete(prompt)
        response_text = str(response).strip()

        # Extract JSON from response
        # Handle cases where LLM adds markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        logger.info(f"✓ Visualization classified: {result.get('analysis_type')}")
        return result

    except Exception as e:
        logger.error(f"Visualization classification failed: {str(e)}")
        # Fallback to safe defaults
        return {
            "analysis_type": "comparison",
            "recommended_charts": [
                {"type": "bar", "label": "Bar Chart", "recommended": True},
                {"type": "table", "label": "Data Table", "recommended": False}
            ]
        }


def initialize_tools(database_url: str) -> tuple:
    """
    Initialize all tools

    Returns:
        Tuple of (tools_list, db_manager)
    """
    db_manager = DatabaseManager(database_url)

    tools = [
        create_sql_tool(db_manager),
        create_schema_tool(db_manager),
        # NO CHART TOOL - Per OptimaX Visualization Workflow spec:
        # LLM uses [CHART_SUGGESTION: ...] tags only
        # System handles rendering deterministically after user selection
    ]

    logger.info(f"Initialized {len(tools)} tools")
    return tools, db_manager
