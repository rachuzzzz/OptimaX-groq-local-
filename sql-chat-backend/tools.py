"""
OptimaX Tools - SQL and Chart Tools for LlamaIndex Agent
=========================================================

Simple tools for:
1. SQL Query Execution (direct SQL, no LLM generation)
2. Chart Type Detection/Recommendation

Author: OptimaX Team
Version: 4.0 (Simplified Single-LLM)
"""

import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text, inspect
import sqlparse
import re
import json
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)

# Global variable to store last tool execution result
_last_sql_result = None


class DatabaseManager:
    """Manages database connection and schema information"""

    def __init__(self, database_url: str):
        """Initialize database connection"""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.schema = self._load_schema()
        self.cached_schema = None  # Free tier optimization: cache schema text
        logger.info("Database connected successfully")

    def _load_schema(self) -> Dict[str, Any]:
        """Load database schema information (lazy loading for faster startup)"""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            schema_info = {"tables": {}}

            # Only load schema for us_accidents table (skip others for speed)
            for table in tables:
                if table == "us_accidents":  # Target table optimization
                    columns = inspector.get_columns(table)
                    schema_info["tables"][table] = {
                        "columns": [
                            {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col["nullable"]
                            }
                            for col in columns
                        ]
                    }

            logger.info(f"Schema loaded: {len(schema_info['tables'])} tables")
            return schema_info

        except Exception as e:
            logger.error(f"Failed to load schema: {str(e)}")
            return {"tables": {}}

    def get_schema_text(self) -> str:
        """Get human-readable schema description (cached for free tier)"""
        # Free tier optimization: return cached schema if available
        if self.cached_schema:
            return self.cached_schema

        lines = ["Database Schema:\n"]

        for table_name, table_info in self.schema["tables"].items():
            lines.append(f"\nTable: {table_name}")
            lines.append("Columns:")
            for col in table_info["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                lines.append(f"  - {col['name']}: {col['type']} ({nullable})")

        self.cached_schema = "\n".join(lines)
        return self.cached_schema

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
                "error": f"SQL validation failed: {validated_sql}"
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
                    "sql": validated_sql
                }

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _validate_sql(self, sql_query: str, row_limit: int) -> tuple:
        """Validate SQL for safety (read-only)"""
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Invalid SQL syntax"

            # Block dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'INSERT', 'CREATE', 'TRUNCATE']
            sql_upper = sql_query.upper()

            for keyword in dangerous_keywords:
                if re.search(rf'\b{keyword}\b', sql_upper):
                    return False, f"Dangerous operation not allowed: {keyword}"

            if not sql_upper.strip().startswith('SELECT'):
                return False, "Only SELECT statements are allowed"

            # Enforce LIMIT
            if 'LIMIT' not in sql_upper:
                sql_query = sql_query.rstrip().rstrip(';') + f' LIMIT {row_limit};'
            else:
                limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
                if limit_match:
                    existing_limit = int(limit_match.group(1))
                    if existing_limit > row_limit:
                        sql_query = re.sub(
                            r'LIMIT\s+\d+',
                            f'LIMIT {row_limit}',
                            sql_query,
                            flags=re.IGNORECASE
                        )

            if not sql_query.rstrip().endswith(';'):
                sql_query = sql_query.rstrip() + ';'

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"


class ChartRecommender:
    """Recommends appropriate chart types based on data structure"""

    @staticmethod
    def recommend_chart(data: List[Dict], columns: List[str]) -> Dict[str, Any]:
        """
        Analyze data and recommend appropriate chart type

        Returns:
            Dict with chart_type, reasoning, and config
        """
        if not data or not columns:
            return {
                "chart_type": "none",
                "reasoning": "No data available for visualization"
            }

        num_rows = len(data)
        num_cols = len(columns)

        # Analyze column types
        first_row = data[0]
        numeric_cols = []
        text_cols = []

        for col in columns:
            value = first_row.get(col)
            if isinstance(value, (int, float)):
                numeric_cols.append(col)
            else:
                text_cols.append(col)

        # Chart recommendation logic
        if num_cols == 2 and len(text_cols) == 1 and len(numeric_cols) == 1:
            # One category, one value -> Bar chart
            return {
                "chart_type": "bar",
                "reasoning": "One categorical column and one numeric column - perfect for bar chart",
                "config": {
                    "x_axis": text_cols[0],
                    "y_axis": numeric_cols[0],
                    "title": f"{numeric_cols[0]} by {text_cols[0]}"
                }
            }

        elif len(text_cols) == 1 and len(numeric_cols) >= 1:
            if num_rows <= 10:
                # Small categorical breakdown -> Pie chart
                return {
                    "chart_type": "pie",
                    "reasoning": "Small number of categories - good for pie chart",
                    "config": {
                        "labels": text_cols[0],
                        "values": numeric_cols[0],
                        "title": f"Distribution of {numeric_cols[0]}"
                    }
                }
            else:
                # Many categories -> Bar chart
                return {
                    "chart_type": "bar",
                    "reasoning": "Multiple categories - bar chart for comparison",
                    "config": {
                        "x_axis": text_cols[0],
                        "y_axis": numeric_cols[0],
                        "title": f"{numeric_cols[0]} by {text_cols[0]}"
                    }
                }

        elif len(numeric_cols) >= 2:
            # Multiple numeric columns
            time_keywords = ['year', 'month', 'day', 'date', 'time', 'hour']
            has_time = any(keyword in col.lower() for col in text_cols for keyword in time_keywords)

            if has_time:
                # Time series -> Line chart
                return {
                    "chart_type": "line",
                    "reasoning": "Time-based data detected - line chart for trends",
                    "config": {
                        "x_axis": text_cols[0] if text_cols else columns[0],
                        "y_axis": numeric_cols[0],
                        "title": f"{numeric_cols[0]} over time"
                    }
                }
            else:
                # Multiple metrics -> Grouped bar
                return {
                    "chart_type": "bar",
                    "reasoning": "Multiple metrics - grouped bar chart for comparison",
                    "config": {
                        "x_axis": text_cols[0] if text_cols else columns[0],
                        "y_axes": numeric_cols,
                        "title": "Multi-metric comparison"
                    }
                }

        else:
            # Default to table
            return {
                "chart_type": "table",
                "reasoning": "Complex data structure - table view recommended",
                "config": {
                    "columns": columns
                }
            }


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
        _last_sql_result = result.copy() if result.get('success') else None

        return json.dumps(result, default=str)

    return FunctionTool.from_defaults(
        fn=execute_sql,
        name="execute_sql",
        description="""Execute a SQL query on the us_accidents table (7.7M traffic accident records).

Use this tool when the user asks for data analysis, statistics, or filtering.

Important:
- Table name: us_accidents
- Always use aggregation (COUNT, AVG, SUM, GROUP BY)
- Order results meaningfully (DESC for most, ASC for least)
- Use LIMIT 10-50 for top/bottom queries

Available columns:
- Geographic: state, city, county, latitude, longitude
- Severity: severity (1-4 scale)
- Weather: weather_condition, temperature_f, visibility_mi, precipitation_in
- Time: start_time, end_time, year, month, day, hour, day_of_week
- Road: street, junction, traffic_signal, crossing, railway

Example SQL:
SELECT state, COUNT(*) as count FROM us_accidents GROUP BY state ORDER BY count DESC LIMIT 10"""
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
        description="Get database schema information including available tables and columns."
    )


def create_chart_tool() -> FunctionTool:
    """Create chart recommendation tool"""

    def recommend_chart(data_json: str) -> str:
        """
        Recommend appropriate chart type for given data.

        Args:
            data_json: JSON string containing data and columns

        Returns:
            JSON string with chart recommendation
        """
        try:
            data_dict = json.loads(data_json)
            data = data_dict.get("data", [])
            columns = data_dict.get("columns", [])

            recommendation = ChartRecommender.recommend_chart(data, columns)
            return json.dumps(recommendation)

        except Exception as e:
            return json.dumps({
                "chart_type": "none",
                "error": str(e)
            })

    return FunctionTool.from_defaults(
        fn=recommend_chart,
        name="recommend_chart",
        description="""Recommend the best chart type for visualizing query results.

Use this when the user asks to visualize, chart, or graph data.
Input should be the data from a SQL query result."""
    )


def get_last_sql_result():
    """Get the last SQL execution result"""
    global _last_sql_result
    return _last_sql_result


def clear_last_sql_result():
    """Clear the last SQL result"""
    global _last_sql_result
    _last_sql_result = None


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
        create_chart_tool()
    ]

    logger.info(f"Initialized {len(tools)} tools")
    return tools, db_manager
