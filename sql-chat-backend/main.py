from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sqlalchemy import create_engine, text, MetaData, inspect
import pandas as pd
import sqlparse
import re
from sqlparse import sql, tokens
import torch

load_dotenv()

app = FastAPI(title="OptimaX SQL Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4201"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# Initialize router model globally
router_pipeline = None

def initialize_router_model():
    """Initialize Phi-3 via Ollama for query routing"""
    global router_pipeline
    try:
        # Use local Ollama Phi-3
        from llama_index.llms.ollama import Ollama

        router_pipeline = Ollama(
            model="phi3:mini",  # Use your specific phi3:mini model
            base_url="http://localhost:11434",
            temperature=0.1,
            request_timeout=30.0
        )
        logger.info("Router model (Ollama Phi-3) connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama Phi-3: {str(e)}")
        # Don't fail - use rule-based routing
        router_pipeline = None
        logger.warning("Ollama Phi-3 not available - will use rule-based routing")

def route_query_with_llm(user_message: str) -> str:
    """Route query using Ollama Phi-3 to classify as SQL_INTENT or CHAT_INTENT"""
    if router_pipeline is None:
        raise RuntimeError("Router model not loaded. Cannot process requests without LLM.")

    system_prompt = """You are an intent classifier.
The user may ask a database-related question or a general chat question.
Respond ONLY with "SQL_INTENT" or "CHAT_INTENT"."""

    input_text = f"{system_prompt}\n\nUser question: {user_message}\n\nIntent:"

    try:
        response = router_pipeline.complete(input_text)
        generated_text = response.text.strip().upper()

        if "SQL_INTENT" in generated_text:
            return "sql"
        elif "CHAT_INTENT" in generated_text:
            return "chat"
        else:
            # Default to SQL if unclear
            return "sql"
    except Exception as e:
        logger.error(f"Error calling Ollama Phi-3: {str(e)}")
        # Fallback to rule-based routing
        return route_query_with_rules(user_message)

def route_query_with_rules(user_message: str) -> str:
    """Fallback rule-based routing"""
    message_lower = user_message.lower()

    # SQL-related keywords
    sql_keywords = ['show', 'count', 'select', 'where', 'group', 'order', 'data', 'table',
                   'accidents', 'severity', 'state', 'city', 'weather', 'temperature',
                   'how many', 'what are', 'find', 'list', 'get', 'which', 'top']

    # Chat-related keywords
    chat_keywords = ['hello', 'hi', 'thanks', 'thank you', 'help', 'how are you',
                    'what is your name', 'who are you', 'goodbye', 'bye']

    # Check for chat keywords first
    for keyword in chat_keywords:
        if keyword in message_lower:
            return "chat"

    # Check for SQL keywords
    for keyword in sql_keywords:
        if keyword in message_lower:
            return "sql"

    # Default to SQL for data-related queries
    return "sql"

def generate_chat_response(user_message: str) -> str:
    """Generate chat response for general questions"""
    if router_pipeline is None:
        # Provide a basic fallback response
        return "Hello! I'm OptimaX, your SQL assistant. I can help you analyze US traffic accident data. What would you like to know?"

    chat_prompt = f"""You are OptimaX, a helpful AI assistant specialized in analyzing US traffic accident data.
You are friendly and professional. Keep responses concise and helpful.
If asked about your capabilities, mention you can analyze US traffic accident data through SQL queries.

User: {user_message}
Assistant:"""

    try:
        response = router_pipeline.complete(chat_prompt)
        generated_text = response.text.strip()

        # Extract the response after "Assistant:" if present
        if "Assistant:" in generated_text:
            chat_response = generated_text.split("Assistant:")[-1].strip()
        else:
            chat_response = generated_text.strip()

        # Ensure we have a valid response
        if not chat_response or len(chat_response) < 5:
            return "Hello! I'm OptimaX, your SQL assistant. I can help you analyze US traffic accident data. What would you like to know?"

        return chat_response
    except Exception as e:
        logger.error(f"Error generating chat response with Ollama: {str(e)}")
        return "Hello! I'm OptimaX, your SQL assistant. I can help you analyze US traffic accident data. What would you like to know?"

class ChatMessage(BaseModel):
    message: str
    system_prompt: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    error: Optional[str] = None

class TextToSQLEngine:
    def __init__(self):
        self.engine = None
        self.text_to_sql_model = None
        self.tokenizer = None
        self.table_schema = None
        self.initialize()

    def initialize(self):
        try:
            # Initialize router model first
            initialize_router_model()

            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL not set in environment")

            self.engine = create_engine(database_url)

            # Use Hugging Face Inference API for SQL generation (no local model needed)
            logger.info("Using Hugging Face Inference API for SQL generation")
            self.tokenizer = None
            self.text_to_sql_model = None
            self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

            if not self.hf_api_key:
                logger.warning("HUGGINGFACE_API_KEY not set - SQL generation will be limited")
            else:
                logger.info("Hugging Face API key configured for remote SQL generation")

            # Get table schema information
            self.load_table_schema()

            logger.info("Text-to-SQL engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Text-to-SQL engine: {str(e)}")
            raise

    def load_table_schema(self):
        """Load detailed table schema with column types"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'us_accidents'
                    ORDER BY ordinal_position
                """))

                schema_info = []
                for row in result:
                    col_name, data_type, nullable = row
                    schema_info.append(f"{col_name} ({data_type})")

                self.table_schema = {
                    "table_name": "us_accidents",
                    "columns": schema_info,
                    "schema_text": f"CREATE TABLE us_accidents (\n  {', '.join(schema_info)}\n);"
                }

                logger.info(f"Loaded schema for table: us_accidents with {len(schema_info)} columns")

        except Exception as e:
            logger.error(f"Failed to load table schema: {str(e)}")
            raise

    def validate_sql(self, sql_query: str) -> tuple[bool, str]:
        """Validate SQL for safety and syntax"""
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Invalid SQL syntax"

            # Check for dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'INSERT', 'CREATE', 'TRUNCATE']
            sql_upper = sql_query.upper()

            for keyword in dangerous_keywords:
                if re.search(rf'\b{keyword}\b', sql_upper):
                    return False, f"Dangerous operation detected: {keyword}"

            # Ensure it's a SELECT statement
            if not sql_upper.strip().startswith('SELECT'):
                return False, "Only SELECT statements are allowed"

            # Add LIMIT if not present and query might return large results
            if 'LIMIT' not in sql_upper and ('GROUP BY' in sql_upper or 'ORDER BY' in sql_upper):
                if not sql_query.rstrip().endswith(';'):
                    sql_query = sql_query.rstrip() + ' LIMIT 1000;'
                else:
                    sql_query = sql_query.rstrip(';') + ' LIMIT 1000;'

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

    def generate_sql(self, question: str, system_prompt: str = None) -> str:
        """Generate SQL using local Ollama with specialized prompting"""
        try:
            # Use the same Ollama instance for SQL generation with specialized prompting
            from llama_index.llms.ollama import Ollama

            sql_llm = Ollama(
                model="phi3:mini",
                base_url="http://localhost:11434",
                temperature=0.1,
                request_timeout=30.0
            )

            # Prepare specialized SQL prompt
            schema_text = self.table_schema.get("schema_text", "") if self.table_schema else ""

            if system_prompt:
                prompt = f"{system_prompt}\n\nQuestion: {question}"
            else:
                prompt = f"""You are a PostgreSQL expert. Generate ONLY the SQL query, no explanations.

Database Schema:
{schema_text}

Important Data Types & Values:
- severity: integer (1=low, 2=medium, 3=high, 4=severe)
- weather_condition: text values like 'Snow', 'Rain', 'Clear', etc.
- state: 2-letter codes like 'CA', 'TX', 'FL'
- Boolean columns: true/false values

Rules:
- Use exact column names from schema
- severity is INTEGER: use numbers (1,2,3,4) not text
- For "severe": use severity >= 3 or severity = 4
- Always use PostgreSQL syntax
- Add LIMIT when appropriate for large results
- Use proper aggregation for counts

Question: {question}

PostgreSQL Query:"""

            logger.info(f"Generating SQL with Ollama: {question}")

            # Generate SQL using Ollama
            response = sql_llm.complete(prompt)
            generated_sql = response.text.strip()

            # Clean the generated SQL
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1].strip()

            # Remove common prefixes
            prefixes_to_remove = ["Query:", "SQL:", "PostgreSQL Query:", "Answer:"]
            for prefix in prefixes_to_remove:
                if generated_sql.startswith(prefix):
                    generated_sql = generated_sql[len(prefix):].strip()

            # Ensure it ends with semicolon
            if not generated_sql.endswith(';'):
                generated_sql += ';'

            logger.info(f"Ollama generated SQL: {generated_sql}")
            return generated_sql

        except Exception as e:
            logger.error(f"Error generating SQL with Ollama: {str(e)}")
            raise Exception(f"SQL generation failed: {str(e)}")


query_engine_instance = TextToSQLEngine()

@app.get("/")
async def root():
    return {"message": "OptimaX SQL Chat API is running"}

@app.get("/health")
async def health_check():
    try:
        with query_engine_instance.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

def format_sql_results(rows, columns):
    """Format SQL results for display"""
    if rows:
        # Format as structured data for frontend
        formatted_data = []
        for i, row in enumerate(rows):
            if len(row) == 2 and isinstance(row[1], (int, float)):
                # Key-value pair with numeric value
                formatted_data.append(f"{i+1}. {row[0]} - {row[1]:,}")
            elif len(row) == 1:
                # Single column result
                formatted_data.append(f"{i+1}. {row[0]}")
            else:
                # Multi-column result
                row_data = []
                for j, col in enumerate(row):
                    if isinstance(col, (int, float)) and col > 1000:
                        row_data.append(f"{col:,}")
                    else:
                        row_data.append(str(col))
                formatted_data.append(f"{i+1}. {' | '.join(row_data)}")

        # Limit display to prevent overwhelming frontend
        if len(formatted_data) > 50:
            return "\n".join(formatted_data[:50]) + f"\n\n... and {len(formatted_data) - 50} more results"
        else:
            return "\n".join(formatted_data)
    else:
        return "No results found."

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Route the query using LLM
        try:
            route = route_query_with_llm(message.message)
            logger.info(f"Query routed to: {route}")
        except RuntimeError as router_error:
            logger.error(f"Router model not available: {str(router_error)}")
            return ChatResponse(
                response="Service temporarily unavailable: Router model failed to load. Please contact support.",
                sql_query=None,
                error=str(router_error)
            )

        if route == "sql":
            # Generate SQL using Text-to-SQL engine
            sql_query = query_engine_instance.generate_sql(
                question=message.message,
                system_prompt=message.system_prompt
            )

            logger.info(f"Generated SQL: {sql_query}")

            # Validate SQL for safety and performance
            is_valid, validated_sql = query_engine_instance.validate_sql(sql_query)
            if not is_valid:
                return ChatResponse(
                    response=f"Query validation failed: {validated_sql}",
                    sql_query=None,
                    error=validated_sql
                )

            logger.info(f"Validated SQL: {validated_sql}")

            # Execute SQL with read-only approach
            with query_engine_instance.engine.connect() as conn:
                result = conn.execute(text(validated_sql))
                rows = result.fetchall()
                columns = result.keys()

            # Format results
            formatted_result = format_sql_results(rows, columns)

            return ChatResponse(
                response=formatted_result,
                sql_query=validated_sql if os.getenv("DEBUG") == "true" else None
            )

        else:  # general chat
            try:
                chat_response = generate_chat_response(message.message)
                return ChatResponse(response=chat_response)
            except RuntimeError as chat_error:
                logger.error(f"Chat model failed: {str(chat_error)}")
                return ChatResponse(
                    response="Service temporarily unavailable: Chat model failed to generate response. Please contact support.",
                    sql_query=None,
                    error=str(chat_error)
                )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return ChatResponse(
            response="Service temporarily unavailable: An unexpected error occurred. Please contact support.",
            sql_query=None,
            error=str(e)
        )

@app.get("/table-info")
async def get_table_info():
    try:
        with query_engine_instance.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'us_accidents'
                ORDER BY ordinal_position
            """))
            columns = [{"name": row[0], "type": row[1], "nullable": row[2]} for row in result]

            count_result = conn.execute(text("SELECT COUNT(*) FROM us_accidents"))
            total_records = count_result.scalar()

            return {
                "table": "us_accidents",
                "columns": columns,
                "total_records": total_records
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)