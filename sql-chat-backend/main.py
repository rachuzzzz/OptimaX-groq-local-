from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from sqlalchemy import create_engine, text
import sqlparse
import re
import requests
from datetime import datetime

load_dotenv()

app = FastAPI(title="OptimaX SQL Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# Initialize router model globally
router_pipeline = None

AVAILABLE_SQL_MODELS = {
    "codellama:7b-instruct-q4_K_M": "CodeLlama 7B Instruct (Quantized) - Installed model"
}

DEFAULT_SQL_MODEL = "codellama:7b-instruct-q4_K_M"
INTENT_MODEL = "phi3:mini"

# Default system prompts
DEFAULT_INTENT_PROMPT = """You are an intent classifier.
The user may ask a database-related question or a general chat question.
Respond ONLY with "SQL_INTENT" or "CHAT_INTENT"."""

DEFAULT_SQL_PROMPT = """You are a PostgreSQL expert. Generate ONLY the SQL query, no explanations.

Database Schema:
{schema_text}

Important Data Types & Values:
- severity: integer (1=low, 2=medium, 3=high, 4=severe)
- weather_condition: text values like 'Snow', 'Rain', 'Clear', etc.
- state: 2-letter codes like 'CA', 'TX', 'FL'
- Boolean columns: true/false values (amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, traffic_calming, traffic_signal, turning_loop)

Rules:
- Use exact column names from schema
- severity is INTEGER: use numbers (1,2,3,4) not text
- For "severe": use severity >= 3 or severity = 4
- Always use PostgreSQL syntax
- Add LIMIT when appropriate for large results
- Use proper aggregation for counts
- For location queries: use traffic_signal = true (not separate tables)
- NO PostGIS functions - use simple WHERE conditions
- NO geography/geometry casting or spatial functions

Examples:
- "accidents near traffic signals" -> WHERE traffic_signal = true
- "accidents at intersections" -> WHERE junction = true
- "accidents at railway crossings" -> WHERE railway = true

Question: {question}

PostgreSQL Query:"""

# Current system prompts (can be modified by users)
current_intent_prompt = DEFAULT_INTENT_PROMPT
current_sql_prompt = DEFAULT_SQL_PROMPT

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
        router_pipeline = None
        logger.error("Ollama Phi-3 not available - system requires LLM models to function")

def route_query_with_llm(user_message: str) -> str:
    """Route query using Ollama Phi-3 to classify as SQL_INTENT or CHAT_INTENT"""
    if router_pipeline is None:
        raise RuntimeError("Router model not loaded. Cannot process requests without LLM.")

    input_text = f"{current_intent_prompt}\n\nUser question: {user_message}\n\nIntent:"

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
        raise RuntimeError(f"Intent classification failed: {str(e)}")


def generate_chat_response(user_message: str) -> str:
    """Generate chat response for general questions"""
    if router_pipeline is None:
        raise RuntimeError("Chat model not loaded. Cannot generate responses without LLM.")

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
            raise RuntimeError("Generated chat response is too short or empty")

        return chat_response
    except Exception as e:
        logger.error(f"Error generating chat response with Ollama: {str(e)}")
        raise RuntimeError(f"Chat response generation failed: {str(e)}")

class ChatMessage(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    sql_model: Optional[str] = DEFAULT_SQL_MODEL
    include_sql: Optional[bool] = None

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None

class SystemPromptUpdate(BaseModel):
    model_type: str  # "intent" or "sql"
    prompt: str

class SystemPromptsResponse(BaseModel):
    intent_prompt: str
    sql_prompt: str
    default_intent_prompt: str
    default_sql_prompt: str

class TextToSQLEngine:
    def __init__(self):
        self.engine = None
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

            logger.info("Using local Ollama for SQL generation")

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
            if 'LIMIT' not in sql_upper:
                # Add LIMIT for safety on queries that might return many rows
                if any(keyword in sql_upper for keyword in ['SELECT *', 'COUNT(*)', 'GROUP BY', 'ORDER BY']):
                    if not sql_query.rstrip().endswith(';'):
                        sql_query = sql_query.rstrip() + ' LIMIT 1000;'
                    else:
                        sql_query = sql_query.rstrip(';') + ' LIMIT 1000;'

            # Ensure query ends with semicolon
            if not sql_query.rstrip().endswith(';'):
                sql_query = sql_query.rstrip() + ';'

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

    def generate_sql(self, question: str, model_name: str = DEFAULT_SQL_MODEL, system_prompt: str = None) -> str:
        """Generate SQL using local Ollama models"""
        try:
            # Prepare specialized SQL prompt with schema grounding
            schema_text = self.table_schema.get("schema_text", "") if self.table_schema else ""

            if system_prompt:
                prompt = f"{system_prompt}\n\nQuestion: {question}"
            else:
                # Use current SQL prompt with schema and question formatting
                prompt = current_sql_prompt.format(schema_text=schema_text, question=question)

            logger.info(f"Generating SQL with local Ollama model {model_name}: {question}")

            # Call local Ollama API
            ollama_url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200
                }
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(ollama_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()

            result = response.json()
            generated_sql = result.get("response", "").strip()

            if not generated_sql:
                raise Exception("Empty response from Ollama")

            # Clean the generated SQL
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1].strip()

            # Remove common prefixes
            prefixes_to_remove = ["Query:", "SQL:", "PostgreSQL Query:", "Answer:", "Response:"]
            for prefix in prefixes_to_remove:
                if generated_sql.startswith(prefix):
                    generated_sql = generated_sql[len(prefix):].strip()

            # Fix common LLM typos in SQL
            typo_fixes = {
                "number_of0f_": "number_of_",
                "accidents0f": "accidents_of",
                "count0f": "count_of",
                "sum0f": "sum_of",
                "avg0f": "avg_of",
                "trafficsignal": "traffic_signal",
                "trafficcalming": "traffic_calming",
                "FORM": "FROM",
                "form": "FROM"
            }
            for typo, fix in typo_fixes.items():
                generated_sql = re.sub(r'\b' + typo + r'\b', fix, generated_sql, flags=re.IGNORECASE)

            # Ensure it ends with semicolon
            if not generated_sql.rstrip().endswith(';'):
                generated_sql = generated_sql.rstrip() + ';'

            logger.info(f"Local Ollama {model_name} generated SQL: {generated_sql}")
            return generated_sql

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling local Ollama API: {str(e)}")
            raise Exception(f"SQL generation failed - Ollama API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating SQL with local Ollama: {str(e)}")
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
    """Format SQL results for display with improved formatting"""
    if not rows:
        return "No results found."

    # Format header
    header = " | ".join(columns) if len(columns) > 1 else columns[0]
    separator = "-" * len(header)

    formatted_data = [header, separator]

    for i, row in enumerate(rows):
        if len(row) == 2 and isinstance(row[1], (int, float)):
            # Key-value pair with numeric value
            formatted_data.append(f"{row[0]} | {row[1]:,}")
        elif len(row) == 1:
            # Single column result
            value = row[0]
            if isinstance(value, (int, float)) and value > 1000:
                formatted_data.append(f"{value:,}")
            else:
                formatted_data.append(str(value))
        else:
            # Multi-column result
            row_data = []
            for col in row:
                if isinstance(col, (int, float)) and col > 1000:
                    row_data.append(f"{col:,}")
                elif col is None:
                    row_data.append("NULL")
                else:
                    row_data.append(str(col))
            formatted_data.append(" | ".join(row_data))

    # Limit display to prevent overwhelming frontend
    if len(formatted_data) > 52:  # 50 data rows + header + separator
        result_rows = formatted_data[:52]
        result_rows.append(f"\n... and {len(rows) - 50} more rows")
        return "\n".join(result_rows)
    else:
        return "\n".join(formatted_data)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    start_time = datetime.now()

    try:
        # Validate sql_model if provided
        if message.sql_model and message.sql_model not in AVAILABLE_SQL_MODELS:
            return ChatResponse(
                response=f"Invalid model '{message.sql_model}'. Available models: {list(AVAILABLE_SQL_MODELS.keys())}",
                error=f"Model '{message.sql_model}' not available",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        # Route the query using LLM (Phi-3 Mini only for intent classification)
        try:
            route = route_query_with_llm(message.message)
            logger.info(f"Query routed to: {route}")
        except RuntimeError as router_error:
            logger.error(f"Router model not available: {str(router_error)}")
            return ChatResponse(
                response="Service unavailable: LLM models are required for operation. Please ensure Ollama is running with phi3:mini model.",
                sql_query=None,
                error=str(router_error),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        if route == "sql":
            # Generate SQL using local Ollama models
            sql_model = message.sql_model or DEFAULT_SQL_MODEL
            include_sql = message.include_sql if message.include_sql is not None else (os.getenv("DEBUG") == "true")

            try:
                sql_query = query_engine_instance.generate_sql(
                    question=message.message,
                    model_name=sql_model,
                    system_prompt=message.system_prompt
                )

                logger.info(f"Generated SQL: {sql_query}")

                # Validate SQL for safety and performance
                is_valid, validated_sql = query_engine_instance.validate_sql(sql_query)
                if not is_valid:
                    return ChatResponse(
                        response=f"Query validation failed: {validated_sql}",
                        sql_query=None,
                        error=validated_sql,
                        model_used=sql_model,
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )

                logger.info(f"Validated SQL: {validated_sql}")

                # Execute SQL with read-only approach
                with query_engine_instance.engine.connect() as conn:
                    result = conn.execute(text(validated_sql))
                    rows = result.fetchall()
                    columns = list(result.keys())

                # Return formatted text for display
                formatted_result = format_sql_results(rows, columns)

                return ChatResponse(
                    response=formatted_result,
                    sql_query=validated_sql if include_sql else None,
                    model_used=sql_model,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

            except Exception as sql_error:
                logger.error(f"SQL generation/execution failed: {str(sql_error)}")
                return ChatResponse(
                    response=f"SQL generation failed: {str(sql_error)}. Please try rephrasing your question.",
                    sql_query=None,
                    error=str(sql_error),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

        else:  # general chat
            try:
                chat_response = generate_chat_response(message.message)
                return ChatResponse(
                    response=chat_response,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            except RuntimeError as chat_error:
                logger.error(f"Chat response generation failed: {str(chat_error)}")
                return ChatResponse(
                    response="Service unavailable: LLM models are required for chat responses. Please ensure Ollama is running with phi3:mini model.",
                    sql_query=None,
                    error=str(chat_error),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return ChatResponse(
            response="An unexpected error occurred while processing your request. Please try again.",
            sql_query=None,
            error=str(e),
            execution_time=(datetime.now() - start_time).total_seconds()
        )

@app.get("/table-info")
async def get_table_info():
    """Get table schema and metadata information"""
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

@app.get("/models")
async def get_available_models():
    """Get list of available SQL generation models"""
    try:
        # Check what models are actually available in Ollama
        ollama_url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags"
        response = requests.get(ollama_url, timeout=5)
        response.raise_for_status()

        ollama_models = response.json()
        available_models = [model["name"] for model in ollama_models.get("models", [])]

        models_status = {}
        for model_name in AVAILABLE_SQL_MODELS.keys():
            models_status[model_name] = model_name in available_models

        return {
            "available_models": AVAILABLE_SQL_MODELS,
            "default_model": DEFAULT_SQL_MODEL,
            "intent_model": INTENT_MODEL,
            "ollama_models": available_models,
            "models_status": models_status,
            "local_setup": True
        }
    except Exception as e:
        logger.error(f"Failed to check Ollama models: {str(e)}")
        return {
            "available_models": AVAILABLE_SQL_MODELS,
            "default_model": DEFAULT_SQL_MODEL,
            "intent_model": INTENT_MODEL,
            "error": f"Could not connect to Ollama: {str(e)}",
            "local_setup": True
        }

@app.get("/health/models")
async def check_model_health():
    """Check health of local Ollama models and database"""
    health_status = {
        "ollama_service": "unhealthy",
        "intent_model": "unhealthy",
        "sql_models": "unhealthy",
        "database": "unhealthy"
    }

    # Check Ollama service
    try:
        ollama_url = f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags"
        response = requests.get(ollama_url, timeout=5)
        response.raise_for_status()

        models_data = response.json()
        available_models = [model["name"] for model in models_data.get("models", [])]
        health_status["ollama_service"] = "healthy"

        # Check specific models
        health_status["intent_model"] = "healthy" if INTENT_MODEL in available_models else "missing"

        sql_models_available = any(model in available_models for model in AVAILABLE_SQL_MODELS.keys())
        health_status["sql_models"] = "healthy" if sql_models_available else "missing"

    except Exception as e:
        logger.error(f"Ollama service health check failed: {str(e)}")

    # Check database
    try:
        with query_engine_instance.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            health_status["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")

    overall_healthy = all(status == "healthy" for status in health_status.values())

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "components": health_status,
        "setup": "local_ollama"
    }

@app.get("/system-prompts", response_model=SystemPromptsResponse)
async def get_system_prompts():
    """Get current and default system prompts for both models"""
    return SystemPromptsResponse(
        intent_prompt=current_intent_prompt,
        sql_prompt=current_sql_prompt,
        default_intent_prompt=DEFAULT_INTENT_PROMPT,
        default_sql_prompt=DEFAULT_SQL_PROMPT
    )

@app.post("/system-prompts")
async def update_system_prompt(prompt_update: SystemPromptUpdate):
    """Update system prompt for a specific model"""
    global current_intent_prompt, current_sql_prompt

    if prompt_update.model_type == "intent":
        current_intent_prompt = prompt_update.prompt
        return {"message": "Intent model prompt updated successfully", "model_type": "intent"}
    elif prompt_update.model_type == "sql":
        current_sql_prompt = prompt_update.prompt
        return {"message": "SQL model prompt updated successfully", "model_type": "sql"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'intent' or 'sql'")

@app.post("/system-prompts/reset")
async def reset_system_prompts():
    """Reset all system prompts to defaults"""
    global current_intent_prompt, current_sql_prompt

    current_intent_prompt = DEFAULT_INTENT_PROMPT
    current_sql_prompt = DEFAULT_SQL_PROMPT

    return {
        "message": "All system prompts reset to defaults",
        "reset_models": ["intent", "sql"]
    }

@app.post("/system-prompts/reset/{model_type}")
async def reset_specific_system_prompt(model_type: str):
    """Reset system prompt for a specific model to default"""
    global current_intent_prompt, current_sql_prompt

    if model_type == "intent":
        current_intent_prompt = DEFAULT_INTENT_PROMPT
        return {"message": "Intent model prompt reset to default", "model_type": "intent"}
    elif model_type == "sql":
        current_sql_prompt = DEFAULT_SQL_PROMPT
        return {"message": "SQL model prompt reset to default", "model_type": "sql"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'intent' or 'sql'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)