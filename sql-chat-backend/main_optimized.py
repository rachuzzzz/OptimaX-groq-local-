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
from datetime import datetime
import asyncio

# Import optimization modules
from heuristic_router import get_heuristic_router
from async_inference import get_inference_engine, cleanup_inference_engine
from query_cache import get_query_cache

load_dotenv()

app = FastAPI(title="OptimaX SQL Chat API - Optimized")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize optimization components
heuristic_router = get_heuristic_router(fallback_to_llm=True)
inference_engine = get_inference_engine()
query_cache = get_query_cache(max_size=500, default_ttl=3600)

AVAILABLE_SQL_MODELS = {
    "qwen2.5-coder:3b": "Qwen2.5-Coder 3B - Fast SQL generation",
    "codellama:7b-instruct-q4_K_M": "CodeLlama 7B Instruct (Quantized) - Fallback model"
}

DEFAULT_SQL_MODEL = "qwen2.5-coder:3b"
INTENT_MODEL = "phi3:mini"

# Default system prompts
DEFAULT_INTENT_PROMPT = """Classify the user's intent.

SQL_INTENT: Questions about traffic accident data, statistics, counts, weather, locations, severity, patterns
CHAT_INTENT: Greetings, general chat, questions about system capabilities

Respond ONLY with "SQL_INTENT" or "CHAT_INTENT"."""

DEFAULT_SQL_PROMPT = """You are a PostgreSQL expert. Generate ONLY the SQL query, no explanations.

Database Schema:
{schema_text}

Important Data Types & Values:
- severity: integer (1=low, 2=medium, 3=high, 4=severe)
- weather_condition: text values like 'Snow', 'Rain', 'Clear', etc.
- state: 2-letter codes like 'CA', 'TX', 'FL'
- Boolean columns: true/false values (amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop, traffic_calming, traffic_signal, turning_loop)

CRITICAL RULES:
1. ALWAYS use aggregation (COUNT, AVG, SUM, etc.) - NEVER return raw accident records
2. Use exact column names from schema
3. severity is INTEGER: use numbers (1,2,3,4) not text
4. For "severe": use severity >= 3 or severity = 4
5. Always use PostgreSQL syntax
6. Group results meaningfully (by city, state, severity, weather, etc.)
7. Order results by count/metric DESC
8. Use LIMIT 10-20 for top results
9. For location queries: use traffic_signal = true (not separate tables)
10. NO PostGIS functions - use simple WHERE conditions
11. NO geography/geometry casting or spatial functions

Query Pattern Examples:
❌ BAD: SELECT * FROM us_accidents WHERE traffic_signal = true AND state = 'CA';
✅ GOOD: SELECT city, COUNT(*) as accident_count FROM us_accidents WHERE traffic_signal = true AND state = 'CA' GROUP BY city ORDER BY accident_count DESC LIMIT 10;

❌ BAD: SELECT id, start_time, severity FROM us_accidents WHERE weather_condition = 'Snow';
✅ GOOD: SELECT state, COUNT(*) as snow_accidents FROM us_accidents WHERE weather_condition ILIKE '%Snow%' GROUP BY state ORDER BY snow_accidents DESC LIMIT 10;

More Examples:
- "accidents near traffic signals in CA" -> SELECT city, COUNT(*) as count FROM us_accidents WHERE traffic_signal = true AND state = 'CA' GROUP BY city ORDER BY count DESC LIMIT 10;
- "severe accidents" -> SELECT state, COUNT(*) as severe_count FROM us_accidents WHERE severity >= 3 GROUP BY state ORDER BY severe_count DESC LIMIT 15;
- "accidents by weather" -> SELECT weather_condition, COUNT(*) as count FROM us_accidents GROUP BY weather_condition ORDER BY count DESC LIMIT 10;

Question: {question}

PostgreSQL Query (MUST use aggregation):"""

# Current system prompts (can be modified by users)
current_intent_prompt = DEFAULT_INTENT_PROMPT
current_sql_prompt = DEFAULT_SQL_PROMPT

# Performance metrics
performance_metrics = {
    "total_requests": 0,
    "heuristic_hits": 0,
    "llm_fallbacks": 0,
    "cache_hits": 0,
    "avg_response_time": 0.0,
    "response_times": []
}

async def route_query_optimized(user_message: str) -> str:
    """
    Optimized query routing with heuristics first, LLM fallback

    Returns: "sql" or "chat"
    """
    # Check cache first
    cached_route = query_cache.get_route_decision(user_message)
    if cached_route:
        logger.info(f"Route decision from cache: {cached_route}")
        performance_metrics["cache_hits"] += 1
        return cached_route

    # Try heuristic routing first (fast, no LLM)
    heuristic_result = heuristic_router.route(user_message)

    if heuristic_result is not None:
        # High confidence heuristic decision
        confidence = heuristic_router.get_confidence(user_message)
        logger.info(f"Heuristic routing: {heuristic_result} (confidence: {confidence:.2f})")
        performance_metrics["heuristic_hits"] += 1

        # Cache the decision
        query_cache.cache_route_decision(user_message, heuristic_result)
        return heuristic_result

    # Fallback to LLM routing for ambiguous cases
    logger.info("Ambiguous query, falling back to LLM routing")
    performance_metrics["llm_fallbacks"] += 1

    input_text = f"{current_intent_prompt}\n\nUser question: {user_message}\n\nIntent:"

    try:
        generated_text = await inference_engine.generate_with_timeout(
            model=INTENT_MODEL,
            prompt=input_text,
            temperature=0.1,
            max_tokens=50,
            timeout_seconds=15.0
        )

        generated_text = generated_text.strip().upper()

        if "SQL_INTENT" in generated_text:
            route = "sql"
        elif "CHAT_INTENT" in generated_text:
            route = "chat"
        else:
            # Default to SQL if unclear
            logger.warning(f"LLM unclear response: {generated_text}, defaulting to SQL")
            route = "sql"

        # Cache the decision
        query_cache.cache_route_decision(user_message, route)
        return route

    except Exception as e:
        logger.error(f"LLM routing failed: {str(e)}, defaulting to SQL")
        return "sql"  # Safe default

async def generate_chat_response_async(user_message: str) -> str:
    """Generate chat response asynchronously"""
    # Check cache first
    cached_response = query_cache.get_chat_response(user_message)
    if cached_response:
        logger.info("Chat response from cache")
        performance_metrics["cache_hits"] += 1
        return cached_response

    chat_prompt = f"""You are OptimaX, a helpful AI assistant specialized in analyzing US traffic accident data.
You are friendly and professional. Keep responses concise and helpful.
If asked about your capabilities, mention you can analyze US traffic accident data through SQL queries.

User: {user_message}
Assistant:"""

    try:
        response = await inference_engine.generate_with_timeout(
            model=INTENT_MODEL,
            prompt=chat_prompt,
            temperature=0.7,
            max_tokens=200,
            timeout_seconds=20.0
        )

        if not response or len(response) < 5:
            raise RuntimeError("Chat response too short or empty")

        # Cache the response
        query_cache.cache_chat_response(user_message, response)
        return response

    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        raise RuntimeError(f"Chat response generation failed: {str(e)}")

class ChatMessage(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    sql_model: Optional[str] = DEFAULT_SQL_MODEL
    include_sql: Optional[bool] = None
    row_limit: Optional[int] = 50

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    query_results: Optional[List[Dict[str, Any]]] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    routing_method: Optional[str] = None  # "heuristic", "llm", "cache"

class SystemPromptUpdate(BaseModel):
    model_type: str
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
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL not set in environment")

            self.engine = create_engine(database_url)
            logger.info("Database connection established")

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

    def validate_sql(self, sql_query: str, row_limit: int = 50) -> tuple[bool, str]:
        """Validate SQL for safety and syntax"""
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return False, "Invalid SQL syntax"

            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'ALTER', 'INSERT', 'CREATE', 'TRUNCATE']
            sql_upper = sql_query.upper()

            for keyword in dangerous_keywords:
                if re.search(rf'\b{keyword}\b', sql_upper):
                    return False, f"Dangerous operation detected: {keyword}"

            if not sql_upper.strip().startswith('SELECT'):
                return False, "Only SELECT statements are allowed"

            # Add or update LIMIT clause
            if 'LIMIT' not in sql_upper:
                if not sql_query.rstrip().endswith(';'):
                    sql_query = sql_query.rstrip() + f' LIMIT {row_limit};'
                else:
                    sql_query = sql_query.rstrip(';') + f' LIMIT {row_limit};'
            else:
                limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
                if limit_match:
                    existing_limit = int(limit_match.group(1))
                    if existing_limit > row_limit:
                        sql_query = re.sub(r'LIMIT\s+\d+', f'LIMIT {row_limit}', sql_query, flags=re.IGNORECASE)

            if not sql_query.rstrip().endswith(';'):
                sql_query = sql_query.rstrip() + ';'

            return True, sql_query

        except Exception as e:
            return False, f"SQL validation error: {str(e)}"

    async def generate_sql_async(self, question: str, model_name: str = DEFAULT_SQL_MODEL, system_prompt: str = None) -> str:
        """Generate SQL asynchronously using optimized inference"""
        schema_text = self.table_schema.get("schema_text", "") if self.table_schema else ""

        if system_prompt:
            prompt = f"{system_prompt}\n\nQuestion: {question}"
        else:
            prompt = current_sql_prompt.format(schema_text=schema_text, question=question)

        logger.info(f"Generating SQL with model {model_name}")

        try:
            generated_sql = await inference_engine.generate_with_timeout(
                model=model_name,
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                timeout_seconds=30.0
            )

            if not generated_sql:
                raise Exception("Empty response from model")

            # Extract SQL from markdown code blocks if present
            if "```" in generated_sql:
                code_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', generated_sql, re.DOTALL)
                if code_block_match:
                    generated_sql = code_block_match.group(1).strip()

            # Remove common prefixes
            sql_prefixes = ["SQL:", "Query:", "PostgreSQL:", "SELECT"]
            for prefix in sql_prefixes[:-1]:
                if generated_sql.upper().startswith(prefix):
                    generated_sql = generated_sql[len(prefix):].strip()

            logger.info(f"Generated SQL: {generated_sql}")
            return generated_sql

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise Exception(f"SQL generation failed: {str(e)}")


query_engine_instance = TextToSQLEngine()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await cleanup_inference_engine()
    logger.info("Inference engine cleaned up")

@app.get("/")
async def root():
    return {"message": "OptimaX SQL Chat API - Optimized", "version": "2.0"}

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
    if not rows:
        return "No results found."

    header = " | ".join(columns) if len(columns) > 1 else columns[0]
    separator = "-" * len(header)

    formatted_data = [header, separator]

    for i, row in enumerate(rows):
        if len(row) == 2 and isinstance(row[1], (int, float)):
            formatted_data.append(f"{row[0]} | {row[1]:,}")
        elif len(row) == 1:
            value = row[0]
            if isinstance(value, (int, float)) and value > 1000:
                formatted_data.append(f"{value:,}")
            else:
                formatted_data.append(str(value))
        else:
            row_data = []
            for col in row:
                if isinstance(col, (int, float)) and col > 1000:
                    row_data.append(f"{col:,}")
                elif col is None:
                    row_data.append("NULL")
                else:
                    row_data.append(str(col))
            formatted_data.append(" | ".join(row_data))

    if len(formatted_data) > 52:
        result_rows = formatted_data[:52]
        result_rows.append(f"\n... and {len(rows) - 50} more rows")
        return "\n".join(result_rows)
    else:
        return "\n".join(formatted_data)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    start_time = datetime.now()
    performance_metrics["total_requests"] += 1

    try:
        # Validate sql_model if provided
        if message.sql_model and message.sql_model not in AVAILABLE_SQL_MODELS:
            return ChatResponse(
                response=f"Invalid model '{message.sql_model}'. Available models: {list(AVAILABLE_SQL_MODELS.keys())}",
                error=f"Model '{message.sql_model}' not available",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        # Check SQL query cache first
        cached_sql_result = query_cache.get_sql_query(message.message)
        if cached_sql_result and "query_results" in cached_sql_result:
            logger.info("Complete SQL result from cache")
            performance_metrics["cache_hits"] += 1

            # Handle cached results
            query_results = cached_sql_result["query_results"]
            if query_results and len(query_results) > 0:
                formatted_result = format_sql_results(
                    [[v for v in row.values()] for row in query_results],
                    list(query_results[0].keys())
                )
            else:
                formatted_result = "No results found."

            execution_time = (datetime.now() - start_time).total_seconds()
            performance_metrics["response_times"].append(execution_time)

            return ChatResponse(
                response=formatted_result,
                sql_query=cached_sql_result["sql_query"] if message.include_sql else None,
                query_results=cached_sql_result["query_results"],
                execution_time=execution_time,
                routing_method="cache"
            )

        # Route the query using optimized routing
        try:
            route = await route_query_optimized(message.message)
            logger.info(f"Query routed to: {route}")
        except Exception as router_error:
            logger.error(f"Router error: {str(router_error)}")
            return ChatResponse(
                response="Service temporarily unavailable. Please try again.",
                sql_query=None,
                error=str(router_error),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

        if route == "sql":
            # Generate SQL using async inference
            sql_model = message.sql_model or DEFAULT_SQL_MODEL
            include_sql = message.include_sql if message.include_sql is not None else (os.getenv("DEBUG") == "true")

            try:
                sql_query = await query_engine_instance.generate_sql_async(
                    question=message.message,
                    model_name=sql_model,
                    system_prompt=message.system_prompt
                )

                # Validate SQL
                row_limit = message.row_limit or 50
                is_valid, validated_sql = query_engine_instance.validate_sql(sql_query, row_limit)
                if not is_valid:
                    return ChatResponse(
                        response=f"Query validation failed: {validated_sql}",
                        sql_query=None,
                        error=validated_sql,
                        model_used=sql_model,
                        execution_time=(datetime.now() - start_time).total_seconds()
                    )

                # Execute SQL
                with query_engine_instance.engine.connect() as conn:
                    result = conn.execute(text(validated_sql))
                    rows = result.fetchall()
                    columns = list(result.keys())

                formatted_result = format_sql_results(rows, columns)
                query_results = [dict(row._mapping) for row in rows] if rows else []

                # Cache the result
                query_cache.cache_sql_query(message.message, validated_sql, query_results)

                execution_time = (datetime.now() - start_time).total_seconds()
                performance_metrics["response_times"].append(execution_time)
                performance_metrics["response_times"] = performance_metrics["response_times"][-100:]  # Keep last 100

                return ChatResponse(
                    response=formatted_result,
                    sql_query=validated_sql if include_sql else None,
                    query_results=query_results,
                    model_used=sql_model,
                    execution_time=execution_time,
                    routing_method="heuristic" if performance_metrics["heuristic_hits"] > 0 else "llm"
                )

            except Exception as sql_error:
                logger.error(f"SQL generation/execution failed: {str(sql_error)}")
                return ChatResponse(
                    response=f"SQL generation failed: {str(sql_error)}. Please try rephrasing your question.",
                    sql_query=None,
                    error=str(sql_error),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

        else:  # chat
            try:
                chat_response = await generate_chat_response_async(message.message)

                execution_time = (datetime.now() - start_time).total_seconds()
                performance_metrics["response_times"].append(execution_time)

                return ChatResponse(
                    response=chat_response,
                    execution_time=execution_time,
                    routing_method="heuristic" if performance_metrics["heuristic_hits"] > 0 else "llm"
                )
            except RuntimeError as chat_error:
                logger.error(f"Chat response generation failed: {str(chat_error)}")
                return ChatResponse(
                    response="Service temporarily unavailable for chat. Please try again.",
                    sql_query=None,
                    error=str(chat_error),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return ChatResponse(
            response="An unexpected error occurred. Please try again.",
            sql_query=None,
            error=str(e),
            execution_time=(datetime.now() - start_time).total_seconds()
        )

@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics for the optimized backend"""
    cache_stats = query_cache.get_stats()
    inference_cache_stats = inference_engine.get_cache_stats()

    avg_response_time = sum(performance_metrics["response_times"]) / len(performance_metrics["response_times"]) if performance_metrics["response_times"] else 0

    return {
        "total_requests": performance_metrics["total_requests"],
        "heuristic_routing": {
            "hits": performance_metrics["heuristic_hits"],
            "percentage": (performance_metrics["heuristic_hits"] / performance_metrics["total_requests"] * 100) if performance_metrics["total_requests"] > 0 else 0
        },
        "llm_fallbacks": {
            "count": performance_metrics["llm_fallbacks"],
            "percentage": (performance_metrics["llm_fallbacks"] / performance_metrics["total_requests"] * 100) if performance_metrics["total_requests"] > 0 else 0
        },
        "cache_stats": {
            "query_cache": cache_stats,
            "inference_cache": inference_cache_stats,
            "total_cache_hits": performance_metrics["cache_hits"]
        },
        "response_times": {
            "average_ms": avg_response_time * 1000,
            "recent_count": len(performance_metrics["response_times"])
        }
    }

@app.post("/performance/reset")
async def reset_performance_metrics():
    """Reset performance metrics"""
    global performance_metrics
    performance_metrics = {
        "total_requests": 0,
        "heuristic_hits": 0,
        "llm_fallbacks": 0,
        "cache_hits": 0,
        "avg_response_time": 0.0,
        "response_times": []
    }
    query_cache.reset_stats()
    inference_engine.clear_cache()
    return {"message": "Performance metrics reset successfully"}

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
    return {
        "available_models": AVAILABLE_SQL_MODELS,
        "default_model": DEFAULT_SQL_MODEL,
        "intent_model": INTENT_MODEL,
        "local_setup": True,
        "optimizations": ["heuristic_routing", "async_inference", "query_cache"]
    }

@app.get("/system-prompts", response_model=SystemPromptsResponse)
async def get_system_prompts():
    """Get current and default system prompts"""
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
