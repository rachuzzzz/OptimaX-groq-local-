"""
OptimaX v4.2 - Split Visualization Architecture
================================================

BREAKTHROUGH: Visualization is now a separate one-shot LLM call
- NO agent for visualization (eliminates timeouts & infinite loops)
- NO tools for visualization (pure classification)
- NO retries (fast, deterministic)

Architecture:
- ReActAgent: SQL queries and data analysis
- One-shot LLM: Visualization classification (classify_visualization_intent)
- Separation of concerns = reliability + speed

Author: OptimaX Team
Version: 4.2
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent

from tools import (
    initialize_tools,
    get_last_sql_result,
    clear_last_sql_result,
    classify_visualization_intent,
)

load_dotenv()

# Configure logging - minimal for free tier performance
logging.basicConfig(
    level=logging.WARNING,  # Free tier optimization: reduce console spam
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep INFO for our app, WARNING for libraries


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system on startup, cleanup on shutdown"""
    global agent, db_manager, llm, custom_prompt_config

    try:
        logger.info("Initializing OptimaX v4.2...")

        # Verify API key
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found! Set it in your environment. "
                "You can get one at: https://console.groq.com/keys"
            )

        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not found in environment variables!")

        # Initialize Groq LLM with optimized settings
        llm = Groq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,        # Deterministic
            max_output_tokens=500,  # Increased for chart reasoning
        )
        logger.info(f"✓ Groq LLM initialized: {GROQ_MODEL}")

        # Initialize tools and get database manager
        tools, db_manager = initialize_tools(DATABASE_URL)
        logger.info(f"✓ Tools initialized: {len(tools)} tools")

        # Generate dynamic system prompt with actual schema
        schema_description = db_manager.get_schema_for_llm()
        dynamic_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{SCHEMA_SECTION}", schema_description)
        logger.info(f"✓ Dynamic schema generated ({len(schema_description)} chars)")

        # Load custom prompt if exists
        if os.path.exists(CUSTOM_PROMPT_FILE):
            try:
                with open(CUSTOM_PROMPT_FILE, 'r') as f:
                    import json
                    config = json.load(f)
                    custom_prompt_config["enabled"] = config.get("enabled", False)
                    custom_prompt_config["prompt"] = config.get("prompt")
                    custom_prompt_config["use_dynamic_schema"] = config.get("use_dynamic_schema", True)

                if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
                    # Use custom prompt
                    if custom_prompt_config["use_dynamic_schema"]:
                        # Inject schema into custom prompt
                        final_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", schema_description)
                        logger.info("✓ Using CUSTOM prompt with dynamic schema injection")
                    else:
                        final_prompt = custom_prompt_config["prompt"]
                        logger.info("✓ Using CUSTOM prompt (no schema injection)")
                else:
                    final_prompt = dynamic_prompt
                    logger.info("✓ Using DEFAULT dynamic prompt")
            except Exception as e:
                logger.warning(f"Failed to load custom prompt: {e}")
                final_prompt = dynamic_prompt
                logger.info("✓ Falling back to DEFAULT dynamic prompt")
        else:
            final_prompt = dynamic_prompt
            logger.info("✓ Using DEFAULT dynamic prompt")

        # Initialize base agent with final prompt (template for sessions)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,        # Enable verbose logs to debug tool calling
            max_iterations=5,    # Limited to prevent excessive queries (Groq optimization)
            system_prompt=final_prompt,
        )
        logger.info("✓ ReActAgent initialized")

        logger.info("=" * 60)
        logger.info("OptimaX v4.2 Ready!")
        logger.info(f"Architecture: Split Viz (ReActAgent + One-shot LLM)")
        logger.info(f"Tools: {len(tools)} (SQL only - Viz handled separately)")
        logger.info(f"Database: {len(db_manager.schema['tables'])} tables detected")
        logger.info(f"Prompt: {'CUSTOM' if custom_prompt_config['enabled'] else 'DEFAULT'}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    yield  # Server is running

    # Cleanup on shutdown (if needed)
    logger.info("Shutting down OptimaX...")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="OptimaX SQL Chat API",
    description="Split visualization architecture: ReActAgent for queries, one-shot LLM for viz",
    version="4.2",
    lifespan=lifespan,
)

# CORS - relaxed for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Free tier: allow all origins for easy testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Available on your API key

# Global instances
agent = None
db_manager = None
llm = None  # Global LLM instance for one-shot visualization calls
sessions: Dict[str, Dict[str, Any]] = {}  # {session_id: {agent, metadata}}


# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    include_sql: Optional[bool] = True
    include_tasks: Optional[bool] = False
    row_limit: Optional[int] = 50


class ChartSuggestion(BaseModel):
    analysis_type: str  # 'comparison', 'time_series', 'proportion', 'correlation', 'distribution'
    reasoning: str
    recommended_charts: List[Dict[str, Any]]  # [{"type": "bar", "label": "Bar Chart", "description": "...", "recommended": true}]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sql_query: Optional[str] = None
    error: Optional[str] = None
    query_results: Optional[List[Dict[str, Any]]] = None  # Frontend expects this
    data: Optional[List[Dict[str, Any]]] = None
    execution_time: float
    tasks: Optional[List[Dict[str, Any]]] = None
    clarification_needed: Optional[bool] = False
    agent_reasoning: Optional[str] = None
    chart_suggestion: Optional[Dict[str, Any]] = None  # New: LLM suggests chart types


# Base System Prompt Template - Schema will be injected dynamically
SYSTEM_PROMPT_TEMPLATE = """You are OptimaX, an autonomous AI agent for database analysis and SQL query generation.

!!!! CRITICAL RULE - READ THIS FIRST !!!!
NEVER run queries to "demonstrate capabilities" or "show examples".
If user asks "what can you do", just describe your capabilities - DO NOT RUN QUERIES.
Only run execute_sql when user asks for SPECIFIC data like "show me records" or "count entries".
Once you have the answer, STOP immediately - don't run additional queries.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

{SCHEMA_SECTION}

WHEN TO USE TOOLS:
1. get_schema: ONLY if user asks about available tables/schema
2. execute_sql: ONLY when user explicitly requests specific data
3. After get_schema for a "what can you do" question, list capabilities and STOP - no execute_sql
4. After execute_sql returns useful data, formulate answer and STOP - don't run more queries

EFFICIENCY RULES:
- Use simple queries - avoid complex JOINs when possible
- Prefer direct table queries over complex joins
- STOP after getting useful results - don't keep exploring
- If you get an error, check schema and try ONE more time, then STOP

RULES:
- ALWAYS use schema prefix if required (check schema section above)
- Always use LIMIT for safety
- For date/time: EXTRACT(MONTH FROM column_name)
- Prefer COUNT/AVG/SUM/GROUP BY over raw rows
- Order DESC for most, ASC for least

RESPONSE:
- Answer the question and STOP
- Be concise and accurate
- Don't run multiple queries unless absolutely necessary
"""

# Will be populated on startup with actual schema
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE

# Custom prompt storage
CUSTOM_PROMPT_FILE = "custom_system_prompt.txt"
custom_prompt_config = {
    "enabled": False,
    "prompt": None,
    "use_dynamic_schema": True
}


def get_or_create_session(session_id: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Get existing session or create new one - reuses base agent for free tier optimization"""
    global agent, llm

    if session_id not in sessions:
        # Use custom agent if custom prompt provided
        session_agent = agent  # Default to global agent

        if custom_prompt:
            logger.info(f"Creating session with custom system prompt: {session_id}")
            tools, _ = initialize_tools(DATABASE_URL)
            session_agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True,
                max_iterations=5,  # Limited to prevent excessive queries
                system_prompt=custom_prompt,
            )

        sessions[session_id] = {
            "agent": session_agent,
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "message_count": 0,
            "custom_prompt": custom_prompt is not None,
        }

        prompt_type = "custom" if custom_prompt else "default"
        logger.info(f"Created new session ({prompt_type} prompt): {session_id}")

    sessions[session_id]["last_active"] = datetime.now()
    return sessions[session_id]


@app.get("/")
async def root():
    return {
        "message": "OptimaX SQL Chat API v4.2",
        "version": "4.2",
        "architecture": "Split Visualization: ReActAgent (queries) + One-shot LLM (viz)",
        "features": [
            "SQL query execution via ReActAgent",
            "One-shot chart classification (no timeouts)",
            "Session-based memory",
            "Natural language interface",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if agent is None:
            return {"status": "unhealthy", "error": "Agent not initialized"}

        return {
            "status": "healthy",
            "version": "4.2",
            "model": GROQ_MODEL,
            "active_sessions": len(sessions),
            "database_tables": len(db_manager.schema['tables']) if db_manager else 0
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/database/schema")
async def get_database_schema():
    """Get current database schema information"""
    try:
        if db_manager is None:
            raise HTTPException(status_code=503, detail="Database not initialized")

        schema_summary = db_manager.get_schema_summary()
        return {
            "success": True,
            "schema": schema_summary,
            "schema_text": db_manager.get_schema_for_llm()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DatabaseConnectionRequest(BaseModel):
    database_url: str


class SystemPromptRequest(BaseModel):
    prompt: str
    use_dynamic_schema: bool = True  # Whether to inject schema into prompt


@app.post("/database/test-connection")
async def test_database_connection(request: DatabaseConnectionRequest):
    """Test a database connection without changing the current one"""
    try:
        from sqlalchemy import create_engine, text

        # Try to connect and query
        test_engine = create_engine(request.database_url)
        with test_engine.connect() as conn:
            # Test query
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # If successful, get table count
        from sqlalchemy import inspect
        inspector = inspect(test_engine)

        # Try to detect schema
        schemas = inspector.get_schema_names()
        table_count = 0

        for schema in schemas:
            if schema not in ['information_schema', 'pg_catalog']:
                tables = inspector.get_table_names(schema=schema)
                table_count += len(tables)

        test_engine.dispose()

        return {
            "success": True,
            "message": "Connection successful",
            "table_count": table_count,
            "schemas": [s for s in schemas if s not in ['information_schema', 'pg_catalog']]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/database/connect")
async def connect_to_database(request: DatabaseConnectionRequest):
    """Connect to a new database and reload schema dynamically"""
    global agent, db_manager, llm, sessions, custom_prompt_config, DATABASE_URL

    try:
        logger.info(f"Attempting to connect to new database...")

        # First, test the connection
        from sqlalchemy import create_engine, text, inspect

        test_engine = create_engine(request.database_url)
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # Get schema info
        inspector = inspect(test_engine)
        schemas = inspector.get_schema_names()
        table_count = 0
        schema_names = []

        for schema in schemas:
            if schema not in ['information_schema', 'pg_catalog']:
                tables = inspector.get_table_names(schema=schema)
                table_count += len(tables)
                schema_names.append(schema)

        test_engine.dispose()

        # Connection successful - now switch to it
        DATABASE_URL = request.database_url

        # Reinitialize tools with new database
        tools, new_db_manager = initialize_tools(DATABASE_URL)
        db_manager = new_db_manager
        logger.info(f"✓ Database manager reinitialized: {table_count} tables")

        # Generate new schema description
        schema_description = db_manager.get_schema_for_llm()

        # Determine final prompt with new schema
        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            if custom_prompt_config["use_dynamic_schema"]:
                final_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", schema_description)
                prompt_type = "CUSTOM (with dynamic schema)"
            else:
                final_prompt = custom_prompt_config["prompt"]
                prompt_type = "CUSTOM (static)"
        else:
            final_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{SCHEMA_SECTION}", schema_description)
            prompt_type = "DEFAULT (dynamic)"

        # Reinitialize agent with new schema
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=5,
            system_prompt=final_prompt,
        )

        # Clear all existing sessions
        cleared_sessions = len(sessions)
        sessions.clear()

        logger.info("=" * 60)
        logger.info(f"✓ Connected to new database!")
        logger.info(f"Tables: {table_count}")
        logger.info(f"Schemas: {', '.join(schema_names)}")
        logger.info(f"Prompt: {prompt_type}")
        logger.info(f"Sessions cleared: {cleared_sessions}")
        logger.info("=" * 60)

        return {
            "success": True,
            "message": "Successfully connected to database",
            "table_count": table_count,
            "schemas": schema_names,
            "prompt_type": prompt_type,
            "sessions_cleared": cleared_sessions
        }

    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/system-prompt/active")
async def get_active_system_prompt():
    """Get the currently active system prompt"""
    try:
        global custom_prompt_config, db_manager

        schema_description = db_manager.get_schema_for_llm() if db_manager else ""

        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            if custom_prompt_config["use_dynamic_schema"]:
                active_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", schema_description)
            else:
                active_prompt = custom_prompt_config["prompt"]

            return {
                "type": "custom",
                "prompt": active_prompt,
                "raw_prompt": custom_prompt_config["prompt"],
                "use_dynamic_schema": custom_prompt_config["use_dynamic_schema"],
                "schema_description": schema_description if custom_prompt_config["use_dynamic_schema"] else None
            }
        else:
            # Default dynamic prompt
            active_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{SCHEMA_SECTION}", schema_description)
            return {
                "type": "default",
                "prompt": active_prompt,
                "raw_prompt": SYSTEM_PROMPT_TEMPLATE,
                "use_dynamic_schema": True,
                "schema_description": schema_description
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/save")
async def save_system_prompt(request: SystemPromptRequest):
    """Save a custom system prompt permanently"""
    try:
        global custom_prompt_config
        import json

        config = {
            "enabled": True,
            "prompt": request.prompt,
            "use_dynamic_schema": request.use_dynamic_schema,
            "saved_at": datetime.now().isoformat()
        }

        # Save to file
        with open(CUSTOM_PROMPT_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Update in-memory config
        custom_prompt_config["enabled"] = True
        custom_prompt_config["prompt"] = request.prompt
        custom_prompt_config["use_dynamic_schema"] = request.use_dynamic_schema

        logger.info("✓ Custom system prompt saved")

        return {
            "success": True,
            "message": "Custom prompt saved. Restart backend to apply changes.",
            "config": config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/reset")
async def reset_system_prompt():
    """Reset to default dynamic system prompt"""
    try:
        global custom_prompt_config

        # Remove custom prompt file
        if os.path.exists(CUSTOM_PROMPT_FILE):
            os.remove(CUSTOM_PROMPT_FILE)

        # Update in-memory config
        custom_prompt_config["enabled"] = False
        custom_prompt_config["prompt"] = None
        custom_prompt_config["use_dynamic_schema"] = True

        logger.info("✓ Reset to default dynamic prompt")

        return {
            "success": True,
            "message": "Reset to default prompt. Restart backend to apply changes."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/apply")
async def apply_system_prompt():
    """Apply the current saved system prompt without restart (hot reload)"""
    try:
        global agent, custom_prompt_config, sessions, llm, db_manager

        # Generate dynamic schema
        schema_description = db_manager.get_schema_for_llm()

        # Load custom prompt if exists
        if os.path.exists(CUSTOM_PROMPT_FILE):
            try:
                with open(CUSTOM_PROMPT_FILE, 'r') as f:
                    import json
                    config = json.load(f)
                    custom_prompt_config["enabled"] = config.get("enabled", False)
                    custom_prompt_config["prompt"] = config.get("prompt")
                    custom_prompt_config["use_dynamic_schema"] = config.get("use_dynamic_schema", True)
            except Exception as e:
                logger.error(f"Failed to load custom prompt during apply: {e}")
                custom_prompt_config["enabled"] = False

        # Determine final prompt
        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            if custom_prompt_config["use_dynamic_schema"]:
                final_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", schema_description)
                prompt_type = "CUSTOM (with dynamic schema)"
            else:
                final_prompt = custom_prompt_config["prompt"]
                prompt_type = "CUSTOM (static)"
        else:
            final_prompt = SYSTEM_PROMPT_TEMPLATE.replace("{SCHEMA_SECTION}", schema_description)
            prompt_type = "DEFAULT (dynamic)"

        # Reinitialize global agent with new prompt
        tools, _ = initialize_tools(DATABASE_URL)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=5,  # Limited to prevent excessive queries (Groq optimization)
            system_prompt=final_prompt,
        )

        # Clear all sessions to force them to use new agent
        cleared_sessions = len(sessions)
        sessions.clear()

        logger.info(f"✓ Applied system prompt: {prompt_type}")
        logger.info(f"✓ Cleared {cleared_sessions} sessions")

        return {
            "success": True,
            "message": f"System prompt applied successfully! Using {prompt_type}",
            "prompt_type": prompt_type,
            "sessions_cleared": cleared_sessions,
            "prompt_preview": final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt
        }

    except Exception as e:
        logger.error(f"Failed to apply system prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply prompt: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint - compatible with frontend expectations"""
    start_time = datetime.now()

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Get or create session (with custom prompt support)
        session_id = message.session_id or str(uuid.uuid4())
        session = get_or_create_session(session_id, message.system_prompt)
        session["message_count"] += 1

        session_agent: ReActAgent = session["agent"]

        prompt_status = "custom prompt" if session.get("custom_prompt") else "default prompt"
        logger.info(
            f"Processing query [session={session_id}, {prompt_status}]: {message.message[:100]}..."
        )

        # ===================================================================
        # INTENT ROUTING - Four-gate system
        # visualization → greeting → ambiguous entity → database query
        # ===================================================================
        user_msg_lower = message.message.lower().strip()

        # GATE 1: Visualization Intent (one-shot LLM classification)
        viz_keywords = ["visualize", "vizualize", "graph", "chart", "plot", "pictorially", "show chart", "show as chart", "display chart"]
        is_viz_request = any(keyword in user_msg_lower for keyword in viz_keywords)

        # Clear previous tool results ONLY if NOT a visualization request
        # (Visualization needs the cached data from previous query!)
        if not is_viz_request:
            clear_last_sql_result()
            logger.info("Cleared cached SQL results (new query)")

        # GATE 1 HANDLER: Visualization Intent
        if is_viz_request:
            last_result = get_last_sql_result()
            logger.info(f"Visualization request - checking cached data: {last_result is not None}")

            if last_result and last_result.get("success") and last_result.get("data"):
                logger.info(f"⚡ One-shot visualization classification (no agent, no tools) - {len(last_result['data'])} rows cached")

                # Prepare data summary for LLM
                data = last_result["data"]
                columns = last_result.get("columns", [])

                # Infer column types from first row
                column_types = {}
                if data and len(data) > 0:
                    first_row = data[0]
                    for col in columns:
                        value = first_row.get(col)
                        if isinstance(value, (int, float)):
                            column_types[col] = "number"
                        elif isinstance(value, bool):
                            column_types[col] = "boolean"
                        elif isinstance(value, str):
                            # Check if it looks like a date
                            if any(keyword in col.lower() for keyword in ["date", "time", "timestamp"]):
                                column_types[col] = "date"
                            else:
                                column_types[col] = "string"
                        else:
                            column_types[col] = "unknown"

                # Generate data summary
                data_summary = f"Dataset with {len(columns)} columns: {', '.join(columns)}"

                # ONE-SHOT LLM CALL (no agent, no retries)
                chart_classification = classify_visualization_intent(
                    llm=llm,
                    data_summary=data_summary,
                    column_types=column_types,
                    row_count=len(data)
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                # Clear the cached result after using it (prevent stale data on next viz request)
                clear_last_sql_result()

                # Return visualization suggestion
                return ChatResponse(
                    response=f"I can visualize this data for you. Here are my suggestions based on the data structure:",
                    session_id=session_id,
                    sql_query=last_result.get("sql"),
                    query_results=data,
                    data=data,
                    execution_time=execution_time,
                    chart_suggestion=chart_classification,
                    error=None,
                )
            else:
                # No cached data for visualization
                logger.warning("No cached data found for visualization request")
                execution_time = (datetime.now() - start_time).total_seconds()
                return ChatResponse(
                    response="I don't have any data to visualize. Please run a query first, then ask me to visualize it.",
                    session_id=session_id,
                    execution_time=execution_time,
                    error="No data available for visualization",
                )

        # GATE 2: Greeting Intent (fast-path response)
        greeting_keywords = [
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        simple_queries = [
            "help",
            "what can you do",
            "who are you",
            "thanks",
            "thank you",
            "okay",
            "ok",
            "cool",
        ]

        is_greeting = (
            any(user_msg_lower == keyword for keyword in greeting_keywords)
            or any(user_msg_lower == keyword for keyword in simple_queries)
        ) and not is_viz_request  # Don't fast-path if it's a viz request

        # GATE 2 HANDLER: Greeting Intent
        if is_greeting:
            logger.info(f"Fast-path greeting detected: {message.message}")
            execution_time = (datetime.now() - start_time).total_seconds()

            greeting_response = """Hello! 👋 I'm **OptimaX** — your AI assistant for airline booking and flight data.

I can help you:
- 🔍 Generate SQL queries and analyze flight data
- 📊 Visualize trends with automatic charts (when you ask)
- 💡 Discover insights from flights, bookings, and passenger data

**Try asking:**
- "Top 10 busiest flight routes"
- "What are the most popular airports?"
- "Show me flights from JFK to LAX"
- "What is the average booking price?"

What would you like to explore?"""

            return ChatResponse(
                response=greeting_response,
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                tasks=[] if not message.include_tasks else None,
                clarification_needed=False,
                agent_reasoning=None,
                chart_recommendation=None,
                error=None,
            )

        # GATE 3: Ambiguous Entity Detection (mentions name/entity without clear action)
        # Detects queries like "tell me about X" without database action verbs
        ambiguous_entity_patterns = [
            "tell me about", "tell me more about", "more about",
            "what about", "who is", "what is", "about",
            "information on", "info on", "details on", "details about",
        ]

        database_action_keywords = [
            # Action verbs - these override ambiguity
            "show", "list", "find", "get", "count", "compare", "filter",
            # Data language
            "records", "rows", "entries", "data",
            # Analysis words
            "average", "total", "top", "most", "least", "all",
        ]

        has_ambiguous_entity = any(pattern in user_msg_lower for pattern in ambiguous_entity_patterns)
        has_database_action = any(keyword in user_msg_lower for keyword in database_action_keywords)

        # GATE 3 HANDLER: Ambiguous Entity → Disambiguation Response
        if has_ambiguous_entity and not has_database_action:
            # Extract the entity name (simple heuristic: words after the pattern)
            entity_name = "this entity"
            for pattern in ambiguous_entity_patterns:
                if pattern in user_msg_lower:
                    parts = user_msg_lower.split(pattern)
                    if len(parts) > 1:
                        entity_name = f'"{parts[1].strip()}"'
                    break

            logger.info(f"Ambiguous entity detected: {entity_name}")
            execution_time = (datetime.now() - start_time).total_seconds()

            disambiguation_response = f"""I found records related to {entity_name} in this database.

This name may correspond to multiple records or entities.

To continue, you can:
• ask for a summary from the database
• request specific fields
• add filters to narrow the result"""

            return ChatResponse(
                response=disambiguation_response,
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                clarification_needed=True,
                error=None,
            )

        # GATE 4: Database Query Intent Check (CRITICAL - prevents non-SQL queries from timing out)
        # Only route to SQL agent if user explicitly wants to retrieve/analyze data
        has_database_intent = has_database_action

        # GATE 4 HANDLER: No Database Intent → General Clarification Response
        if not has_database_intent:
            logger.info(f"No clear database intent detected: {message.message}")
            execution_time = (datetime.now() - start_time).total_seconds()

            clarification_response = """I can query the database, but I need more context.

What would you like to do with this information?

**Examples:**
- "Show me flights departing from JFK"
- "Count total bookings"
- "Find top 10 passengers with most points"
- "List all airports in the database\""""

            return ChatResponse(
                response=clarification_response,
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                clarification_needed=True,
                error=None,
            )

        # ===================================================================
        # SQL AGENT EXECUTION - Only reached if clear database action detected
        # (Passed all 4 gates: not viz, not greeting, not ambiguous, has action)
        # ===================================================================
        logger.info(f"Database action confirmed - routing to SQL agent")

        # Execute query through agent with timeout
        try:
            response = await asyncio.wait_for(
                session_agent.achat(message.message),
                timeout=35,  # Groq can be slower on complex queries with joins
            )
            response_text = str(response)
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout for session {session_id}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return ChatResponse(
                response=(
                    "Groq server seems busy or timed out. "
                    "Please try again in a few seconds."
                ),
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                error="Timeout",
            )
        except Exception as e:
            # Catch things like "Reached max iterations."
            logger.error(f"Agent error for session {session_id}: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return ChatResponse(
                response=(
                    "I ran into an internal reasoning limit while processing your query. "
                    "Try simplifying or narrowing the question slightly."
                ),
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                error=str(e),
            )

        # Extract SQL and data from global storage (set by tools)
        sql_query = None
        data = None

        last_result = get_last_sql_result()
        if last_result and last_result.get("success"):
            sql_query = last_result.get("sql")
            data = last_result.get("data")
            logger.info(
                f"Extracted SQL and {len(data) if data else 0} rows from tool execution"
            )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Add user hint if data was returned (encourage chart usage)
        if data and len(data) > 0:
            response_text += (
                "\n\n💡 *Tip: Ask me to 'visualize this' or 'show as chart' "
                "and I'll suggest the best chart options.*"
            )

        # Return in format frontend expects
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sql_query=sql_query,
            query_results=data,  # Frontend expects 'query_results'
            data=data,
            execution_time=execution_time,
            tasks=[] if not message.include_tasks else None,
            clarification_needed=False,
            agent_reasoning=None,
            chart_suggestion=None,  # No chart suggestions from agent anymore
            error=None,
        )

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        execution_time = (datetime.now() - start_time).total_seconds()

        # Free tier friendly error message
        error_msg = (
            "Groq server seems busy or encountered an error. "
            "Please try again in a few seconds."
        )
        if "429" in str(e) or "rate limit" in str(e).lower():
            error_msg = "Rate limit reached. Please wait a moment before trying again."

        return ChatResponse(
            response=error_msg,
            session_id=message.session_id or str(uuid.uuid4()),
            sql_query=None,
            query_results=None,
            data=None,
            execution_time=execution_time,
            error=str(e),
        )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    session_list = []
    for sid, session in sessions.items():
        session_list.append(
            {
                "session_id": sid,
                "created_at": session["created_at"].isoformat(),
                "last_active": session["last_active"].isoformat(),
                "message_count": session["message_count"],
            }
        )

    return {
        "total_sessions": len(session_list),
        "sessions": session_list,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/models")
async def get_models():
    """Get model information"""
    return {
        "model": GROQ_MODEL,
        "provider": "Groq",
        "architecture": "Single LLM for all tasks",
        "version": "4.1",
    }


@app.get("/table-info")
async def get_table_info():
    """Get database table information"""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        schema = db_manager.get_schema_text()
        return {
            "schema": schema,
            "table": "us_accidents",
            "total_records": "7.7M+",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
async def get_performance():
    """Get performance metrics (for frontend compatibility)"""
    total_sessions = len(sessions)
    total_messages = sum(s["message_count"] for s in sessions.values())

    return {
        "total_requests": total_messages,
        "active_sessions": total_sessions,
        "total_sessions_created": total_sessions,
        "avg_tasks_per_query": 1.0,
        "clarification_requests": 0,
        "clarification_rate": 0.0,
        "multi_step_queries": 0,
    }


@app.get("/agent/info")
async def get_agent_info():
    """Get agent information (for frontend compatibility)"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return {
        "agent_type": "LlamaIndex ReActAgent",
        "agent_model": GROQ_MODEL,
        "tools": ["execute_sql", "get_schema"],
        "visualization": "One-shot LLM (separate from agent)",
        "features": {
            "multi_turn_memory": True,
            "split_visualization_architecture": True,
            "session_management": True,
            "no_viz_timeouts": True,
        },
        "version": "4.2",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
