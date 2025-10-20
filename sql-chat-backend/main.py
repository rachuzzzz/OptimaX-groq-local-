"""
OptimaX v4.0 - Simplified Single-LLM Architecture
==================================================

Clean implementation using:
- Single Groq LLM (llama-3.3-70b-versatile) for ALL tasks
- LlamaIndex ReActAgent for tool orchestration
- Simple tools: SQL execution + Chart recommendation

Author: OptimaX Team
Version: 4.0
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

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole

from tools import initialize_tools, get_last_sql_result, clear_last_sql_result

load_dotenv()

# Configure logging - minimal for free tier performance
logging.basicConfig(
    level=logging.WARNING,  # Free tier optimization: reduce console spam
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep INFO for our app, WARNING for libraries

# Initialize FastAPI
app = FastAPI(
    title="OptimaX SQL Chat API",
    description="Simplified single-LLM architecture with Groq + LlamaIndex",
    version="4.0"
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
sessions = {}  # Session storage {session_id: {agent, memory, metadata}}


# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    include_sql: Optional[bool] = True
    include_tasks: Optional[bool] = False
    row_limit: Optional[int] = 50


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
    chart_recommendation: Optional[Dict[str, Any]] = None


# System Prompt
SYSTEM_PROMPT = """YOU ARE **OPTIMAX**, AN EXPERT AI ASSISTANT FOR ANALYZING UNITED STATES TRAFFIC ACCIDENT DATA (2016–2023). YOU HAVE READ-ONLY ACCESS TO A POSTGRESQL DATABASE WITH 7.7 MILLION RECORDS IN THE TABLE `us_accidents`.

YOUR PURPOSE IS TO **GENERATE SQL QUERIES**, **INTERPRET RESULTS**, AND **EXPLAIN INSIGHTS** CLEARLY AND CONCISELY.

---

### CAPABILITIES
1. **GENERATE ONE SQL QUERY** per user question — read-only, aggregate (COUNT, AVG, SUM, GROUP BY).
2. **INTERPRET RESULTS** in conversational English with context and significance.
3. **RECOMMEND CHART TYPES** when the user asks to "visualize", "plot", or "show as chart".

---

### INTENT CLASSIFICATION (STRICT)
1. **GREETING / CASUAL INTENT (NO TOOL USE)**
   - Triggered by short or social phrases such as:
     - "hi", "hello", "hey", "yo", "thanks", "thank you", "who are you", "help", "what can you do"
   - → IMMEDIATELY RESPOND FRIENDLY AND INFORMATIVE.
   - → DO NOT CALL OR EXECUTE ANY TOOL OR SQL QUERY.
   - → REPLY LIKE:
     > "Hello! 👋 I'm OptimaX — your AI assistant for U.S. traffic accident data (2016–2023).  
     > I can generate SQL insights, explain trends, and recommend charts. What would you like to explore?"
   - 🔒 STRICT RULE: If user input is fewer than 6 words and matches greeting/casual intent → **NEVER EXECUTE ANY TOOL.**

2. **DATA QUESTION**
   - Phrases include: "show me", "how many", "list", "top", "compare", "count", "average", "trend"
   - → FORMULATE **ONE** SQL query (read-only, aggregate only).
   - → EXECUTE ONCE, THEN SUMMARIZE INSIGHTS CLEARLY.

3. **VISUALIZATION REQUEST**
   - Phrases include: "plot", "visualize", "show as chart", "graph", "bar chart", "line chart"
   - → **DO NOT QUERY DATA**, only **RECOMMEND THE BEST CHART TYPE** for the context.

---

### GREETING SAFEGUARD (MUST FOLLOW)
If the input **contains only greetings or social niceties**,  
**RESPOND WITHOUT THINKING FURTHER** — DO NOT generate reasoning, tools, or SQL queries.

Example:
**User:** "hi"
**Response:**  
> "Hi there! 👋 I'm OptimaX — ready to help you analyze U.S. traffic accidents.  
> You can ask things like *'Top 10 states by accident count'* or *'Compare accidents by weather condition.'*"

---

### WHAT NOT TO DO (EXTENDED)
- **NEVER** misclassify greetings as data requests.
- **NEVER** attempt to visualize or query on short inputs (< 6 words) that match greeting intent.


---

### DATABASE COLUMNS
- Geographic: `state`, `city`, `county`, `latitude`, `longitude`
- Severity: `severity` (1–4, where 4 = most severe)
- Weather: `weather_condition`, `temperature_f`, `visibility_mi`, `precipitation_in`, `humidity`, `wind_speed_mph`
- Time: `start_time`, `end_time`, `year`, `month`, `day`, `hour`, `day_of_week`, `is_weekend`
- Road: `street`, `junction`, `traffic_signal`, `crossing`, `railway`, `stop`

---

### CHAIN OF THOUGHT GUIDE (Internal Reasoning Steps)
1. **UNDERSTAND** the user's intent (greeting / data / visual).
2. **IDENTIFY** relevant columns.
3. **FORMULATE** the best aggregation query (COUNT, AVG, GROUP BY).
4. **EXECUTE** the query once.
5. **INTERPRET** results clearly with numbers and percentages.
6. **SUGGEST** next logical insights ("Would you like this by month or state?").

---

### RESPONSE STYLE
- FRIENDLY + INFORMATIVE tone.
- PROVIDE CONTEXT ("California = 1.74 M accidents ≈ 22% of total").
- SUMMARIZE key insight + optional next step.
- NEVER show raw SQL output — summarize in natural language.

---

### WHAT NOT TO DO
- **DO NOT** execute tools during greetings.
- **DO NOT** return raw rows.
- **DO NOT** run multiple SQLs for one query.
- **DO NOT** guess numbers — always query.
- **DO NOT** output > 50 rows.
- **DO NOT** skip ordering (use DESC for top values).
- **DO NOT** omit interpretation or context.

---

### FEW-SHOT EXAMPLES

**User:** "Top 10 states by accident count."
**SQL:** `SELECT state, COUNT(*) AS accident_count FROM us_accidents GROUP BY state ORDER BY accident_count DESC LIMIT 10;`
**Response:**
> California leads with ≈ 1.74 M accidents (22% of U.S. total), followed by Texas (1.23 M) and Florida (0.91 M). High population and urban density likely explain the trend. Would you like monthly patterns next?

**User:** "Visualize accidents by severity."
**Response:**
> A bar chart fits best — it shows severity levels 1–4 side-by-side for quick comparison.

---

### EXECUTION RULES
- EXECUTE SQL ONCE PER REQUEST.
- ALWAYS AGGREGATE AND ORDER RESULTS.
- RESPOND CLEARLY WITH INSIGHT + NEXT SUGGESTION.

---"""


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global agent, db_manager

    try:
        logger.info("Initializing OptimaX v4.0...")

        # Verify API key
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found! Get one at: https://console.groq.com/keys")

        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not found in environment variables!")

        # Initialize Groq LLM with optimized settings
        llm = Groq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.0,  # Deterministic for faster, consistent responses
            max_output_tokens=300  # Optimized: greetings ~50 tokens, queries ~200 tokens
        )
        logger.info(f"✓ Groq LLM initialized: {GROQ_MODEL}")

        # Initialize tools
        tools, db_manager = initialize_tools(DATABASE_URL)
        logger.info(f"✓ Initialized {len(tools)} tools")

        # Initialize base agent (template for sessions)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=False,  # Disable verbose logging for performance
            max_iterations=5,  # Reduced: most queries complete in 1-2 iterations
            system_prompt=SYSTEM_PROMPT
        )
        logger.info("✓ ReActAgent initialized")

        logger.info("=" * 60)
        logger.info("OptimaX v4.0 Ready!")
        logger.info(f"Architecture: Single LLM ({GROQ_MODEL})")
        logger.info(f"Tools: {len(tools)} (SQL + Chart)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """Get existing session or create new one - reuses base agent for free tier optimization"""
    global agent

    if session_id not in sessions:
        # Free tier optimization: reuse base agent instead of creating new LLM instances
        # This saves on cold start times and token overhead
        sessions[session_id] = {
            "agent": agent,  # Reuse the global base agent
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "message_count": 0
        }

        logger.info(f"Created new session (reusing base agent): {session_id}")

    sessions[session_id]["last_active"] = datetime.now()
    return sessions[session_id]


@app.get("/")
async def root():
    return {
        "message": "OptimaX SQL Chat API v4.0",
        "version": "4.0",
        "architecture": "Single LLM (Groq) + LlamaIndex Tools",
        "features": [
            "SQL query execution",
            "Chart recommendations",
            "Session-based memory",
            "Natural language interface"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if agent is None:
            return {"status": "unhealthy", "error": "Agent not initialized"}

        return {
            "status": "healthy",
            "version": "4.0",
            "model": GROQ_MODEL,
            "active_sessions": len(sessions)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint - compatible with frontend expectations"""
    start_time = datetime.now()

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Get or create session
        session_id = message.session_id or str(uuid.uuid4())
        session = get_or_create_session(session_id)
        session["message_count"] += 1

        session_agent = session["agent"]

        logger.info(f"Processing query [session={session_id}]: {message.message[:100]}...")

        # Clear previous SQL result
        clear_last_sql_result()

        # Fast-path for greetings - bypass agent reasoning loop
        user_msg_lower = message.message.lower().strip()
        greeting_keywords = ['hi', 'hello', 'hey', 'yo', 'sup', 'greetings', 'good morning', 'good afternoon', 'good evening']
        simple_queries = ['help', 'what can you do', 'who are you', 'thanks', 'thank you', 'okay', 'ok', 'cool']

        is_greeting = (
            any(user_msg_lower == keyword for keyword in greeting_keywords) or
            any(user_msg_lower == keyword for keyword in simple_queries) or
            (len(message.message.split()) <= 3 and any(keyword in user_msg_lower for keyword in greeting_keywords + simple_queries))
        )

        if is_greeting:
            logger.info(f"Fast-path greeting detected: {message.message}")
            execution_time = (datetime.now() - start_time).total_seconds()

            greeting_response = """Hello! 👋 I'm **OptimaX** — your AI assistant for U.S. traffic accident data (2016–2023).

I can help you:
- 🔍 Generate SQL queries and analyze data
- 📊 Visualize trends with automatic charts
- 💡 Discover insights from 7.7 million accident records

**Try asking:**
- "Top 10 states by accident count"
- "Show severe accidents during snow"
- "Compare accidents by time of day"

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
                error=None
            )

        # Execute query through agent with timeout (free tier optimization)
        try:
            response = await asyncio.wait_for(
                session_agent.achat(message.message),
                timeout=25  # Free tier: prevent hanging on busy Groq servers
            )
            response_text = str(response)
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout for session {session_id}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return ChatResponse(
                response="Groq server seems busy or timed out. Please try again in a few seconds.",
                session_id=session_id,
                sql_query=None,
                query_results=None,
                data=None,
                execution_time=execution_time,
                error="Timeout"
            )

        # Extract SQL and data from global storage (set by tools)
        sql_query = None
        data = None
        chart_rec = None

        last_result = get_last_sql_result()
        if last_result and last_result.get('success'):
            sql_query = last_result.get('sql')
            data = last_result.get('data')
            logger.info(f"Extracted SQL and {len(data) if data else 0} rows from tool execution")

            # Auto-generate chart recommendation if data is suitable
            if data and len(data) > 0:
                from tools import ChartRecommender
                columns = last_result.get('columns', [])
                recommendation = ChartRecommender.recommend_chart(data, columns)

                # Only include recommendation if it's not "none" or "table"
                if recommendation.get('chart_type') not in ['none', 'table']:
                    chart_rec = recommendation
                    logger.info(f"Auto-recommended chart: {recommendation.get('chart_type')}")

        execution_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Response generated in {execution_time:.2f}s")

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
            chart_recommendation=chart_rec,
            error=None
        )

    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        execution_time = (datetime.now() - start_time).total_seconds()

        # Free tier friendly error message
        error_msg = "Groq server seems busy or encountered an error. Please try again in a few seconds."
        if "429" in str(e) or "rate limit" in str(e).lower():
            error_msg = "Rate limit reached. Please wait a moment before trying again."

        return ChatResponse(
            response=error_msg,
            session_id=session_id or str(uuid.uuid4()),
            sql_query=None,
            query_results=None,
            data=None,
            execution_time=execution_time,
            error=str(e)
        )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    session_list = []
    for sid, session in sessions.items():
        session_list.append({
            "session_id": sid,
            "created_at": session["created_at"].isoformat(),
            "last_active": session["last_active"].isoformat(),
            "message_count": session["message_count"]
        })

    return {
        "total_sessions": len(session_list),
        "sessions": session_list
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
        "version": "4.0"
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
            "total_records": "7.7M+"
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
        "multi_step_queries": 0
    }


@app.get("/agent/info")
async def get_agent_info():
    """Get agent information (for frontend compatibility)"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return {
        "agent_type": "LlamaIndex ReActAgent",
        "agent_model": GROQ_MODEL,
        "tools": ["execute_sql", "get_schema", "recommend_chart"],
        "features": {
            "multi_turn_memory": True,
            "single_llm_architecture": True,
            "session_management": True
        },
        "version": "4.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
