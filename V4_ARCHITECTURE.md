# OptimaX v4.0 - Simplified Architecture

## Complete Rewrite - Clean Single-LLM Design

### Architecture Overview

**BEFORE (v3.0):**
- Dual-model architecture (Groq + Ollama)
- Complex agent_core.py with task management
- Separate SQL generation LLM
- Multiple backend files (agent_core, agent_tools, main_agentic)

**NOW (v4.0):**
- **Single Groq LLM** (llama-3.3-70b-versatile) for ALL tasks
- **LlamaIndex ReActAgent** for orchestration
- **2 simple tools**: SQL execution + Chart recommendation
- **2 backend files**: main.py + tools.py

---

## New File Structure

```
sql-chat-backend/
├── main.py              # FastAPI app + Agent setup (NEW - clean)
├── tools.py             # SQL + Chart tools (NEW - simple)
├── requirements.txt     # Updated dependencies (minimal)
├── .env                 # GROQ_API_KEY + DATABASE_URL
└── [OLD FILES DELETED]
    ✗ agent_core.py
    ✗ agent_tools.py
    ✗ main_agentic.py
    ✗ example_agentic_session.py
```

---

## What Changed

### 1. Single LLM for Everything
- **Groq llama-3.3-70b-versatile** handles:
  - Intent detection (chat vs data query)
  - SQL query generation
  - Tool calling decisions
  - Natural language responses
  - Chart recommendations

### 2. Simplified Tools

**Tool 1: `execute_sql(sql_query)`**
- Takes raw SQL query as input
- Validates for safety (read-only)
- Executes on PostgreSQL
- Returns JSON results

**Tool 2: `recommend_chart(data_json)`**
- Analyzes data structure
- Recommends chart type (bar, line, pie, table)
- Returns config for frontend

**Tool 3: `get_schema()`**
- Returns database schema
- Helps LLM understand available columns

### 3. No More Complexity
- ❌ No dual-model routing
- ❌ No complex task decomposition
- ❌ No separate SQL generation LLM
- ❌ No multi-step planning
- ✅ Direct tool calling by single LLM
- ✅ Session-based memory with ChatMemoryBuffer
- ✅ Clean FastAPI endpoints

---

## How It Works

### Query Flow

1. **User sends message** → `/chat` endpoint
2. **Session management** → Get or create agent with memory
3. **LlamaIndex ReActAgent** → Processes query
   - Classifies intent (greeting vs data query)
   - Generates SQL if needed
   - Calls `execute_sql` tool
   - Formats response naturally
4. **Response** → Natural language + data/SQL

### Example Interaction

**User:** "show me top 5 states by accidents"

**Agent reasoning:**
1. Detects data query intent
2. Generates SQL: `SELECT state, COUNT(*) as count FROM us_accidents GROUP BY state ORDER BY count DESC LIMIT 5`
3. Calls `execute_sql` tool
4. Receives results
5. Responds: "Here are the top 5 states by accident count: California leads with 1.74M accidents..."

---

## API Endpoints

### Core Endpoints
- `POST /chat` - Main chat interface
- `GET /health` - Health check
- `GET /sessions` - List active sessions
- `DELETE /sessions/{id}` - Clear session

### Info Endpoints
- `GET /models` - Model information
- `GET /table-info` - Database schema
- `GET /` - API info

---

## Installation & Setup

### 1. Install Dependencies
```bash
cd sql-chat-backend
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:pass@host:port/database
```

### 3. Run Server
```bash
python main.py
```

Server starts on `http://localhost:8000`

---

## Key Features

### ✅ What Works
- Natural language queries
- SQL generation and execution
- Chart recommendations
- Session-based conversation memory
- Multi-turn context awareness
- Intent classification (chat vs data)
- Safety validation (read-only queries)

### 🎯 Benefits
- **Simpler**: 2 files instead of 4
- **Faster**: Single LLM, no routing overhead
- **Cheaper**: Only Groq API calls (no local Ollama needed)
- **Cleaner**: No complex task management
- **Maintainable**: Easy to understand and modify

---

## Dependencies

```
# Core Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.8.0

# LlamaIndex + Groq
llama-index-core==0.12.0
llama-index-llms-groq==0.3.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.23

# Data Processing
pandas==2.1.4

# Utilities
python-dotenv==1.0.0
sqlparse==0.4.4
```

**Removed:**
- ❌ llama-index-llms-ollama (no local LLM needed)
- ❌ sentence-transformers (not needed)
- ❌ torch/transformers (not needed)

---

## System Prompt

The agent uses a comprehensive system prompt that:
- Defines OptimaX personality
- Classifies intents (greeting vs data query)
- Provides SQL query examples
- Lists all database columns
- Sets response style guidelines

See `SYSTEM_PROMPT` in `main.py` for full text.

---

## Chart Recommendation Logic

**ChartRecommender** analyzes:
- Number of columns
- Data types (numeric vs text)
- Number of rows
- Presence of time-based columns

**Recommendations:**
- 1 category + 1 value → **Bar chart**
- Small categories (≤10) → **Pie chart**
- Time series → **Line chart**
- Multiple metrics → **Grouped bar**
- Complex data → **Table**

---

## Frontend Compatibility

The new backend maintains compatibility with existing frontend:
- Same `/chat` endpoint structure
- Returns `data`, `sql_query` in response
- Adds `chart_recommendation` field
- Session management unchanged

**Frontend may need updates for:**
- Chart recommendation field
- Simplified response structure

---

## Testing

### Quick Test
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "show me top 5 states by accidents"}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Performance

**Expected response times:**
- Greetings: ~1-2 seconds (LLM response)
- Data queries: ~3-5 seconds (SQL generation + execution)
- Cached queries: N/A (no caching yet - can add later)

**Cost:**
- Only Groq API usage (~$0.40/1M tokens)
- No local compute needed

---

## Next Steps

### Potential Enhancements
1. Add result caching for repeated queries
2. Implement chart generation on backend
3. Add data export tools (CSV, Excel)
4. Enhanced error messages
5. Query history per session
6. Rate limiting

### Frontend Updates Needed
- Handle `chart_recommendation` field
- Display chart based on recommendation
- Update error handling if needed

---

## Migration from v3.0

**What to do:**
1. ✅ Backend rewritten - deploy new code
2. ✅ Old files deleted automatically
3. ⚠️ Frontend may need minor updates
4. ⚠️ Test all frontend features
5. ✅ No database changes needed

**Breaking changes:**
- Response structure simplified
- No `tasks` array in response
- Added `chart_recommendation` field
- Session agents now independent

---

## Summary

OptimaX v4.0 is a **complete rewrite** with:
- ✅ **Simpler architecture** (single LLM)
- ✅ **Cleaner codebase** (2 files)
- ✅ **Better maintainability**
- ✅ **Same functionality**
- ✅ **Lower complexity**

**Philosophy:** Do more with less. One powerful LLM can handle everything.
