<div align="center">
  <img src="LOGO.png" alt="OptimaX Logo" width="200"/>

  # OptimaX v4.3 - Database-Agnostic SQL Chat

  **AI-Powered Natural Language to SQL with Intelligent Query Governance**

  [![Angular](https://img.shields.io/badge/Angular-20.3-DD0031?logo=angular)](https://angular.io/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?logo=postgresql)](https://www.postgresql.org/)
  [![Groq](https://img.shields.io/badge/Groq-Cloud_LLM-FF6B6B)](https://groq.com/)

</div>

---

## 📖 About OptimaX

OptimaX v4.3 is an intelligent, **database-agnostic** SQL chat application that enables users to query any PostgreSQL database using natural language. Built with sophisticated intent routing, query governance, and dynamic join path inference for safe, intelligent database interactions.

### 🎯 Key Features

- **Database Agnostic** - Works with ANY PostgreSQL database via dynamic schema detection
- **4-Gate Intent Routing** - Intelligent query classification for optimal performance
- **Query Governance Layer** - Rule-based complexity analysis prevents multi-query loops
- **Dynamic Join Path Inference (DJPI v3)** - Automatic relationship discovery for multi-table queries
- **Single LLM Architecture** - Groq llama-3.3-70b for all tasks (chat, SQL, reasoning)
- **Natural Language Queries** - Ask questions in plain English, get SQL results
- **Modern Glass UI** - Beautiful glass morphism design with smooth animations
- **Real-time Charts** - Automatic chart detection and visualization
- **Session Memory** - Multi-turn conversation context with per-session isolation
- **Live Schema Browsing** - View and test database connections dynamically

### 🚀 What's New in v4.3

#### Database-Agnostic Architecture
- **Auto-detect any PostgreSQL database** - No hardcoded schemas
- **Dynamic schema loading** - Tables and columns discovered at startup
- **Live connection testing** - Switch databases without restart
- **Frontend schema viewer** - Browse database structure interactively

#### 4-Gate Intent Routing System (v4.2)
1. **Gate 1: Visualization Detection** - Fast-path for chart requests using cached results
2. **Gate 2: Greeting Intent** - 200ms hardcoded responses for greetings
3. **Gate 3: LLM Intent Classification** - Autonomous reasoning for query categorization
4. **Gate 4: Query Governance** - Rule-based complexity analysis

#### Query Governance Layer (v4.2)
- **Analytical Query Detection** - Identifies multi-objective queries
- **Staged Execution Guidance** - Prevents agent loops with clear next steps
- **5 Signal Categories** - Ranking, classification, time windows, behavioral, flagging
- **Predictable Behavior** - Mimics enterprise BI tool UX

---

## 🏗️ Architecture (v4.3)

```
┌─────────────────────────────────────────────────────────────────┐
│            OptimaX 4.3 - Database-Agnostic Architecture         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Frontend (Angular 20)                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Glass Morphism UI                                       │  │
│  │ • Chart.js Visualization                                  │  │
│  │ • Real-time Chat Interface                                │  │
│  │ • Database Schema Browser                                 │  │
│  │ • Connection Settings Modal                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ↓ HTTP/REST                            │
│                                                                   │
│  Backend (FastAPI + LlamaIndex)                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4-GATE INTENT ROUTING                                    │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Gate 1: Visualization Detection (one-shot LLM)     │  │  │
│  │  │ Gate 2: Greeting Fast-path (~200ms)                │  │  │
│  │  │ Gate 3: LLM Intent Classification                   │  │  │
│  │  │ Gate 4: Query Complexity Governance (rule-based)    │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                      ↓                                     │  │
│  │  DYNAMIC SCHEMA INJECTION                                 │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ • Auto-detect database schema                       │  │  │
│  │  │ • Categorize columns (keys, time, text, numeric)    │  │  │
│  │  │ • Inject into LLM system prompt                     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                      ↓                                     │  │
│  │  DJPI v3 (Dynamic Join Path Inference)                    │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ • Relationship discovery (name matching, FK patterns) │ │
│  │  │ • Acyclic path enforcement (no cycles)              │  │  │
│  │  │ • Max 4-hop depth limit                             │  │  │
│  │  │ • Cost-aware scoring (semantic strength)            │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                      ↓                                     │  │
│  │  ReActAgent (Groq llama-3.3-70b)                          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ Tools:                                              │  │  │
│  │  │   • execute_sql - Safe query execution              │  │  │
│  │  │   • get_schema - Schema introspection               │  │  │
│  │  │   • classify_visualization - Chart recommendations  │  │  │
│  │  │                                                      │  │  │
│  │  │ • Multi-turn memory (per-session)                   │  │  │
│  │  │ • Max 5 iterations, 35s timeout                     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│                           ↓ SQL                                  │
│                                                                   │
│  Database (Any PostgreSQL DB)                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Dynamic schema detection                                │  │
│  │ • Auto-categorized columns                                │  │
│  │ • Live connection support                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Query Processing Flow

```
User Query → Gate 1 (Viz?) → Gate 2 (Greeting?) → Gate 3 (Intent)
    → Gate 4 (Complexity) → Schema Injection → DJPI (if multi-table)
    → ReActAgent → SQL Execution → Response
```

**Response Times:**
- Greetings: ~200ms (Gate 2 fast-path)
- Visualizations: ~1-2s (Gate 1 one-shot)
- Simple queries: 3-5s (direct SQL agent)
- Complex queries: 5-15s (DJPI + agent reasoning)

---

## 🚀 Quick Start Guide

### Prerequisites

- **Node.js** 18+ and npm ([Download](https://nodejs.org/))
- **Python** 3.9+ ([Download](https://www.python.org/downloads/))
- **PostgreSQL** 14+ ([Download](https://www.postgresql.org/download/))
- **Groq API Key** ([Get free key](https://console.groq.com/keys))
- **Git** ([Download](https://git-scm.com/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX
```

### Step 2: Setup Database

OptimaX v4.3 works with **any PostgreSQL database**. You have two options:

#### Option A: Use Your Own Database (Recommended)

Simply configure your existing PostgreSQL database URL in Step 3.3.

#### Option B: Use Sample Dataset (US Traffic Accidents)

##### 2.1 Create PostgreSQL Database

```bash
# Open PostgreSQL command line
psql -U postgres

# Create database
CREATE DATABASE traffic_db;

# Exit psql
\q
```

##### 2.2 Load Database Schema

```bash
# Load the schema
psql -U postgres -d traffic_db -f create_accidents_table.sql
```

##### 2.3 Import Dataset

1. Download the US Accidents dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Extract the CSV file
3. Import data into PostgreSQL:

```bash
# Using psql COPY command
psql -U postgres -d traffic_db

# Inside psql:
\COPY us_accidents FROM 'path/to/US_Accidents.csv' DELIMITER ',' CSV HEADER;
```

**Note:** Import may take 30-60 minutes depending on your system.

### Step 3: Setup Backend

#### 3.1 Navigate to Backend Directory

```bash
cd sql-chat-backend
```

#### 3.2 Install Python Dependencies

```bash
# Using a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3.3 Configure Environment Variables

Create a `.env` file in the `sql-chat-backend` directory:

```env
# .env file
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/your_database_name
```

**Get your Groq API key at:** https://console.groq.com/keys

**Database URL Format:**
```
postgresql://[username]:[password]@[host]:[port]/[database_name]

Examples:
postgresql://postgres:mypassword@localhost:5432/traffic_db
postgresql://user:pass@db.example.com:5432/myapp_production
```

#### 3.4 Start Backend Server

```bash
python main.py
```

**Success Indicators:**
```
✓ Database connected: your_database_name
✓ Schema loaded: X tables detected
✓ Agent initialized with dynamic schema
✓ Server running on http://localhost:8000
```

Backend will be available at: `http://localhost:8000`

### Step 4: Setup Frontend

#### 4.1 Open New Terminal and Navigate to Frontend

```bash
cd sql-chat-app
```

#### 4.2 Install Node Dependencies

```bash
npm install
```

#### 4.3 Start Development Server

```bash
ng serve
# Or use npm
npm start
```

Frontend will be available at: `http://localhost:4200`

### Step 5: Access the Application

1. Open your browser and navigate to `http://localhost:4200`
2. Click **"Table Schema"** in sidebar to view detected tables
3. Start chatting with your database!

---

## 💻 Usage Examples

### Database-Agnostic Queries

OptimaX automatically understands your database schema. No configuration needed!

#### Schema Exploration
```
"What tables are available?"
"Show me the schema"
"What columns does the users table have?"
```

#### Simple Queries
```
"Show me the top 10 records from [your_table]"
"Count total rows in [your_table]"
"What are the unique values in [column_name]?"
```

#### Multi-table Queries (DJPI v3)
```
"Find all orders with customer information"
"Show products with their categories"
"List flights with passenger details"
```

**DJPI automatically discovers join paths:**
- No manual relationship configuration
- Acyclic path enforcement (prevents cycles)
- Max 4-hop depth limit (prevents timeouts)
- Semantic scoring for best path selection

### Traffic Accidents Dataset Queries (if using sample data)

#### Geographic Analysis
```
"Show me the top 10 states with the most accidents"
"Which city has the most accidents?"
"Find accidents in California"
```

#### Weather Analysis
```
"Show accidents during rain"
"Which weather conditions cause the most accidents?"
```

#### Severity Analysis
```
"Count accidents by severity level"
"Show me severe accidents in New York"
```

#### Temporal Analysis
```
"Show accidents by month in 2021"
"What time of day has the most accidents?"
```

### Visualization Examples

OptimaX detects chart opportunities automatically:

```
User: "Show top 10 states"
OptimaX: [Returns data + suggests bar chart]

User: "Show me a chart"
OptimaX: [Generates chart from last query]
```

### Query Governance Examples

**Simple Query (Allowed):**
```
User: "Show top 10 routes"
→ Gate 4: ranking only → SIMPLE ✅
→ Executes SQL directly
```

**Analytical Query (Governed):**
```
User: "Find top 20 VIP customers in last quarter, flag inactive ones"
→ Gate 4: ranking + classification + time_windows + flagging → ANALYTICAL ⚠️
→ Returns staged execution guidance (NO SQL execution)
→ Suggests breaking into steps
```

---

## 🛠️ Technology Stack

### Frontend
- **Framework:** Angular 20.3.2
- **Language:** TypeScript 5.9
- **Styling:** SCSS with Glass Morphism
- **Charts:** Chart.js 4.5.0 + ng2-charts 8.0.0

### Backend
- **Framework:** FastAPI 0.104.1
- **Language:** Python 3.9+
- **LLM:** Groq llama-3.3-70b-versatile
- **Agent Framework:** LlamaIndex 0.12.0 (ReActAgent)
- **Database:** PostgreSQL 14+ (SQLAlchemy 2.0.23)
- **Data Processing:** pandas 2.1.4, sqlparse 0.4.4

### AI/ML
- **LLM Provider:** Groq Cloud
- **Model:** llama-3.3-70b-versatile (single model for all tasks)
- **Agent:** LlamaIndex ReActAgent with tool calling
- **Governance:** Rule-based complexity classifier (no ML/embeddings)

---

## 📁 Project Structure

```
OptimaX/
├── sql-chat-app/                          # Angular Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/
│   │   │   │   ├── chat-interface/       # Main chat UI + DB settings modal
│   │   │   │   ├── chart-visualization/  # Chart rendering
│   │   │   │   ├── loading-screen/       # Loading animation
│   │   │   │   └── system-prompt-manager/ # Custom prompt UI
│   │   │   └── services/
│   │   │       └── chat.service.ts       # API communication
│   │   └── styles.scss                   # Global glass morphism styles
│   └── package.json
│
├── sql-chat-backend/                      # FastAPI Backend
│   ├── main.py                            # FastAPI app + 4-gate routing (900+ lines)
│   ├── tools.py                           # DatabaseManager + Tools (600+ lines)
│   ├── join_path_inference.py             # DJPI v3 implementation (500+ lines)
│   ├── requirements.txt                   # Python dependencies
│   └── .env                               # Environment config
│
├── DYNAMIC_SCHEMA_FEATURE.md              # v4.3 schema detection docs
├── QUERY_GOVERNANCE.md                    # v4.2 governance layer docs
├── V4_ARCHITECTURE.md                     # v4.0 baseline architecture
├── LOGO.png                               # Application logo
└── README.md                              # This file
```

---

## 🔧 Configuration

### Backend Configuration

Edit `sql-chat-backend/.env`:

```env
# Groq API (required)
GROQ_API_KEY=your_groq_api_key_here

# Database connection (required - works with ANY PostgreSQL DB)
DATABASE_URL=postgresql://user:password@host:port/database

# Examples:
# Local development:
DATABASE_URL=postgresql://postgres:mypassword@localhost:5432/myapp_db

# Production:
DATABASE_URL=postgresql://admin:securepass@db.production.com:5432/app_production

# Cloud (AWS RDS):
DATABASE_URL=postgresql://dbuser:pass@mydb.us-east-1.rds.amazonaws.com:5432/production
```

**OptimaX will automatically:**
- Detect all tables in your database
- Categorize columns by type (keys, time, text, numeric, other)
- Generate LLM-optimized schema descriptions
- Inject schema into agent system prompt

### Frontend Configuration

The frontend is pre-configured to connect to `http://localhost:8000` (backend).

To change, edit `sql-chat-app/src/app/services/chat.service.ts`:

```typescript
private apiUrl = 'http://localhost:8000';  // Update if needed
```

### Custom System Prompt (Optional)

Create `sql-chat-backend/custom_system_prompt.txt` (JSON format):

```json
{
  "prompt": "You are OptimaX, a helpful database assistant...\n{SCHEMA_SECTION}\n..."
}
```

**Note:** Use `{SCHEMA_SECTION}` placeholder for dynamic schema injection.

---

## 🧪 Testing

### Backend Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "llama-3.3-70b-versatile",
  "active_sessions": 0,
  "database_tables": 8
}
```

### Test Database Connection

```bash
curl -X POST http://localhost:8000/database/test-connection \
  -H "Content-Type: application/json" \
  -d '{"database_url": "postgresql://user:pass@localhost:5432/testdb"}'
```

### Test Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "what tables exist?"}'
```

### View Database Schema

```bash
curl http://localhost:8000/database/schema
```

---

## 📈 API Endpoints

### Chat Endpoints
- `POST /chat` - Send message, get response
  - Body: `{message, session_id?, system_prompt?, include_sql?, row_limit?}`
  - Returns: `ChatResponse` with SQL, data, chart suggestion, execution time

### Database Management (NEW in v4.3)
- `GET /database/schema` - Get current database schema summary
- `POST /database/test-connection` - Test database URL validity
- `POST /database/connect` - Switch to new database dynamically

### Health & Info
- `GET /health` - Health check + database table count
- `GET /` - API info and feature list
- `GET /models` - Model information
- `GET /table-info` - Database schema (legacy endpoint)

### Session Management
- `GET /sessions` - List active sessions
- `DELETE /sessions/{id}` - Delete session

---

## 🎯 Understanding the 4-Gate System

### Gate 1: Visualization Detection
- **Trigger:** Keywords like "chart", "graph", "visualization"
- **Condition:** Must have cached SQL result from previous query
- **Action:** One-shot LLM call for chart classification
- **Response Time:** ~1-2s

### Gate 2: Greeting Intent
- **Trigger:** Keywords like "hi", "hello", "thanks", "help"
- **Action:** Hardcoded fast-path response (no LLM)
- **Response Time:** ~200ms

### Gate 3: LLM Intent Classification
- **Trigger:** All non-greeting, non-viz queries
- **Action:** One LLM call for intent classification
- **Output:** `database_query` | `greeting` | `clarification_needed`
- **Response Time:** 2-3s

### Gate 4: Query Complexity Governance
- **Trigger:** Queries classified as `database_query`
- **Action:** Rule-based analysis (<1ms execution)
- **Classification:**
  - **Simple (0-1 signals):** Execute SQL via agent
  - **Analytical (2+ signals):** Return staged execution guidance
- **Signal Categories:** ranking, classification, time_windows, behavioral, flagging

**Example Flow:**
```
"hi" → Gate 2 → Greeting response ⚡ (~200ms)
"show chart" → Gate 1 → Chart suggestion 📊 (~1-2s)
"top 10 states" → Gate 3 → Gate 4 (SIMPLE) → SQL agent ✅ (3-5s)
"top VIP customers last quarter" → Gate 4 (ANALYTICAL) → Governed ⚠️ (2-3s)
```

---

## 🔒 Query Governance Layer

### Why Governance?

**Problem:** Complex analytical queries with multiple objectives can cause:
- Agent loops (trying different SQL approaches)
- Timeouts (query too complex)
- Unpredictable behavior (stochastic LLM decisions)

**Solution:** Detect analytical queries and guide staged execution.

### Signal Categories

OptimaX detects 5 independent signal categories:

1. **Ranking:** top, best, worst, highest, lowest, rank
2. **Classification:** VIP, frequent, inactive, segment, group
3. **Time Windows:** last, past, previous, recent, days, months
4. **Behavioral:** preferred, most common, average, typical, pattern
5. **Flagging:** identify, mark, flag, whether, detect

### Classification Rules

- **0-1 signals:** SIMPLE query → Execute via SQL agent
- **2+ signals:** ANALYTICAL query → Governance response (no SQL)

### Governance Response

For analytical queries, OptimaX returns:
- List of detected objectives
- Explanation of why staged execution is needed
- Suggested valid first step
- Actionable guidance

**User benefits:**
- Clear, predictable behavior
- Faster responses (no agent loops)
- Step-by-step analysis workflow
- Matches enterprise BI tool UX

---

## 🔗 Dynamic Join Path Inference (DJPI v3)

### What is DJPI?

DJPI v3 automatically discovers relationships between database tables to enable multi-table queries without manual configuration.

### How It Works

1. **Relationship Discovery**
   - Exact column name matches (e.g., `account_id` in both tables)
   - Foreign key naming patterns (`*_id`, `*_code`)
   - Type compatibility checks

2. **Join Scoring**
   - STRONG: `_id` columns + type match → 100+ points
   - MEDIUM: Exact name + compatible types → 40-80 points
   - WEAK: Timestamp/text columns → negative penalties

3. **Path Finding**
   - Modified Dijkstra algorithm (maximizes score)
   - Acyclic constraint (no table visited twice)
   - Max 4-hop depth limit (prevents deep joins)

4. **Agent Injection**
   - Prepends join guidance (not SQL)
   - LLM generates actual SQL
   - Prevents timeout from random join attempts

### Example

```
Query: "Find flights with passenger information"

DJPI Analysis:
→ Identifies tables: flight, passenger, booking
→ Discovers join path:
   flight.booking_id → booking.booking_id
   booking.passenger_id → passenger.passenger_id
→ Injects guidance: "Join through booking table"

Agent generates:
SELECT f.*, p.*
FROM flight f
JOIN booking b ON f.booking_id = b.booking_id
JOIN passenger p ON b.passenger_id = p.passenger_id
```

---

## 🐛 Troubleshooting

### Backend Won't Start

**Error:** `GROQ_API_KEY not found`

**Solution:**
```bash
# Add to .env file in sql-chat-backend directory
GROQ_API_KEY=gsk_your_key_here
```

### Database Connection Failed

**Error:** `could not connect to server` or `database does not exist`

**Solution:**
- Verify PostgreSQL is running: `pg_isready` or check services
- Check `.env` has correct credentials
- Test connection: `psql -U username -d database_name`
- Use `/database/test-connection` endpoint to validate URL

### No Tables Detected

**Error:** `No tables found in database`

**Solution:**
- Check database has tables: `\dt` in psql
- Verify database name is correct in DATABASE_URL
- Ensure user has SELECT permissions on tables
- System schemas (information_schema, pg_catalog) are automatically filtered

### Frontend Build Errors

**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### Agent Timeout

**Error:** `Agent execution exceeded 35 seconds`

**Solution:**
- Query may be too complex (DJPI max depth is 4 hops)
- Try breaking query into smaller steps
- Check if table relationships exist for joins
- Governance layer may have detected analytical query

---

## 📚 Documentation

Detailed documentation available:

- **[V4_ARCHITECTURE.md](./V4_ARCHITECTURE.md)** - v4.0 baseline architecture, single-LLM design
- **[QUERY_GOVERNANCE.md](./QUERY_GOVERNANCE.md)** - v4.2 governance layer, 4-gate routing
- **[DYNAMIC_SCHEMA_FEATURE.md](./DYNAMIC_SCHEMA_FEATURE.md)** - v4.3 database-agnostic features
- **DJPI v3** - Documented in `join_path_inference.py:1-60`

---

## 🚢 Deployment

### Production Build

#### Frontend
```bash
cd sql-chat-app
ng build --configuration production
# Output: dist/sql-chat-app/
```

#### Backend
```bash
cd sql-chat-backend

# Production server with workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Or with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables (Production)

```env
GROQ_API_KEY=your_production_key
DATABASE_URL=postgresql://user:pass@production-db:5432/app_db
```

### Docker Deployment (Optional)

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is for educational and demonstration purposes.

**Dataset License (if using sample data):** [US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

---

## 🙏 Acknowledgments

- **Groq** - Fast LLM inference platform
- **LlamaIndex** - Agent framework and tool orchestration
- **Angular Team** - Frontend framework
- **FastAPI** - Python web framework
- **Sobhan Moosavi** - US Accidents dataset creator

---

## 📞 Support

### Documentation
- [V4.3 Database-Agnostic Features](DYNAMIC_SCHEMA_FEATURE.md)
- [V4.2 Query Governance & 4-Gate Routing](QUERY_GOVERNANCE.md)
- [V4.0 Architecture Guide](V4_ARCHITECTURE.md)

### Need Help?
1. Check documentation files above
2. Review API endpoints at `http://localhost:8000/docs` (FastAPI auto-docs)
3. Test database connection via `/database/test-connection`
4. Check application logs for errors
5. Verify health check: `curl http://localhost:8000/health`

### Common Issues

**Q: Can OptimaX work with my custom database?**
A: Yes! OptimaX v4.3 is database-agnostic. Just set DATABASE_URL to any PostgreSQL database.

**Q: How do I see what tables OptimaX detected?**
A: Click "Table Schema" in the sidebar or call `GET /database/schema`.

**Q: Why did my query get "governed"?**
A: Your query has 2+ analytical signal categories. Break it into smaller steps for best results.

**Q: Can I switch databases without restarting?**
A: Yes! Use the "Connection Settings" modal in the frontend or `POST /database/connect` endpoint.

---

## 🎉 Success Indicators

You're ready when you see:
- ✅ Backend starts at port 8000
- ✅ Frontend loads at `http://localhost:4200`
- ✅ Health check returns `{"status": "healthy", "database_tables": X}`
- ✅ Database schema detected (check logs or `/database/schema`)
- ✅ Groq LLM initialized (`llama-3.3-70b-versatile`)
- ✅ Charts rendering correctly
- ✅ 4-gate routing active (check logs for gate decisions)

---

## 🗺️ Version History

- **v4.3** (Current) - Database-agnostic architecture, dynamic schema loading, live connection testing
- **v4.2** - 4-gate intent routing, query governance layer, rule-based complexity classifier
- **v4.0** - Single-LLM architecture, simplified codebase, LlamaIndex ReActAgent
- **v3.x** - Dual-LLM architecture (deprecated)

---

<div align="center">

**Version:** 4.3.0
**Status:** ✅ Production Ready
**Architecture:** Database-Agnostic + 4-Gate Routing + Query Governance + DJPI v3

**Built with ❤️ using Angular, FastAPI, LlamaIndex, and Groq**

[⬆ Back to Top](#optimax-v43---database-agnostic-sql-chat)

</div>
