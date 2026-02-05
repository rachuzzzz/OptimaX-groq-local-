<div align="center">
  <img src="LOGO.png" alt="OptimaX Logo" width="200"/>

  # OptimaX v6.1 - Production-Grade NL-SQL Platform

  **AI-Powered Natural Language to SQL with Environment Hardening & Docker Deployment**

  [![Angular](https://img.shields.io/badge/Angular-20.3-DD0031?logo=angular)](https://angular.io/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
  [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?logo=postgresql)](https://www.postgresql.org/)
  [![Groq](https://img.shields.io/badge/Groq-Cloud_LLM-FF6B6B)](https://groq.com/)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
  [![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12.0-purple)](https://www.llamaindex.ai/)

</div>

---

## What's New in v6.1

### Production-Grade Environment Hardening
- **Deterministic Virtual Environment** - `.venv` with fail-fast guards
- **VS Code Hard Lock** - Interpreter, Pylance, and debugger all use same Python
- **Startup Validation** - `env_guard.py` validates environment before any code runs
- **LlamaIndex Verification** - Build-time and runtime verification of NL-SQL components

### Docker Deployment
- **Multi-stage Production Build** - Minimal attack surface, non-root user
- **Build-time Verification** - Docker build fails if LlamaIndex is broken
- **docker-compose Ready** - One command deployment with health checks
- **Team Sharing** - Share via Docker Hub or image export

### New Verification Endpoints
- `GET /health` - Enhanced health check with LlamaIndex status
- `GET /verify/llamaindex` - Comprehensive NL-SQL verification

---

## About OptimaX

OptimaX v6.1 is an intelligent, **database-agnostic** SQL chat application that enables users to query any PostgreSQL database using natural language. Built with sophisticated intent routing, query governance, dynamic join path inference, and production-grade environment hardening.

### Key Features

- **Database Agnostic** - Works with ANY PostgreSQL database via dynamic schema detection
- **4-Gate Intent Routing** - Intelligent query classification for optimal performance
- **Query Governance Layer** - Rule-based complexity analysis prevents multi-query loops
- **Dynamic Join Path Inference (DJPI v3)** - Automatic relationship discovery for multi-table queries
- **Production Docker Support** - One command deployment with docker-compose
- **Environment Hardening** - Fail-fast guards eliminate Python interpreter ambiguity
- **LlamaIndex NL-SQL** - Native NLSQLTableQueryEngine for accurate SQL generation
- **Modern Glass UI** - Beautiful glass morphism design with smooth animations
- **Real-time Charts** - Automatic chart detection and visualization

---

## Quick Start

### Option A: Docker Deployment (Recommended)

The fastest way to run OptimaX. Requires only Docker and a Groq API key.

```bash
# 1. Clone the repository
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX

# 2. Configure environment
cp .env.docker.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Start all services
docker-compose up -d

# 4. Verify deployment
curl http://localhost:8000/health
curl http://localhost:8000/verify/llamaindex

# 5. Access the application
# Frontend: http://localhost:4200
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

For detailed Docker instructions, see **[DOCKER.md](./DOCKER.md)**.

### Option B: Local Development

#### Prerequisites

- **Node.js** 18+ and npm ([Download](https://nodejs.org/))
- **Python** 3.11+ ([Download](https://www.python.org/downloads/))
- **PostgreSQL** 14+ ([Download](https://www.postgresql.org/download/))
- **Groq API Key** ([Get free key](https://console.groq.com/keys))

#### Step 1: Clone Repository

```bash
git clone https://github.com/rachuzzzz/OptimaX-groq-local-.git
cd OptimaX
```

#### Step 2: Setup Backend

```bash
cd sql-chat-backend

# Create virtual environment (IMPORTANT: use .venv, not venv)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify LlamaIndex installation
python verify_llamaindex.py
```

#### Step 3: Configure Environment

Create `sql-chat-backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://postgres:password@localhost:5432/your_database
```

#### Step 4: Start Backend

```bash
python main.py
```

**Success Indicators:**
```
[env_guard] Environment validation passed
[env_guard] LlamaIndex NL-SQL verification: OK
Database connected: your_database_name
Schema loaded: X tables detected
Server running on http://localhost:8000
```

#### Step 5: Setup Frontend

```bash
# New terminal
cd sql-chat-app
npm install
ng serve
```

Frontend available at: `http://localhost:4200`

#### Step 6: Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# LlamaIndex verification
curl http://localhost:8000/verify/llamaindex
```

---

## Architecture (v6.1)

```
+-----------------------------------------------------------------------+
|                  OptimaX v6.1 - Production Architecture                |
+-----------------------------------------------------------------------+
|                                                                         |
|  ENVIRONMENT HARDENING                                                  |
|  +-------------------------------------------------------------------+ |
|  | .venv/              - Deterministic virtual environment           | |
|  | env_guard.py        - Fail-fast startup validation                | |
|  | verify_llamaindex.py - Comprehensive LlamaIndex verification      | |
|  | .vscode/settings.json - Hard-locked interpreter path              | |
|  +-------------------------------------------------------------------+ |
|                                                                         |
|  FRONTEND (Angular 20)                                                  |
|  +-------------------------------------------------------------------+ |
|  | Glass Morphism UI | Chart.js Visualization | Schema Browser       | |
|  +-------------------------------------------------------------------+ |
|                         | HTTP/REST                                     |
|                         v                                               |
|  BACKEND (FastAPI + LlamaIndex)                                         |
|  +-------------------------------------------------------------------+ |
|  |  4-GATE INTENT ROUTING                                            | |
|  |  +-------------------------------------------------------------+  | |
|  |  | Gate 1: Visualization Detection (cached result + LLM)       |  | |
|  |  | Gate 2: Greeting Fast-path (~200ms, no LLM)                 |  | |
|  |  | Gate 3: LLM Intent Classification                           |  | |
|  |  | Gate 4: Query Complexity Governance (rule-based)            |  | |
|  |  +-------------------------------------------------------------+  | |
|  |                                                                    | |
|  |  NL-SQL ENGINE (LlamaIndex)                                       | |
|  |  +-------------------------------------------------------------+  | |
|  |  | NLSQLTableQueryEngine - Native SQL generation               |  | |
|  |  | SQLDatabase - Schema introspection                          |  | |
|  |  | Groq LLM - llama-3.3-70b-versatile                          |  | |
|  |  +-------------------------------------------------------------+  | |
|  |                                                                    | |
|  |  SEMANTIC MEDIATION (v6.1)                                        | |
|  |  +-------------------------------------------------------------+  | |
|  |  | Intent Accumulator - Multi-turn context                     |  | |
|  |  | Semantic Role Resolver - Entity disambiguation              |  | |
|  |  | Relational Corrector - Join path validation                 |  | |
|  |  +-------------------------------------------------------------+  | |
|  |                                                                    | |
|  |  VERIFICATION ENDPOINTS                                           | |
|  |  +-------------------------------------------------------------+  | |
|  |  | /health - Database, LLM, schema status                      |  | |
|  |  | /verify/llamaindex - NL-SQL component verification          |  | |
|  |  +-------------------------------------------------------------+  | |
|  +-------------------------------------------------------------------+ |
|                         | SQL                                           |
|                         v                                               |
|  DATABASE (PostgreSQL)                                                  |
|  +-------------------------------------------------------------------+ |
|  | Dynamic schema detection | Auto-categorized columns               | |
|  +-------------------------------------------------------------------+ |
|                                                                         |
+-----------------------------------------------------------------------+
```

---

## Project Structure

```
OptimaX/
├── sql-chat-app/                          # Angular Frontend
│   ├── src/app/
│   │   ├── components/
│   │   │   ├── chat-interface/           # Main chat UI
│   │   │   ├── chart-visualization/      # Chart rendering
│   │   │   └── loading-screen/           # Loading animation
│   │   └── services/
│   │       └── chat.service.ts           # API communication
│   ├── Dockerfile                         # Production frontend image
│   └── package.json
│
├── sql-chat-backend/                      # FastAPI Backend
│   ├── main.py                            # FastAPI app + 4-gate routing
│   ├── tools.py                           # DatabaseManager + Tools
│   ├── join_path_inference.py             # DJPI v3 implementation
│   ├── env_guard.py                       # Fail-fast environment validation
│   ├── verify_llamaindex.py               # LlamaIndex verification script
│   ├── docker-verify.py                   # Build-time verification
│   ├── intent_accumulator.py              # Multi-turn intent tracking
│   ├── semantic_role_resolver.py          # Entity disambiguation
│   ├── relational_corrector.py            # Join path correction
│   ├── requirements.txt                   # Python dependencies
│   ├── Dockerfile                         # Production backend image
│   ├── .dockerignore                      # Docker build exclusions
│   └── .env                               # Environment config (not committed)
│
├── .vscode/                               # VS Code configuration
│   ├── settings.json                      # Hard-locked Python interpreter
│   └── launch.json                        # Debug configurations
│
├── docker-compose.yml                     # Multi-container orchestration
├── .env.docker.example                    # Docker environment template
├── DOCKER.md                              # Docker deployment guide
├── START-OPTIMAX.bat                      # Windows launcher
├── README.md                              # This file
└── LOGO.png                               # Application logo
```

---

## API Endpoints

### Health & Verification
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with database, LLM, and LlamaIndex status |
| `/verify/llamaindex` | GET | Comprehensive NL-SQL verification |
| `/` | GET | API info and feature list |
| `/docs` | GET | Interactive API documentation (Swagger) |

### Chat
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, get SQL response |

### Database Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/database/schema` | GET | Get current database schema |
| `/database/test-connection` | POST | Test database URL validity |
| `/database/connect` | POST | Switch to new database |

### Sessions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions` | GET | List active sessions |
| `/sessions/{id}` | DELETE | Delete session |

---

## Environment Hardening

### Why Environment Hardening?

Python projects can suffer from interpreter ambiguity:
- VS Code using different Python than the terminal
- Pylance analyzing code with different packages than runtime
- Debugger running with different interpreter than production

OptimaX v6.1 eliminates this with:

### 1. Deterministic Virtual Environment

```bash
# Always use .venv (not venv, not conda)
python -m venv .venv
```

### 2. VS Code Hard Lock

`.vscode/settings.json` forces VS Code to use the correct interpreter:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/sql-chat-backend/.venv/Scripts/python.exe"
}
```

### 3. Fail-Fast Startup Guard

`env_guard.py` validates at startup:
- Correct interpreter path (contains `.venv`)
- All required packages installed
- LlamaIndex NL-SQL components importable
- Environment variables present

### 4. LlamaIndex Verification

```bash
# Manual verification
python verify_llamaindex.py

# API verification
curl http://localhost:8000/verify/llamaindex
```

---

## Docker Deployment

### Quick Start

```bash
# Configure
cp .env.docker.example .env
# Edit .env with your GROQ_API_KEY

# Deploy
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

### Build-Time Verification

The Docker build includes fail-fast verification:

```
[1/3] Verifying imports...
  [OK] llama_index
  [OK] llama_index.core
  [OK] llama_index.core.query_engine
  [OK] llama_index.llms.groq

[2/3] Verifying NL-SQL classes...
  [OK] NLSQLTableQueryEngine
  [OK] SQLDatabase
  [OK] Settings
  [OK] Groq

[3/3] Verifying package versions...
  [OK] llama-index==0.12.0
```

If ANY check fails, the Docker build fails immediately.

### Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 4200 | Angular application |
| Backend | 8000 | FastAPI + LlamaIndex |
| PostgreSQL | 5432 | Database (optional) |

For detailed Docker instructions, see **[DOCKER.md](./DOCKER.md)**.

---

## Technology Stack

### Frontend
- **Framework:** Angular 20.3
- **Language:** TypeScript 5.9
- **Styling:** SCSS with Glass Morphism
- **Charts:** Chart.js 4.5.0 + ng2-charts

### Backend
- **Framework:** FastAPI 0.104.1
- **Language:** Python 3.11+
- **LLM Provider:** Groq (llama-3.3-70b-versatile)
- **NL-SQL Engine:** LlamaIndex 0.12.0 (NLSQLTableQueryEngine)
- **Database:** PostgreSQL 14+ (SQLAlchemy 2.0)

### DevOps
- **Containerization:** Docker + docker-compose
- **Environment:** Hardened .venv with fail-fast guards
- **IDE:** VS Code with hard-locked interpreter

---

## Troubleshooting

### LlamaIndex Verification Failed

```bash
# Check installation
python verify_llamaindex.py

# Reinstall if needed
pip uninstall llama-index llama-index-core llama-index-llms-groq -y
pip install llama-index==0.12.0 llama-index-llms-groq==0.3.0
```

### Wrong Python Interpreter

```bash
# Verify .venv is active
which python  # Should show .venv path

# Recreate if needed
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Docker Build Fails

```bash
# Clean rebuild
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
docker-compose up -d
```

### Backend Can't Connect to Database (Docker)

```bash
# Check postgres is running
docker-compose ps postgres

# Verify DATABASE_URL uses service name 'postgres', not 'localhost'
# Correct: postgresql://postgres:postgres@postgres:5432/optimax_db
# Wrong:   postgresql://postgres:postgres@localhost:5432/optimax_db
```

---

## Version History

| Version | Key Features |
|---------|--------------|
| **v6.1** (Current) | Environment hardening, Docker deployment, LlamaIndex verification |
| v4.3 | Database-agnostic architecture, dynamic schema loading |
| v4.2 | 4-gate intent routing, query governance layer |
| v4.0 | Single-LLM architecture, LlamaIndex ReActAgent |

---

## Documentation

- **[DOCKER.md](./DOCKER.md)** - Complete Docker deployment guide
- **[V4_ARCHITECTURE.md](./V4_ARCHITECTURE.md)** - Core architecture documentation
- **[QUERY_GOVERNANCE.md](./QUERY_GOVERNANCE.md)** - 4-gate routing and governance
- **[DYNAMIC_SCHEMA_FEATURE.md](./DYNAMIC_SCHEMA_FEATURE.md)** - Database-agnostic features

---

## Support

### Verification Commands

```bash
# Backend health
curl http://localhost:8000/health

# LlamaIndex verification
curl http://localhost:8000/verify/llamaindex

# Docker service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### Common Questions

**Q: Can OptimaX work with my custom database?**
A: Yes! OptimaX is database-agnostic. Set DATABASE_URL to any PostgreSQL database.

**Q: How do I verify LlamaIndex is working?**
A: Run `curl http://localhost:8000/verify/llamaindex` or `python verify_llamaindex.py`.

**Q: Why use .venv instead of venv?**
A: `.venv` is the VS Code convention and ensures consistent interpreter detection.

**Q: Can I run without Docker?**
A: Yes! Follow the "Local Development" quick start guide.

---

<div align="center">

**OptimaX v6.1**
**Status:** Production Ready
**Architecture:** NL-SQL + 4-Gate Routing + Environment Hardening + Docker

Built with Angular, FastAPI, LlamaIndex, and Groq

[Back to Top](#optimax-v61---production-grade-nl-sql-platform)

</div>
