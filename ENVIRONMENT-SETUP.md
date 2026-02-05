# OptimaX Environment Hardening Guide

This document explains how the Python environment is configured to eliminate all ambiguity between VS Code, the debugger, and runtime.

## Quick Start

### Windows (Recommended)

```powershell
# 1. Create virtual environment
cd sql-chat-backend
powershell -ExecutionPolicy Bypass -File setup-venv.ps1

# 2. Verify setup
cd ..
CHECK-SETUP.bat

# 3. Run the application
START-OPTIMAX.bat
```

### macOS/Linux

```bash
# 1. Create virtual environment
cd sql-chat-backend
chmod +x setup-venv.sh
./setup-venv.sh

# 2. Activate and run
source .venv/bin/activate
python -m uvicorn main:app --reload
```

---

## Architecture Overview

```
OptimaX/
├── .vscode/
│   ├── settings.json    # Forces VS Code to use .venv interpreter
│   ├── launch.json      # Debug configs use same interpreter
│   └── extensions.json  # Recommended extensions
├── sql-chat-backend/
│   ├── .venv/           # ← Single source of truth for Python
│   ├── env_guard.py     # Fail-fast startup validation
│   ├── main.py          # Imports env_guard FIRST
│   ├── requirements.txt # Locked dependencies
│   ├── pyrightconfig.json # Pylance configuration
│   ├── setup-venv.ps1   # Windows setup script
│   ├── setup-venv.sh    # Unix setup script
│   └── migrate-venv.bat # Migration from venv/ to .venv/
├── START-OPTIMAX.bat    # Environment-aware launcher
└── CHECK-SETUP.bat      # Comprehensive verification
```

---

## Deliverables

### 1. Virtual Environment (`.venv/`)

**Location:** `sql-chat-backend/.venv/`

**Why `.venv` instead of `venv`?**
- VS Code's Python extension uses `.venv` as the default discovery name
- Dotfile convention indicates "local/private" resources
- Consistent with modern Python tooling (PDM, Poetry)

**Creation Commands:**

```powershell
# Windows (PowerShell)
cd sql-chat-backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Windows (CMD)
cd sql-chat-backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

# macOS/Linux
cd sql-chat-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Dependency Locking (`requirements.txt`)

All dependencies are pinned to exact versions:

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.104.1 | REST API framework |
| `uvicorn` | 0.24.0 | ASGI server |
| `pydantic` | 2.8.0 | Data validation |
| `llama-index` | 0.12.0 | Meta-package (version sync) |
| `llama-index-core` | 0.12.0 | Query engines, agents |
| `llama-index-llms-groq` | 0.3.0 | Groq LLM integration |
| `sqlalchemy` | 2.0.23 | Database ORM |
| `psycopg2-binary` | 2.9.9 | PostgreSQL driver |
| `pandas` | 2.1.4 | Data processing |
| `python-dotenv` | 1.0.0 | Environment config |
| `sqlparse` | 0.4.4 | SQL parsing |
| `requests` | 2.31.0 | HTTP client |
| `aiohttp` | 3.9.1 | Async HTTP |

---

### 3. VS Code Hard Lock

**`.vscode/settings.json`** forces:
- `python.defaultInterpreterPath` → `.venv/Scripts/python.exe`
- Pylance uses the same interpreter
- Terminal auto-activates the venv
- Files in `.venv/` excluded from search

**`.vscode/launch.json`** ensures:
- Debug configurations specify the exact Python path
- Environment variables loaded from `.env`
- Working directory set to `sql-chat-backend/`

**After opening VS Code:**
1. Python extension should auto-detect `.venv`
2. Check bottom-left status bar shows: `Python 3.x.x ('.venv': venv)`
3. If not, run: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv`

---

### 4. Startup Fail-Fast Guard

**`sql-chat-backend/env_guard.py`** validates:

1. **Interpreter Check** - Running from `.venv`, not global Python
2. **Package Imports** - All required packages importable
3. **LlamaIndex NL-SQL** - `NLSQLTableQueryEngine` available
4. **Environment Variables** - `GROQ_API_KEY`, `DATABASE_URL` set
5. **Multiple Interpreter Warning** - Warns if PATH has multiple Pythons

**Integration in `main.py`:**
```python
# FIRST IMPORT - before everything else
from env_guard import validate_environment
validate_environment(strict=False)  # Warnings only
# Or: validate_environment(strict=True)  # Hard fail
```

**Standalone Testing:**
```bash
cd sql-chat-backend
.venv\Scripts\python env_guard.py --strict
```

---

### 5. Health Check Endpoint

**`GET /health`** now returns comprehensive status:

```json
{
  "status": "healthy",
  "version": "6.1",
  "checks": {
    "nl_sql_engine": {"status": "ok", "type": "NLSQLTableQueryEngine"},
    "database": {"status": "ok", "tables": 15, "schema": "postgres_air"},
    "foreign_keys": {"status": "ok", "foreign_keys_loaded": 23},
    "python_environment": {
      "status": "ok",
      "interpreter": "C:\\...\\sql-chat-backend\\.venv\\Scripts\\python.exe",
      "version": "3.11.5",
      "in_venv": true
    },
    "llama_index": {"status": "ok", "version": "0.12.0"}
  },
  "layers": {
    "semantic_intent": "enabled",
    "intent_accumulation": "enabled",
    "relational_correction": "enabled"
  }
}
```

---

### 6. Cleanup & Migration

**Migrating from `venv/` to `.venv/`:**

```batch
cd sql-chat-backend
migrate-venv.bat
```

This renames `venv/` to `.venv/` for VS Code alignment.

**Detecting Wrong Interpreter:**

The environment guard warns if:
- Running outside a virtual environment
- Packages imported from outside `.venv/`
- Multiple Python interpreters in PATH

---

## Troubleshooting

### Yellow Underlines in VS Code (Unresolved Imports)

1. Open Command Palette: `Ctrl+Shift+P`
2. Run: "Python: Select Interpreter"
3. Choose: `.venv` from `sql-chat-backend`
4. Reload window: `Ctrl+Shift+P` → "Developer: Reload Window"

### Debugger Uses Wrong Python

1. Open `.vscode/launch.json`
2. Verify `"python"` path points to `.venv/Scripts/python.exe`
3. Delete any cached `.pyc` files

### Terminal Not Activating venv

```powershell
# PowerShell
cd sql-chat-backend
. .\.venv\Scripts\Activate.ps1

# CMD
cd sql-chat-backend
.venv\Scripts\activate.bat
```

### "Module not found" at Runtime

```bash
# Verify you're in the right environment
python -c "import sys; print(sys.executable)"
# Should show: ...\sql-chat-backend\.venv\Scripts\python.exe

# Reinstall packages
pip install -r requirements.txt
```

### Environment Guard Fails

```bash
# Run with verbose output
python env_guard.py --report

# Check what's wrong
python env_guard.py --strict
```

---

## Tradeoffs

| Decision | Rationale |
|----------|-----------|
| `.venv` inside `sql-chat-backend/` | Keeps venv close to requirements.txt; VS Code prefers this |
| `strict=False` by default | Allows startup with warnings for demos; set `strict=True` for production |
| No `pyproject.toml` | Project uses pip+requirements.txt pattern; migration to Poetry/PDM optional |
| Windows-first scripts | Primary development platform; Unix scripts provided for cross-platform |

---

## Verification Checklist

Run these to confirm everything works:

```batch
REM 1. Environment check
CHECK-SETUP.bat

REM 2. Health check (after backend starts)
curl http://localhost:8000/health

REM 3. Manual interpreter verification
sql-chat-backend\.venv\Scripts\python -c "import llama_index; print(llama_index.__file__)"
```

All three should pass without errors.
