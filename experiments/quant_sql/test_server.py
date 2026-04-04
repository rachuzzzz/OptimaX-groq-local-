"""
test_server.py — Quant SQL Test Interface Backend
==================================================
Lightweight FastAPI server for manual testing.
Runs on port 8001 to avoid conflicting with OptimaX (port 8000).

Start:
    python experiments/quant_sql/test_server.py
Open:  http://localhost:8001

Endpoints:
    GET  /              → serves the test UI
    GET  /api/status    → model availability + DB connectivity
    GET  /api/queries   → 14 test queries
    POST /api/generate  → run Q4/Q8 inference or proxy to OptimaX
    POST /api/execute   → execute SQL against the database
"""

import asyncio
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

_HERE = pathlib.Path(__file__).parent

# ── Locate .env (sql-chat-backend/.env lives next to main.py) ───────────────
_root = _HERE.parent.parent            # OptimaX project root
_dotenv_candidates = [
    _root / "sql-chat-backend" / ".env",
    _root / ".env",
    pathlib.Path.cwd() / ".env",
]
try:
    from dotenv import load_dotenv
    for _c in _dotenv_candidates:
        if _c.exists():
            load_dotenv(_c)
            break
    else:
        load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.getenv("DATABASE_URL")

# ── Experiment modules ───────────────────────────────────────────────────────
sys.path.insert(0, str(_HERE))
from run_experiment import _run_inference, extract_sql, MODEL_Q4, MODEL_Q8, _check_paths
from test_queries import TEST_QUERIES
from schema_ddl import build_prompt

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Quant SQL Test Interface", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Serve UI ──────────────────────────────────────────────────────────────────
_UI_PATH = _HERE / "test_ui.html"


@app.get("/", response_class=HTMLResponse)
def index():
    if not _UI_PATH.exists():
        return HTMLResponse("<h1>test_ui.html not found</h1>", status_code=500)
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str   # "Q4_K_M" | "Q8_0" | "OPTIMAX"
    query: str

class ExecuteRequest(BaseModel):
    sql: str


# ── /api/status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    errors   = _check_paths()
    llama_ok = not any("llama-cli" in e for e in errors)
    q4_ok    = MODEL_Q4.exists() and llama_ok
    q8_ok    = MODEL_Q8.exists() and llama_ok

    # Lightweight OptimaX health check (GET /docs)
    optimax_ok = False
    try:
        req = urllib.request.Request("http://localhost:8000/docs", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            optimax_ok = r.status == 200
    except Exception:
        pass

    return {
        "q4_available":      q4_ok,
        "q8_available":      q8_ok,
        "llama_available":   llama_ok,
        "db_available":      bool(DATABASE_URL),
        "optimax_available": optimax_ok,
        "errors":            errors[:3],  # truncate long error list
    }


# ── /api/queries ──────────────────────────────────────────────────────────────

@app.get("/api/queries")
def get_queries():
    return [
        {"id": q["id"], "tier": q["tier"], "nl": q["nl"]}
        for q in TEST_QUERIES
    ]


# ── /api/generate ─────────────────────────────────────────────────────────────

def _generate_optimax(query: str) -> dict:
    """Proxy NL query to OptimaX /chat endpoint."""
    import uuid
    payload = json.dumps({
        "message":    query,
        "session_id": str(uuid.uuid4()),  # isolated session — no state bleed
        "include_sql": True,
        "row_limit":   50,
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"error": f"OptimaX not reachable at localhost:8000 — {exc}"}

    sql  = data.get("sql_query") or ""
    rows = data.get("query_results") or data.get("data") or []
    cols = list(rows[0].keys()) if rows else []

    return {
        "sql":           sql or None,
        "rows":          rows,
        "columns":       cols,
        "elapsed_s":     data.get("execution_time"),
        "clarification": data.get("clarification_needed", False),
        "clar_message":  data.get("response", "") if data.get("clarification_needed") else "",
        "optimax_msg":   data.get("response", ""),
        "error":         data.get("error"),
    }


def _generate_local(model_label: str, query: str) -> dict:
    """Run llama-cli Q4 or Q8 inference and return generated SQL."""
    model_path = MODEL_Q4 if model_label == "Q4_K_M" else MODEL_Q8
    if not model_path.exists():
        return {"error": f"Model file not found: {model_path}"}

    prompt = build_prompt(query)
    result = _run_inference(model_path, prompt, verbose=False, timeout=180)

    if result["error"]:
        return {"error": result["error"], "elapsed_s": result["elapsed_s"]}

    sql = extract_sql(result["raw_output"])
    return {
        "sql":        sql,
        "elapsed_s":  result["elapsed_s"],
        "raw_output": result["raw_output"],
        "error":      None if sql else "Could not extract SQL from model output",
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.query.strip():
        return JSONResponse({"error": "Query cannot be empty"}, status_code=400)

    if req.model == "OPTIMAX":
        result = await asyncio.to_thread(_generate_optimax, req.query)
    elif req.model in ("Q4_K_M", "Q8_0"):
        result = await asyncio.to_thread(_generate_local, req.model, req.query)
    else:
        return JSONResponse({"error": f"Unknown model: {req.model}"}, status_code=400)

    return result


# ── /api/execute ──────────────────────────────────────────────────────────────

def _execute_sql(sql: str) -> dict:
    if not DATABASE_URL:
        return {"error": "DATABASE_URL not configured. Check sql-chat-backend/.env"}
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 10})
        with engine.connect() as conn:
            r    = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in r.fetchmany(200)]
            cols = list(r.keys())
        return {"rows": rows, "columns": cols, "count": len(rows)}
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    if not req.sql.strip():
        return JSONResponse({"error": "SQL cannot be empty"}, status_code=400)
    result = await asyncio.to_thread(_execute_sql, req.sql)
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Quant SQL Test Interface")
    parser.add_argument("--port", type=int, default=8001, help="Port (default 8001)")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    path_errors = _check_paths()
    print(f"\n{'─'*50}")
    print(f"  Quant SQL Test Interface")
    print(f"{'─'*50}")
    print(f"  URL:     http://{args.host}:{args.port}")
    print(f"  Q4 model: {'OK' if MODEL_Q4.exists() else 'NOT FOUND'} ({MODEL_Q4.name})")
    print(f"  Q8 model: {'OK' if MODEL_Q8.exists() else 'NOT FOUND'} ({MODEL_Q8.name})")
    print(f"  Database: {'configured' if DATABASE_URL else 'NOT configured (set DATABASE_URL in .env)'}")
    print(f"  OptimaX:  http://localhost:8000 (start separately if needed)")
    print(f"{'─'*50}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
