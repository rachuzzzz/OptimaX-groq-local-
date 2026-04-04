"""
run_through_optimax.py — Production Pipeline Baseline
======================================================

Feeds the same 14 NL queries through the live OptimaX FastAPI endpoint
(default: http://localhost:8000) so the evaluator can three-way compare:

  Q4_K_M   (local llama.cpp)
  Q8_0     (local llama.cpp)
  OPTIMAX  (production Groq llama-3.3-70b via OptimaX governance pipeline)

Uses only stdlib urllib.request — no extra pip packages needed.

The output JSON format matches run_experiment.py so evaluate_results.py can
load both together.

Usage:
    python run_through_optimax.py                     # hit localhost:8000
    python run_through_optimax.py --url http://host:8000
    python run_through_optimax.py --query-id 9        # single query
    python run_through_optimax.py --session-id abc123 # fixed session
    python run_through_optimax.py --verbose
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from test_queries import TEST_QUERIES, TIER_LABELS

OUTPUTS = _HERE / "outputs"

DEFAULT_URL = "http://localhost:8000"
ENDPOINT    = "/chat"

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    """
    POST a JSON payload to url, return parsed JSON response.
    Raises urllib.error.URLError on network failure.
    """
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _check_server(base_url: str) -> bool:
    """
    Return True if the OptimaX server is reachable at base_url.
    Tries GET /docs (FastAPI auto-generated) as a health check.
    """
    try:
        req = urllib.request.Request(
            base_url + "/docs",
            headers={"Accept": "text/html"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# SQL extraction from OptimaX response
# ---------------------------------------------------------------------------

def _extract_sql_from_response(resp: dict) -> str | None:
    """
    Extract SQL from an OptimaX /chat response dict.

    OptimaX returns:
      resp["sql_query"]   — the SQL that went through governance (primary)
      resp["response"]    — natural language answer (may embed SQL)
    """
    # Primary: sql_query field is explicitly returned
    if resp.get("sql_query") and resp["sql_query"].strip():
        return resp["sql_query"].strip()

    # Fallback: parse SQL from the response text
    import re
    text = resp.get("response") or ""
    match = re.search(r"```sql\s*([\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"SELECT[\s\S]+?(?:LIMIT \d+|;|$)", text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return None


# ---------------------------------------------------------------------------
# Main query runner
# ---------------------------------------------------------------------------

def run_query(
    query: dict,
    base_url: str,
    session_id: str,
    verbose: bool = False,
    timeout: int = 60,
) -> dict:
    """
    Send one NL query to the OptimaX endpoint.  Returns a result record
    compatible with run_experiment.py format.
    """
    qid  = query["id"]
    tier = query["tier"]
    nl   = query["nl"]

    print(f"  [Q{qid:02d}][OPTIMAX] {nl[:60]}", end=" ... ", flush=True)

    payload = {
        "message":     nl,
        "session_id":  session_id,
        "include_sql": True,
        "row_limit":   50,
    }

    if verbose:
        print(f"\n  POST {base_url}{ENDPOINT}")
        print(f"  Payload: {json.dumps(payload)}")

    t0 = time.monotonic()
    error_msg     = None
    resp          = {}
    extracted_sql = None

    try:
        resp = _post_json(base_url + ENDPOINT, payload, timeout=timeout)
        elapsed = time.monotonic() - t0

        if verbose:
            print(f"\n  Response keys: {list(resp.keys())}")
            print(f"  sql_query: {str(resp.get('sql_query', ''))[:120]}")
            print(f"  response: {str(resp.get('response', ''))[:200]}")

        extracted_sql = _extract_sql_from_response(resp)

        if resp.get("error"):
            error_msg = resp["error"]

        if resp.get("clarification_needed"):
            # OptimaX asked for clarification — we log it but don't retry
            error_msg = f"OptimaX requested clarification: {resp.get('response', '')[:120]}"

        status = "ok" if extracted_sql else "extraction_failed"
        if resp.get("error"):
            status = "api_error"

    except urllib.error.URLError as exc:
        elapsed   = time.monotonic() - t0
        error_msg = f"Network error: {exc}"
        status    = "network_error"
    except Exception as exc:
        elapsed   = time.monotonic() - t0
        error_msg = f"Unexpected error: {exc}"
        status    = "error"

    print(f"{status} ({elapsed:.1f}s)")

    return {
        "query_id":         qid,
        "tier":             tier,
        "nl":               nl,
        "model":            "OPTIMAX",
        "model_path":       base_url,
        "prompt_len":       0,                # N/A for API calls
        "raw_output":       json.dumps(resp, default=str),
        "extracted_sql":    extracted_sql,
        "elapsed_s":        round(elapsed, 2),
        "returncode":       0,
        "status":           status,
        "extraction_error": error_msg,
        "gold_sql":         query["gold_sql"],
        "key_joins":        query["key_joins"],
        "notes":            query["notes"],
        # Extra context from OptimaX
        "optimax_response": resp.get("response", ""),
        "optimax_error":    resp.get("error"),
        "clarification":    resp.get("clarification_needed", False),
        "exec_time":        resp.get("execution_time"),
        "query_results":    resp.get("query_results"),
    }


def save_results(results: list, label: str = "optimax") -> pathlib.Path:
    """Save results to the outputs folder in the same format as run_experiment.py."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS / f"experiment_{label}_{ts}.json"

    payload = {
        "experiment": "quant_sql_optimax_baseline",
        "timestamp":  datetime.now().isoformat(),
        "source":     "OptimaX FastAPI /chat endpoint",
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run test queries through the OptimaX production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"OptimaX base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        metavar="N",
        help="Run only query #N (1-14)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        metavar="ID",
        help="Fixed session ID (default: auto-generated per query to avoid state bleed)",
    )
    parser.add_argument(
        "--shared-session",
        action="store_true",
        help="Use one shared session for all queries (preserves multi-turn context)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print request/response details",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    # ── Server health check ──────────────────────────────────────────────────
    print(f"\n[INFO] Checking OptimaX server at {args.url} ...")
    if not _check_server(args.url):
        print(
            f"[ERROR] Cannot reach {args.url}\n"
            f"  Start the server with:\n"
            f"    cd sql-chat-backend && uvicorn main:app --reload --port 8000\n"
            f"  Then re-run this script."
        )
        sys.exit(1)
    print(f"[OK]   Server is reachable")

    # ── Query selection ──────────────────────────────────────────────────────
    queries = TEST_QUERIES
    if args.query_id is not None:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"[ERROR] No query with id={args.query_id}. Valid: 1-14.")
            sys.exit(1)
        print(f"[INFO] Running single query: #{args.query_id}")
    else:
        print(f"[INFO] Running all {len(queries)} queries through OptimaX")
        for tier in sorted({q["tier"] for q in queries}):
            tier_qs = [q for q in queries if q["tier"] == tier]
            print(f"  Tier {tier} ({TIER_LABELS[tier]}): {len(tier_qs)} queries")

    # ── Run ──────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"SOURCE: OptimaX Pipeline ({args.url})")
    print(f"{'═'*60}")

    import uuid
    shared_sid = args.session_id or str(uuid.uuid4())

    results: list = []
    for q in queries:
        # Use a fresh session per query (avoids intent accumulator state bleed)
        # unless --shared-session was requested
        if args.shared_session:
            sid = shared_sid
        else:
            sid = str(uuid.uuid4())

        record = run_query(
            q,
            base_url=args.url,
            session_id=sid,
            verbose=args.verbose,
            timeout=args.timeout,
        )
        results.append(record)

    # ── Summary ──────────────────────────────────────────────────────────────
    ok  = sum(1 for r in results if r["status"] == "ok")
    tot = len(results)
    print(f"\n  Summary: {ok}/{tot} SQL captured successfully")

    # Count clarification requests
    clarified = sum(1 for r in results if r.get("clarification"))
    if clarified:
        print(f"  Note: {clarified} queries triggered OptimaX clarification flow")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = save_results(results)

    print(
        f"\nCombine with local experiment results for three-way comparison:\n"
        f"  python experiments/quant_sql/evaluate_results.py"
        f" --file outputs/experiment_combined_*.json"
        f" --file {out_path}"
    )


if __name__ == "__main__":
    main()
