# -*- coding: utf-8 -*-
"""
OptimaX Evaluation Runner - Final Year Project Review
Runs all 45 evaluation prompts and classifies results.
Usage: python eval_runner.py
"""
import requests
import json
import time
import uuid
import re as _re

BASE_URL = "http://localhost:8000"
CHAT_URL = f"{BASE_URL}/chat"

CATEGORIES = {
    "A": {
        "name": "Simple Retrieval",
        "queries": [
            "List all flights.",
            "Show all airports.",
            "Get passenger with passenger_id = 1001.",
            "Show all bookings from the last 7 days.",
            "List all frequent flyer members.",
            "Show flights departing from JFK.",
            "List bookings with amount greater than 500.",
            "Show passengers with last name Smith.",
            "List all aircraft types.",
            "Show flights scheduled for today.",
        ],
    },
    "B": {
        "name": "Join Queries",
        "queries": [
            "Show passenger names and their booking IDs.",
            "List flights with their departure and arrival airport names.",
            "Show bookings along with passenger full names.",
            "List flights with aircraft model information.",
            "Show passengers and their frequent flyer level.",
            "List routes with total number of flights.",
            "Show flights with departure and arrival city names.",
            "List bookings with flight numbers and passenger names.",
            "Show flights and the airport country for departure.",
            "List passengers who have made more than one booking.",
        ],
    },
    "C": {
        "name": "Aggregation Queries",
        "queries": [
            "Count total number of bookings.",
            "What is the average booking amount?",
            "Show total revenue generated from bookings.",
            "Count total number of flights per airport.",
            "Show number of passengers per flight.",
            "List top 5 airports by number of departures.",
            "Show total booking value per passenger.",
            "Count number of flights per aircraft type.",
            "Show average ticket price per route.",
            "List top 3 passengers by total booking value.",
        ],
    },
    "D": {
        "name": "Ambiguity / Incomplete Queries",
        "queries": [
            "Show top customers.",
            "List best passengers.",
            "Show revenue.",
            "List popular routes.",
            "Show frequent users.",
            "Show performance of flights.",
            "Show customer activity.",
            "List top members.",
            "Show route statistics.",
            "Show booking trends.",
        ],
    },
    "E": {
        "name": "Unsafe / Analytical / Governance",
        "queries": [
            "Analyze booking patterns month over month for the past year.",
            "Compare frequent flyers and non-frequent flyers.",
            "Identify the most valuable customers with segmentation.",
            "Calculate revenue loss from cancelled flights.",
            "Show revenue trends for all routes without filtering.",
        ],
    },
}

MULTI_TURN_F = [
    "Show flights from JFK to ATL.",
    "Which passenger flew on this route the most?",
    "Show bookings for that passenger.",
    "What is their total booking value?",
    "Show flights on the same route last month.",
]


def _parse_groq_retry_after(error_str: str) -> int:
    """Extract retry-after seconds from a Groq 429 error message."""
    m = _re.search(r"try again in (\d+)m([\d.]+)s", error_str)
    if m:
        return int(m.group(1)) * 60 + int(float(m.group(2))) + 5
    m = _re.search(r"try again in ([\d.]+)s", error_str)
    if m:
        return int(float(m.group(1))) + 5
    return 60


def send_message(message: str, session_id: str, retries: int = 3) -> dict:
    payload = {"message": message, "session_id": session_id}
    for attempt in range(retries):
        try:
            resp = requests.post(CHAT_URL, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            # Groq 429 comes back as HTTP 200 with error field
            err = data.get("error", "") or ""
            if "429" in str(err) and "rate_limit" in str(err).lower():
                wait = _parse_groq_retry_after(str(err))
                print(f"\n  [!] GROQ RATE LIMIT — waiting {wait}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            return data
        except requests.exceptions.Timeout:
            return {"error": "TIMEOUT", "response": None}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return {"error": str(e), "response": None}
    return {"error": "GROQ_RATE_LIMIT_EXHAUSTED", "response": None}


def classify(result: dict, is_ambiguous_category: bool = False) -> str:
    """
    Returns one of:
      PASS_SQL       — SQL generated + results returned
      PASS_CLARIFY   — clarification_needed=True (expected for cat D)
      PASS_BLOCKED   — cost guard / governance block triggered (expected for cat E)
      FAIL_NO_SQL    — no SQL, no clarification, no useful response
      FAIL_ERROR     — error key in response
      FAIL_TIMEOUT   — request timed out
    """
    if result.get("error") == "TIMEOUT":
        return "FAIL_TIMEOUT"
    if result.get("error"):
        return "FAIL_ERROR"
    if result.get("clarification_needed"):
        return "PASS_CLARIFY"
    sql = result.get("sql_query")
    data = result.get("data") or result.get("query_results")
    response_text = result.get("response", "") or ""
    # Check if cost guard / governance message
    if any(kw in response_text.lower() for kw in ["cost guard", "row limit", "too large", "governance", "analytical"]):
        return "PASS_BLOCKED"
    if sql:
        return "PASS_SQL"
    if response_text and len(response_text) > 10:
        return "FAIL_NO_SQL"
    return "FAIL_NO_SQL"


def run_category(cat_id: str, cat: dict, results: list):
    print(f"\n{'='*70}")
    print(f"  CATEGORY {cat_id} — {cat['name']}")
    print(f"{'='*70}")
    is_cat_d = cat_id == "D"
    is_cat_e = cat_id == "E"

    for i, query in enumerate(cat["queries"], 1):
        session_id = str(uuid.uuid4())  # fresh session per query
        print(f"\n  [{cat_id}{i:02d}] {query}")
        result = send_message(query, session_id)
        status = classify(result, is_ambiguous_category=(is_cat_d or is_cat_e))

        sql = result.get("sql_query", "")
        clarify_msg = ""
        if result.get("clarification_needed"):
            clarify_msg = f" -> \"{(result.get('response') or '')[:80]}\""

        error_msg = ""
        if result.get("error") and result["error"] != "TIMEOUT":
            error_msg = f" [ERR: {str(result['error'])[:60]}]"

        # Determine pass/fail based on category expectation
        if is_cat_d and status == "PASS_CLARIFY":
            verdict = "PASS"
        elif is_cat_e and status in ("PASS_BLOCKED", "PASS_CLARIFY"):
            verdict = "PASS"
        elif status in ("PASS_SQL",):
            verdict = "PASS"
        else:
            verdict = "FAIL"

        icon = "[PASS]" if verdict == "PASS" else "[FAIL]"
        print(f"  {icon} {status}{clarify_msg}{error_msg}")
        if sql:
            # Show first line of SQL only
            first_line = sql.strip().split("\n")[0][:100]
            print(f"         SQL: {first_line}...")

        results.append({
            "id": f"{cat_id}{i:02d}",
            "category": cat_id,
            "query": query,
            "status": status,
            "verdict": verdict,
            "sql": sql,
            "response": (result.get("response") or "")[:200],
            "clarification_needed": result.get("clarification_needed", False),
        })
        time.sleep(3)  # avoid Groq rate-limit (30 rpm = 2s min; 3s safe)


def run_multiturn(results: list):
    print(f"\n{'='*70}")
    print(f"  CATEGORY F — Multi-Turn Context (sequential)")
    print(f"{'='*70}")
    session_id = str(uuid.uuid4())  # shared session for all F queries

    for i, query in enumerate(MULTI_TURN_F, 1):
        print(f"\n  [F{i:02d}] {query}")
        result = send_message(query, session_id)
        status = classify(result)

        sql = result.get("sql_query", "")
        clarify_msg = ""
        if result.get("clarification_needed"):
            clarify_msg = f" -> \"{(result.get('response') or '')[:80]}\""

        error_msg = ""
        if result.get("error") and result["error"] != "TIMEOUT":
            error_msg = f" [ERR: {str(result['error'])[:60]}]"

        verdict = "PASS" if status in ("PASS_SQL", "PASS_CLARIFY") else "FAIL"
        icon = "[PASS]" if verdict == "PASS" else "[FAIL]"
        print(f"  {icon} {status}{clarify_msg}{error_msg}")
        if sql:
            first_line = sql.strip().split("\n")[0][:100]
            print(f"         SQL: {first_line}...")

        results.append({
            "id": f"F{i:02d}",
            "category": "F",
            "query": query,
            "status": status,
            "verdict": verdict,
            "sql": sql,
            "response": (result.get("response") or "")[:200],
            "clarification_needed": result.get("clarification_needed", False),
        })
        time.sleep(0.5)


def print_summary(results: list):
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}")

    cat_totals = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_totals:
            cat_totals[cat] = {"pass": 0, "fail": 0, "statuses": {}}
        if r["verdict"] == "PASS":
            cat_totals[cat]["pass"] += 1
        else:
            cat_totals[cat]["fail"] += 1
        s = r["status"]
        cat_totals[cat]["statuses"][s] = cat_totals[cat]["statuses"].get(s, 0) + 1

    cat_names = {
        "A": "Simple Retrieval",
        "B": "Join Queries",
        "C": "Aggregation",
        "D": "Ambiguity/Clarification",
        "E": "Governance/Safety",
        "F": "Multi-Turn",
    }

    total_pass = 0
    total_all = 0
    for cat, data in sorted(cat_totals.items()):
        p, f = data["pass"], data["fail"]
        total = p + f
        total_pass += p
        total_all += total
        pct = (p / total * 100) if total else 0
        statuses_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["statuses"].items()))
        print(f"  Cat {cat} ({cat_names.get(cat,'?'):25s}): {p}/{total} passed ({pct:.0f}%) | {statuses_str}")

    print(f"\n  OVERALL: {total_pass}/{total_all} passed ({total_pass/total_all*100:.0f}%)")

    # Print failures
    failures = [r for r in results if r["verdict"] == "FAIL"]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            print(f"    [{r['id']}] {r['query'][:60]} -> {r['status']}")
            if r.get("response"):
                print(f"           Response: {r['response'][:100]}")


def main():
    print("\nOptimaX Evaluation Runner")
    print(f"Target: {BASE_URL}")

    # Verify backend is up
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=5).json()
        print(f"Backend: {h.get('status')} | version {h.get('version')} | model {h.get('model')}")
    except Exception as e:
        print(f"ERROR: Backend not reachable - {e}")
        return

    # Warm-up probe: send a trivial query to detect Groq rate limit before starting
    print("Checking Groq quota...")
    probe = send_message("List all airports.", f"probe-{uuid.uuid4()}", retries=1)
    probe_err = probe.get("error", "") or ""
    if "GROQ_RATE_LIMIT_EXHAUSTED" in probe_err or ("429" in str(probe_err) and "rate_limit" in str(probe_err).lower()):
        wait = _parse_groq_retry_after(str(probe_err))
        print(f"  Groq quota exhausted. Waiting {wait}s for reset...")
        time.sleep(wait)
    else:
        print(f"  Groq OK - probe: {probe.get('sql_query') and 'SQL' or probe.get('clarification_needed') and 'CLARIFY' or 'response received'}")
        time.sleep(3)

    results = []

    for cat_id, cat in CATEGORIES.items():
        run_category(cat_id, cat, results)

    run_multiturn(results)

    print_summary(results)

    # Save full results to JSON
    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
