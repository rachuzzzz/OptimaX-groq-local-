"""
run_experiment.py — Quantization SQL Generation Experiment
===========================================================

Runs each of the 14 test queries through two local GGUF models:
  Q4_K_M  — 4-bit mixed precision  (~4.9 GB)
  Q8_0    — 8-bit uniform precision (~8.5 GB)

Both models are Meta-Llama-3.1-8B-Instruct (bartowski GGUF conversion).
The single controlled variable is quantization precision.

All inference is deterministic: temperature=0, seed=42, top_k=1.

Results are saved as JSON to experiments/quant_sql/outputs/.

Usage:
    python run_experiment.py              # run both Q4 and Q8
    python run_experiment.py --q4-only   # Q4 only
    python run_experiment.py --q8-only   # Q8 only
    python run_experiment.py --verbose   # print prompts + raw outputs
    python run_experiment.py --query-id 9          # single query (both models)
    python run_experiment.py --q4-only --query-id 9 --verbose  # debug one
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

# ─── Shared modules (in the same directory) ────────────────────────────────
_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from schema_ddl import build_prompt
from test_queries import TEST_QUERIES, TIER_LABELS

# ─── Path Detection ─────────────────────────────────────────────────────────
# Do NOT hardcode the Windows username.  Find the Quant-SQL-Experiment folder
# on the user's Desktop regardless of who is logged in.
_desktop = pathlib.Path.home() / "Desktop" / "Quant-SQL-Experiment"
_binary  = "llama-cli.exe" if sys.platform == "win32" else "llama-cli"

LLAMA_CLI = _desktop / "llama.cpp" / _binary
MODEL_Q4  = _desktop / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
MODEL_Q8  = _desktop / "models" / "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
OUTPUTS   = _HERE / "outputs"

# ─── Inference settings ──────────────────────────────────────────────────────
INFERENCE = {
    "n_predict":  768,   # max tokens to generate (SQL rarely exceeds 400 tokens)
    "ctx_size":   4096,  # context window (schema DDL ~2.5 KB; full prompt ~3 KB)
    "temperature": 0.0,
    "top_p":       1.0,
    "top_k":       1,
    "seed":        42,
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_paths() -> list:
    """
    Verify that llama-cli and both GGUF models exist on the Desktop.
    Returns list of error strings (empty = all OK).
    """
    errors = []
    if not _desktop.exists():
        errors.append(
            f"Quant-SQL-Experiment folder not found at: {_desktop}\n"
            f"  Expected structure:\n"
            f"    {_desktop}/\n"
            f"      llama.cpp/llama-cli.exe\n"
            f"      models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf\n"
            f"      models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf\n"
        )
        return errors  # can't check sub-paths if parent missing

    if not LLAMA_CLI.exists():
        errors.append(f"llama-cli not found: {LLAMA_CLI}")
    if not MODEL_Q4.exists():
        errors.append(
            f"Q4 model not found: {MODEL_Q4}\n"
            f"  Download: huggingface-cli download bartowski/"
            f"Meta-Llama-3.1-8B-Instruct-GGUF "
            f"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir {_desktop / 'models'}"
        )
    if not MODEL_Q8.exists():
        errors.append(
            f"Q8 model not found: {MODEL_Q8}\n"
            f"  Download: huggingface-cli download bartowski/"
            f"Meta-Llama-3.1-8B-Instruct-GGUF "
            f"Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --local-dir {_desktop / 'models'}"
        )
    return errors


def _run_inference(
    model_path: pathlib.Path,
    prompt: str,
    verbose: bool = False,
    timeout: int = 180,
) -> dict:
    """
    Run llama-cli with the given prompt and return a result dict.

    Writes the prompt to a temp file (avoids OS command-line length limits).
    Returns:
        {
            "raw_output": str,
            "elapsed_s": float,
            "returncode": int,
            "error": str | None,
        }
    """
    # Write prompt to temp file so the command line stays short
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="quant_prompt_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(prompt)

        cmd = [
            str(LLAMA_CLI),
            "-m",    str(model_path),
            "--file", tmp_path,
            "-n",    str(INFERENCE["n_predict"]),
            "-c",    str(INFERENCE["ctx_size"]),
            "--temp", str(INFERENCE["temperature"]),
            "--top-p", str(INFERENCE["top_p"]),
            "--top-k", str(INFERENCE["top_k"]),
            "--seed",  str(INFERENCE["seed"]),
            "--no-display-prompt",  # suppress echoing the prompt
            "--log-disable",        # suppress llama.cpp internal logging
        ]

        if verbose:
            print(f"\n[CMD] {' '.join(cmd[:4])} ... (prompt file: {tmp_path})")

        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0

        raw = proc.stdout

        # Fallback: some llama-cli builds write to stderr
        if not raw.strip() and proc.stderr.strip():
            raw = proc.stderr

        return {
            "raw_output": raw,
            "elapsed_s": round(elapsed, 2),
            "returncode": proc.returncode,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "raw_output": "",
            "elapsed_s": timeout,
            "returncode": -1,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as exc:
        return {
            "raw_output": "",
            "elapsed_s": 0.0,
            "returncode": -1,
            "error": str(exc),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_sql(raw_output: str) -> str | None:
    """
    Extract the SQL query from llama.cpp's raw output.

    Handles the following output patterns:
      1. Clean: "SELECT ... LIMIT 10"
      2. Model echoes prefix: "SQLQuery: SELECT ..."
      3. Model follows full template: "SELECT ...\nSQLResult: ..."
      4. Model includes preamble before SQL
      5. Markdown fences: ```sql SELECT ... ```

    Returns the cleaned SQL string, or None if extraction fails.
    """
    if not raw_output or not raw_output.strip():
        return None

    text = raw_output.strip()

    # ── Step 1: Remove markdown SQL fences ─────────────────────────────────
    fence_match = re.search(
        r"```(?:sql)?\s*(SELECT[\s\S]+?)```",
        text,
        re.IGNORECASE,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # ── Step 2: Extract text between SQLQuery: and SQLResult: ───────────────
    sq_match = re.search(
        r"SQLQuery:\s*([\s\S]+?)(?=SQLResult:|Answer:|$)",
        text,
        re.IGNORECASE,
    )
    if sq_match:
        candidate = sq_match.group(1).strip()
        if re.search(r"\bSELECT\b", candidate, re.IGNORECASE):
            # Trim at SQLResult: if model repeated the continuation
            candidate = re.split(r"SQLResult:", candidate, flags=re.IGNORECASE)[0]
            return candidate.strip()

    # ── Step 3: Find first SELECT statement ─────────────────────────────────
    select_match = re.search(
        r"(SELECT\b[\s\S]+?)(?:SQLResult:|Answer:|$)",
        text,
        re.IGNORECASE,
    )
    if select_match:
        candidate = select_match.group(1).strip()
        # Trim excessive continuation (model kept going past one statement)
        # Heuristic: stop at a blank line followed by non-SQL text
        lines = candidate.splitlines()
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # blank line — check if next substantive content is still SQL
                sql_lines.append("")
                continue
            # Stop if line looks like template continuation
            if re.match(r"^(SQLResult|Answer|Question)\s*:", stripped, re.IGNORECASE):
                break
            sql_lines.append(line)
        candidate = "\n".join(sql_lines).strip()
        if candidate:
            return candidate

    return None


def _model_label(model_path: pathlib.Path) -> str:
    name = model_path.name
    if "Q4" in name.upper():
        return "Q4_K_M"
    if "Q8" in name.upper():
        return "Q8_0"
    return name.split(".")[0]


def run_single(
    query: dict,
    model_path: pathlib.Path,
    verbose: bool = False,
) -> dict:
    """
    Run one test query through one model.  Returns a result record.
    """
    qid   = query["id"]
    tier  = query["tier"]
    nl    = query["nl"]
    label = _model_label(model_path)

    print(f"  [Q{qid:02d}][{label}] {nl[:60]}", end=" ... ", flush=True)

    prompt = build_prompt(nl)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"PROMPT (first 400 chars):\n{prompt[:400]}…")

    inference = _run_inference(model_path, prompt, verbose=verbose)

    if verbose and inference["raw_output"]:
        print(f"\nRAW OUTPUT:\n{inference['raw_output'][:600]}")

    sql = None
    extraction_error = None
    if inference["error"]:
        extraction_error = inference["error"]
    else:
        sql = extract_sql(inference["raw_output"])
        if sql is None:
            extraction_error = "SQL extraction failed — no SELECT found in output"

    status = "ok" if sql else ("inference_error" if inference["error"] else "extraction_failed")
    print(f"{status} ({inference['elapsed_s']}s)")

    return {
        "query_id":        qid,
        "tier":            tier,
        "nl":              nl,
        "model":           label,
        "model_path":      str(model_path),
        "prompt_len":      len(prompt),
        "raw_output":      inference["raw_output"],
        "extracted_sql":   sql,
        "elapsed_s":       inference["elapsed_s"],
        "returncode":      inference["returncode"],
        "status":          status,
        "extraction_error": extraction_error,
        "gold_sql":        query["gold_sql"],
        "key_joins":       query["key_joins"],
        "notes":           query["notes"],
    }


def run_model(
    model_path: pathlib.Path,
    queries: list,
    verbose: bool = False,
) -> list:
    """Run all queries through one model.  Returns list of result records."""
    label = _model_label(model_path)
    print(f"\n{'═'*60}")
    print(f"MODEL: {label}  ({model_path.name})")
    print(f"{'═'*60}")

    results = []
    for q in queries:
        record = run_single(q, model_path, verbose=verbose)
        results.append(record)

    ok  = sum(1 for r in results if r["status"] == "ok")
    tot = len(results)
    print(f"\n  Summary: {ok}/{tot} SQL extracted successfully")
    return results


def save_results(results: list, label: str) -> pathlib.Path:
    """Save results to a timestamped JSON file in the outputs folder."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS / f"experiment_{label}_{ts}.json"

    payload = {
        "experiment": "quant_sql",
        "timestamp":  datetime.now().isoformat(),
        "inference":  INFERENCE,
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n  Saved → {out_path}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization SQL experiment — Q4 vs Q8 SQL generation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--q4-only",  action="store_true", help="Run Q4_K_M model only")
    parser.add_argument("--q8-only",  action="store_true", help="Run Q8_0 model only")
    parser.add_argument("--verbose",  action="store_true", help="Print prompts and raw outputs")
    parser.add_argument(
        "--query-id",
        type=int,
        default=None,
        metavar="N",
        help="Run only query #N (1-14) for quick testing",
    )
    args = parser.parse_args()

    # ── Path validation ─────────────────────────────────────────────────────
    errors = _check_paths()
    if errors:
        print("\n[ERROR] Setup problems detected:")
        for e in errors:
            print(f"  ✗ {e}")
        print(
            "\nFix the above issues, then re-run.\n"
            "See experiments/quant_sql/README.md for setup instructions."
        )
        sys.exit(1)
    print("[OK] All paths verified:")
    print(f"  llama-cli : {LLAMA_CLI}")
    print(f"  Q4 model  : {MODEL_Q4}")
    print(f"  Q8 model  : {MODEL_Q8}")
    print(f"  Outputs   : {OUTPUTS}")

    # ── Query selection ─────────────────────────────────────────────────────
    queries = TEST_QUERIES
    if args.query_id is not None:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"\n[ERROR] No query with id={args.query_id}. Valid: 1-14.")
            sys.exit(1)
        print(f"\n[INFO] Running single query: #{args.query_id}")
    else:
        print(f"\n[INFO] Running all {len(queries)} queries across 5 tiers")
        for tier in sorted({q["tier"] for q in queries}):
            tier_qs = [q for q in queries if q["tier"] == tier]
            print(f"  Tier {tier} ({TIER_LABELS[tier]}): {len(tier_qs)} queries")

    # ── Model selection ──────────────────────────────────────────────────────
    models_to_run = []
    if not args.q8_only:
        models_to_run.append(MODEL_Q4)
    if not args.q4_only:
        models_to_run.append(MODEL_Q8)

    if not models_to_run:
        print("\n[ERROR] --q4-only and --q8-only are mutually exclusive.")
        sys.exit(1)

    # ── Run experiments ──────────────────────────────────────────────────────
    all_results: list = []
    saved_files: list = []

    for model_path in models_to_run:
        label   = _model_label(model_path)
        results = run_model(model_path, queries, verbose=args.verbose)
        all_results.extend(results)
        out = save_results(results, label)
        saved_files.append(out)

    # ── Combined save (for evaluate_results.py) ──────────────────────────────
    if len(models_to_run) > 1:
        combined_path = save_results(all_results, "combined")
        saved_files.append(combined_path)
        print(f"\n[INFO] Combined results → {combined_path}")
        print(
            f"\nRun evaluator:\n"
            f"  python experiments/quant_sql/evaluate_results.py "
            f"--file {combined_path}"
        )
    else:
        print(
            f"\nRun evaluator:\n"
            f"  python experiments/quant_sql/evaluate_results.py "
            f"--file {saved_files[0]}"
        )

    # ── Quick tier summary ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("QUICK SUMMARY")
    print(f"{'─'*60}")
    for model_path in models_to_run:
        label = _model_label(model_path)
        model_results = [r for r in all_results if r["model"] == label]
        ok = sum(1 for r in model_results if r["status"] == "ok")
        print(f"  {label}: {ok}/{len(model_results)} SQL extracted")
        for tier in sorted({q["tier"] for q in queries}):
            tier_res = [r for r in model_results if r["tier"] == tier]
            tier_ok  = sum(1 for r in tier_res if r["status"] == "ok")
            print(f"    Tier {tier}: {tier_ok}/{len(tier_res)}")


if __name__ == "__main__":
    main()
