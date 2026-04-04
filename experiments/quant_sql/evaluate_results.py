"""
evaluate_results.py — Structural SQL Evaluation
================================================

Loads experiment JSON output(s) from run_experiment.py (or run_through_optimax.py)
and performs structural validation of each generated SQL query against the
declared postgres_air schema.

Validation dimensions (stdlib regex only, no sqlparse):
  1. Basic validity       — does the output parse as a SELECT statement?
  2. Schema qualification — does it use the postgres_air. prefix?
  3. Table existence      — are all referenced tables in the schema?
  4. Column existence     — are referenced qualified columns real?
  5. JOIN correctness     — do ON conditions match declared FK relationships?
  6. Aggregation structure — does GROUP BY cover non-aggregated SELECT columns?

Special detection:
  Query 9 (frequent flyer) — correct FK path vs. wrong phone-string join.

Usage:
    python evaluate_results.py --file outputs/experiment_combined_*.json
    python evaluate_results.py --dir  outputs/
    python evaluate_results.py --file f1.json --file f2.json
"""

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))
from schema_ddl import SCHEMA_TABLES, FK_RELATIONSHIPS, FK_JOIN_PAIRS

# ---------------------------------------------------------------------------
# Scoring weights (per query, per dimension)
# ---------------------------------------------------------------------------
W_VALID_SQL      = 2   # Is it a SELECT statement?
W_SCHEMA_PREFIX  = 1   # Uses postgres_air. prefix
W_TABLE_EXISTS   = 2   # No hallucinated tables
W_COL_EXISTS     = 2   # No hallucinated qualified columns
W_JOIN_CORRECT   = 3   # ON conditions follow declared FK pairs (Tier 3-5)
W_AGG_STRUCTURE  = 2   # GROUP BY covers non-aggregated SELECTs (Tier 2, 5)
W_CORRECT_FF_PATH= 3   # Query 9 bonus for taking the right FK path

MAX_SCORE_BY_TIER = {
    1: W_VALID_SQL + W_SCHEMA_PREFIX + W_TABLE_EXISTS + W_COL_EXISTS,
    2: W_VALID_SQL + W_SCHEMA_PREFIX + W_TABLE_EXISTS + W_COL_EXISTS + W_AGG_STRUCTURE,
    3: W_VALID_SQL + W_SCHEMA_PREFIX + W_TABLE_EXISTS + W_COL_EXISTS + W_JOIN_CORRECT,
    4: W_VALID_SQL + W_SCHEMA_PREFIX + W_TABLE_EXISTS + W_COL_EXISTS + W_JOIN_CORRECT,
    5: W_VALID_SQL + W_SCHEMA_PREFIX + W_TABLE_EXISTS + W_COL_EXISTS + W_JOIN_CORRECT + W_AGG_STRUCTURE,
}

# ---------------------------------------------------------------------------
# SQL Parsing helpers (stdlib regex only)
# ---------------------------------------------------------------------------

_SCHEMA_PREFIX = "postgres_air"

def _strip_comments(sql: str) -> str:
    """Remove -- and /* */ comments."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def _strip_string_literals(sql: str) -> str:
    """Replace string literal content with placeholder to avoid false matches."""
    return re.sub(r"'[^']*'", "'__LIT__'", sql)


def _extract_table_refs(sql: str) -> list[str]:
    """
    Extract bare table names from FROM/JOIN clauses.
    Handles:  FROM postgres_air.flight f  →  "flight"
              JOIN postgres_air.booking AS b  →  "booking"
              FROM flight  →  "flight"
    Returns list of lowercase bare names.
    """
    clean = _strip_string_literals(_strip_comments(sql))
    pattern = re.compile(
        r"(?:FROM|JOIN)\s+"
        r"(?:postgres_air\.)?(\w+)"
        r"(?:\s+(?:AS\s+)?(\w+))?",
        re.IGNORECASE,
    )
    tables = []
    for m in pattern.finditer(clean):
        tname = m.group(1).lower()
        # Skip SQL keywords that might follow FROM/JOIN in edge cases
        if tname in {"select", "where", "on", "and", "or", "not", "join",
                     "left", "right", "inner", "outer", "cross", "natural"}:
            continue
        tables.append(tname)
    return tables


def _extract_alias_map(sql: str) -> dict[str, str]:
    """
    Build {alias: bare_table_name} from FROM/JOIN declarations.
    Covers:  FROM postgres_air.flight f
             JOIN postgres_air.booking AS b
             FROM booking  (no alias — maps name→name)
    """
    clean = _strip_string_literals(_strip_comments(sql))
    pattern = re.compile(
        r"(?:FROM|JOIN)\s+"
        r"(?:postgres_air\.)?(\w+)"
        r"(?:\s+(?:AS\s+)?(\w+))?",
        re.IGNORECASE,
    )
    alias_map: dict[str, str] = {}
    for m in pattern.finditer(clean):
        tname = m.group(1).lower()
        alias = (m.group(2) or tname).lower()
        if alias in {"on", "where", "set", "inner", "outer", "left",
                     "right", "cross", "join", "group", "order", "having"}:
            alias = tname
        alias_map[alias] = tname
        alias_map[tname] = tname  # table name always maps to itself
    return alias_map


def _extract_on_conditions(sql: str, alias_map: dict[str, str]) -> list[tuple[str, str]]:
    """
    Extract (table.col, table.col) pairs from ON clauses.
    Resolves aliases using alias_map.
    Returns list of canonical (table.col, table.col) tuples.
    """
    clean = _strip_string_literals(_strip_comments(sql))
    # Match: alias.col = alias.col  (with optional spaces around =)
    pattern = re.compile(
        r"\bON\b\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)",
        re.IGNORECASE,
    )
    pairs = []
    for m in pattern.finditer(clean):
        a1, c1, a2, c2 = m.group(1).lower(), m.group(2).lower(), m.group(3).lower(), m.group(4).lower()
        t1 = alias_map.get(a1, a1)
        t2 = alias_map.get(a2, a2)
        pairs.append((f"{t1}.{c1}", f"{t2}.{c2}"))
    return pairs


def _extract_select_columns(sql: str) -> list[str]:
    """
    Extract column expressions from the SELECT list (before FROM).
    Returns list of lowercased column expressions.
    """
    clean = _strip_string_literals(_strip_comments(sql))
    select_match = re.search(r"\bSELECT\b([\s\S]+?)\bFROM\b", clean, re.IGNORECASE)
    if not select_match:
        return []
    select_body = select_match.group(1)
    # Split on commas (avoid splitting inside parens)
    cols = []
    depth = 0
    buf   = ""
    for ch in select_body:
        if ch == "(":
            depth += 1; buf += ch
        elif ch == ")":
            depth -= 1; buf += ch
        elif ch == "," and depth == 0:
            cols.append(buf.strip()); buf = ""
        else:
            buf += ch
    if buf.strip():
        cols.append(buf.strip())
    return [c.lower() for c in cols if c.strip()]


def _has_aggregate(expr: str) -> bool:
    return bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", expr, re.IGNORECASE))


def _extract_group_by(sql: str) -> list[str]:
    """Extract GROUP BY expressions as lowercased list."""
    clean = _strip_string_literals(_strip_comments(sql))
    gb_match = re.search(
        r"\bGROUP\s+BY\b([\s\S]+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)",
        clean, re.IGNORECASE,
    )
    if not gb_match:
        return []
    gb_body = gb_match.group(1)
    items = [x.strip().lower() for x in gb_body.split(",") if x.strip()]
    return items


def _extract_qualified_columns(sql: str) -> list[tuple[str, str]]:
    """
    Extract (table_or_alias, column) from alias.column patterns in the SQL.
    """
    clean = _strip_string_literals(_strip_comments(sql))
    pattern = re.compile(r"\b(\w+)\.(\w+)\b")
    results = []
    for m in pattern.finditer(clean):
        left  = m.group(1).lower()
        right = m.group(2).lower()
        # Skip schema prefix hits (postgres_air.tablename)
        if left == "postgres_air":
            continue
        results.append((left, right))
    return results

# ---------------------------------------------------------------------------
# Structural validators
# ---------------------------------------------------------------------------

def validate(result: dict) -> dict:
    """
    Validate one experiment result record.
    Returns an evaluation dict with dimension scores and notes.
    """
    sql  = result.get("extracted_sql") or ""
    tier = result.get("tier", 0)
    qid  = result.get("query_id", 0)

    eval_record: dict = {
        "query_id": qid,
        "tier":     tier,
        "nl":       result.get("nl", ""),
        "model":    result.get("model", ""),
        "sql":      sql,
        "status":   result.get("status", "unknown"),
    }

    # ── No SQL generated ──────────────────────────────────────────────────────
    if not sql or not sql.strip():
        eval_record.update({
            "score":           0,
            "max_score":       MAX_SCORE_BY_TIER.get(tier, 8),
            "valid_sql":       False,
            "has_schema_prefix": False,
            "hallucinated_tables": [],
            "hallucinated_columns": [],
            "join_violations": [],
            "agg_structure":   None,
            "ff_path":         None,
            "notes":           ["No SQL extracted"],
        })
        return eval_record

    clean_sql = _strip_comments(sql)
    notes:  list[str] = []
    score: int = 0
    max_score  = MAX_SCORE_BY_TIER.get(tier, 8)

    # ── 1. Basic validity ──────────────────────────────────────────────────────
    is_select = bool(re.search(r"^\s*SELECT\b", clean_sql.strip(), re.IGNORECASE))
    has_from  = bool(re.search(r"\bFROM\b", clean_sql, re.IGNORECASE))
    valid_sql = is_select and has_from
    if valid_sql:
        score += W_VALID_SQL
    else:
        notes.append("Not a valid SELECT ... FROM statement")

    # ── 2. Schema prefix ──────────────────────────────────────────────────────
    has_prefix = bool(re.search(r"\bpostgres_air\.", clean_sql, re.IGNORECASE))
    if has_prefix:
        score += W_SCHEMA_PREFIX
    else:
        notes.append("Missing postgres_air. schema prefix")

    # ── 3. Table existence ─────────────────────────────────────────────────────
    referenced_tables = _extract_table_refs(clean_sql)
    hallucinated_tables = [t for t in referenced_tables if t not in SCHEMA_TABLES]
    if not hallucinated_tables:
        score += W_TABLE_EXISTS
    else:
        notes.append(f"Hallucinated tables: {hallucinated_tables}")

    # ── 4. Column existence ────────────────────────────────────────────────────
    alias_map  = _extract_alias_map(clean_sql)
    qual_cols  = _extract_qualified_columns(clean_sql)
    hallucinated_columns: list[str] = []

    for alias, col in qual_cols:
        table = alias_map.get(alias)
        if table is None:
            continue  # unresolved alias — skip (alias validator handles)
        if table not in SCHEMA_TABLES:
            continue  # already flagged as hallucinated table
        table_cols = SCHEMA_TABLES[table]
        if col not in table_cols:
            hallucinated_columns.append(f"{alias}.{col} (table={table})")

    if not hallucinated_columns:
        score += W_COL_EXISTS
    else:
        notes.append(f"Hallucinated columns: {hallucinated_columns}")

    # ── 5. JOIN correctness (Tier 3+) ─────────────────────────────────────────
    join_violations: list[str] = []
    on_pairs = _extract_on_conditions(clean_sql, alias_map)

    if tier >= 3:
        for left, right in on_pairs:
            canonical = f"{left}||{right}"
            if canonical not in FK_JOIN_PAIRS:
                # Could be a WHERE-style equality not in FK spec — check if it
                # involves tables that ARE in the schema
                left_t  = left.split(".")[0]
                right_t = right.split(".")[0]
                if left_t in SCHEMA_TABLES and right_t in SCHEMA_TABLES:
                    join_violations.append(f"{left} = {right}")

        if not join_violations:
            score += W_JOIN_CORRECT
        else:
            notes.append(f"JOIN violations (not in FK spec): {join_violations}")

    # ── 6. Aggregation structure (Tier 2, 5) ──────────────────────────────────
    agg_structure: str | None = None
    if tier in (2, 5):
        select_cols = _extract_select_columns(clean_sql)
        group_by    = _extract_group_by(clean_sql)

        # Check if there are any aggregates in SELECT
        has_agg   = any(_has_aggregate(c) for c in select_cols)
        # Non-aggregate columns that should appear in GROUP BY
        non_agg   = [c for c in select_cols if not _has_aggregate(c)]

        if has_agg and non_agg:
            # Each non-aggregate SELECT expr should appear (or its base name)
            # in the GROUP BY list
            missing_from_gb = []
            for col_expr in non_agg:
                # Strip alias:  "f.departure_airport AS dep"  →  "f.departure_airport"
                base = re.split(r"\s+as\s+", col_expr, flags=re.IGNORECASE)[0].strip()
                base_bare = base.split(".")[-1] if "." in base else base
                in_gb = any(
                    base in g or base_bare in g for g in group_by
                )
                if not in_gb:
                    missing_from_gb.append(col_expr)

            if not missing_from_gb:
                agg_structure = "valid"
                score += W_AGG_STRUCTURE
            else:
                agg_structure = f"missing in GROUP BY: {missing_from_gb}"
                notes.append(f"Aggregation structure: {agg_structure}")
        elif not has_agg and tier == 5:
            agg_structure = "no aggregate found (expected for Tier 5)"
            notes.append(agg_structure)
        else:
            agg_structure = "ok (no non-agg columns or no agg)"
            score += W_AGG_STRUCTURE

    # ── 7. Query 9 — Frequent Flyer path detection ────────────────────────────
    ff_path: dict | None = None
    if qid == 9:
        correct_path = (
            "account.frequent_flyer_id||frequent_flyer.frequent_flyer_id"
            in FK_JOIN_PAIRS
        )  # this is always True (in our FK_JOIN_PAIRS)

        # Check whether the GENERATED SQL uses the correct join
        uses_ff_id = any(
            ("frequent_flyer_id" in l and "frequent_flyer_id" in r)
            for l, r in on_pairs
        )
        uses_phone_join = bool(
            re.search(r"ph\.phone\s*=\s*ff\.phone|ff\.phone\s*=\s*ph\.phone", clean_sql, re.IGNORECASE)
            or re.search(r"\bphone\b.*\.\bphone\b.*=.*\bphone\b.*\.\bphone\b", clean_sql, re.IGNORECASE)
        )

        if uses_ff_id and not uses_phone_join:
            ff_path = {"verdict": "CORRECT", "path": "account.frequent_flyer_id → frequent_flyer.frequent_flyer_id"}
            score += W_CORRECT_FF_PATH
            max_score += W_CORRECT_FF_PATH
        elif uses_phone_join:
            ff_path = {"verdict": "WRONG", "path": "phone string match (not FK path)"}
            max_score += W_CORRECT_FF_PATH
        else:
            ff_path = {"verdict": "UNCLEAR", "path": str(on_pairs)}
            max_score += W_CORRECT_FF_PATH
            notes.append("Query 9: could not determine FF join path clearly")

    eval_record.update({
        "score":               score,
        "max_score":           max_score,
        "pct":                 round(100 * score / max_score, 1) if max_score else 0,
        "valid_sql":           valid_sql,
        "has_schema_prefix":   has_prefix,
        "hallucinated_tables": hallucinated_tables,
        "hallucinated_columns": hallucinated_columns,
        "join_violations":     join_violations,
        "on_pairs_found":      on_pairs,
        "agg_structure":       agg_structure,
        "ff_path":             ff_path,
        "notes":               notes,
    })
    return eval_record


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(evals: list[dict], model_filter: str | None = None) -> None:
    """Print a structured evaluation report."""
    if model_filter:
        evals = [e for e in evals if e["model"] == model_filter]

    models  = sorted({e["model"] for e in evals})
    tiers   = sorted({e["tier"]  for e in evals})

    print(f"\n{'═'*72}")
    print(f"  QUANTIZATION SQL EVALUATION REPORT")
    print(f"{'═'*72}")
    print(f"  Models evaluated: {', '.join(models)}")
    print(f"  Total records:    {len(evals)}")

    # ── Per-model summary ────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  OVERALL MODEL SCORES")
    print(f"{'─'*72}")

    model_totals: dict = {}
    for model in models:
        m_evals = [e for e in evals if e["model"] == model]
        total   = sum(e["score"]     for e in m_evals)
        mx      = sum(e["max_score"] for e in m_evals)
        pct     = round(100 * total / mx, 1) if mx else 0
        model_totals[model] = {"total": total, "max": mx, "pct": pct}
        print(f"  {model:<15} {total:>4}/{mx:<4}  ({pct:>5.1f}%)")

    # ── Per-tier breakdown ───────────────────────────────────────────────────
    from test_queries import TIER_LABELS
    print(f"\n{'─'*72}")
    print("  PER-TIER BREAKDOWN")
    print(f"{'─'*72}")
    tier_header = f"  {'Tier':<8}" + "".join(f"{m:<18}" for m in models)
    print(tier_header)
    print(f"  {'─'*68}")

    for tier in tiers:
        row = f"  Tier {tier}  "
        for model in models:
            m_t = [e for e in evals if e["model"] == model and e["tier"] == tier]
            t   = sum(e["score"]     for e in m_t)
            mx  = sum(e["max_score"] for e in m_t)
            pct = round(100 * t / mx, 1) if mx else 0
            row += f"{t}/{mx} ({pct:>5.1f}%)    "
        print(row)

    # ── Per-query breakdown ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  PER-QUERY COMPARISON")
    print(f"{'─'*72}")

    query_ids = sorted({e["query_id"] for e in evals})
    for qid in query_ids:
        q_evals = [e for e in evals if e["query_id"] == qid]
        first   = q_evals[0]
        nl_trunc = first["nl"][:50]
        print(f"\n  Q{qid:02d} [T{first['tier']}] {nl_trunc}")

        for e in sorted(q_evals, key=lambda x: x["model"]):
            verdict = f"{e['score']}/{e['max_score']} ({e['pct']}%)"
            issues  = "; ".join(e["notes"]) if e["notes"] else "clean"
            print(f"    {e['model']:<14} {verdict:<16} {issues}")

            # Show FF path verdict for Q9
            if qid == 9 and e.get("ff_path"):
                fp = e["ff_path"]
                print(f"    {'':14} FF path: {fp['verdict']} — {fp['path']}")

        # Cross-model winner
        if len(q_evals) > 1:
            best = max(q_evals, key=lambda x: x["score"])
            tied = all(e["score"] == q_evals[0]["score"] for e in q_evals)
            winner = "TIE" if tied else f"WINNER: {best['model']}"
            print(f"    → {winner}")

    # ── Frequent flyer path summary ──────────────────────────────────────────
    ff_evals = [e for e in evals if e.get("ff_path")]
    if ff_evals:
        print(f"\n{'─'*72}")
        print("  KNOWN FAILURE — Q9 FREQUENT FLYER FK PATH")
        print(f"  Correct: passenger→account→frequent_flyer via account.frequent_flyer_id")
        print(f"  Wrong:   passenger→account→phone→frequent_flyer via phone string match")
        print(f"{'─'*72}")
        for e in ff_evals:
            fp = e["ff_path"]
            print(f"  {e['model']:<14} → {fp['verdict']:8} ({fp['path']})")

    # ── Overall winner ───────────────────────────────────────────────────────
    if len(models) > 1:
        print(f"\n{'─'*72}")
        print("  OVERALL VERDICT")
        print(f"{'─'*72}")
        best_model = max(model_totals, key=lambda m: model_totals[m]["pct"])
        for m, t in model_totals.items():
            marker = " ← WINNER" if m == best_model else ""
            print(f"  {m:<15} {t['pct']:>5.1f}%{marker}")
        print()
        diff = abs(
            model_totals[models[0]]["pct"] - model_totals[models[1]]["pct"]
        ) if len(models) == 2 else 0
        if diff < 5:
            print("  Difference < 5% — models are essentially equivalent at this tier distribution.")
        elif diff < 15:
            print(f"  Moderate advantage ({diff:.1f}%) for {best_model}.")
        else:
            print(f"  Significant advantage ({diff:.1f}%) for {best_model}.")
            print("  This supports the hypothesis that quantization degrades multi-hop JOIN reasoning.")


def load_results(paths: list[pathlib.Path]) -> list[dict]:
    """Load and flatten result records from one or more JSON files."""
    all_results: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("results", [])
        all_results.extend(records)
    return all_results


def save_eval(evals: list[dict], source_path: pathlib.Path) -> pathlib.Path:
    """Save evaluation records alongside the source JSON."""
    out_path = source_path.parent / source_path.name.replace(
        "experiment_", "eval_"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evals, f, indent=2, default=str)
    print(f"\n  Evaluation JSON → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate quantization experiment results structurally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file", "-f",
        action="append",
        dest="files",
        metavar="PATH",
        help="JSON result file (can repeat for multiple files)",
    )
    parser.add_argument(
        "--dir", "-d",
        dest="result_dir",
        metavar="DIR",
        help="Evaluate all experiment_*.json files in this directory",
    )
    parser.add_argument(
        "--model",
        dest="model_filter",
        metavar="NAME",
        default=None,
        help="Filter report to one model (e.g. Q4_K_M)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save eval JSON (just print report)",
    )
    args = parser.parse_args()

    # Collect input files
    input_paths: list[pathlib.Path] = []

    if args.result_dir:
        d = pathlib.Path(args.result_dir)
        input_paths.extend(sorted(d.glob("experiment_*.json")))

    if args.files:
        for f in args.files:
            p = pathlib.Path(f)
            if not p.exists():
                print(f"[ERROR] File not found: {p}")
                sys.exit(1)
            input_paths.append(p)

    if not input_paths:
        # Default: look in the local outputs/ directory
        default_dir = _HERE / "outputs"
        input_paths = sorted(default_dir.glob("experiment_*.json"))
        if not input_paths:
            print(
                "[ERROR] No result files found.\n"
                "  Run: python run_experiment.py\n"
                "  Then: python evaluate_results.py"
            )
            sys.exit(1)
        print(f"[INFO] Using outputs from: {default_dir}")

    print(f"[INFO] Loading {len(input_paths)} file(s):")
    for p in input_paths:
        print(f"  {p}")

    results = load_results(input_paths)
    print(f"[INFO] {len(results)} result records loaded")

    # Evaluate
    evals = [validate(r) for r in results]

    # Report
    print_report(evals, model_filter=args.model_filter)

    # Save
    if not args.no_save and input_paths:
        save_eval(evals, input_paths[-1])


if __name__ == "__main__":
    main()
