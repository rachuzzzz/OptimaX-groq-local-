# Quantization SQL Quality Experiment

Compares SQL generation quality of Meta-Llama-3.1-8B-Instruct under two
quantization levels:

| Variant  | Precision | Size    | Tradeoff                        |
|----------|-----------|---------|----------------------------------|
| Q4\_K\_M | 4-bit     | ~4.9 GB | Fastest; risk of relational drift |
| Q8\_0    | 8-bit     | ~8.5 GB | Slower; higher relational fidelity |

**Single controlled variable:** quantization precision.
Everything else is held constant: same base weights, same prompt format,
`temperature=0`, `seed=42`, `top_k=1`.

**Hypothesis:** Q4 will degrade on multi-hop FK JOIN reasoning (Tier 4-5)
while handling single-table queries (Tier 1-2) acceptably.

---

## Prerequisites

### 1. llama.cpp + models

The experiment expects:

```
Desktop/
└── Quant-SQL-Experiment/
    ├── llama.cpp/
    │   └── llama-cli.exe          ← verified working
    ├── models/
    │   ├── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    │   └── Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
    └── outputs/
```

Download GGUF models (bartowski's conversion, HuggingFace):

```powershell
# from the Quant-SQL-Experiment folder
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir models

huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
  --local-dir models
```

### 2. Python environment

The experiment scripts use **only Python stdlib** — no new pip packages.
You can run them from the OptimaX venv or any Python 3.10+ interpreter.

```powershell
# From the OptimaX root (not sql-chat-backend)
python experiments/quant_sql/run_experiment.py --q4-only --query-id 9
```

---

## Running the Experiment

### Quick smoke test (one query, Q4 only)

```powershell
python experiments/quant_sql/run_experiment.py --q4-only --query-id 9 --verbose
```

### Full experiment (both models, all 14 queries)

```powershell
python experiments/quant_sql/run_experiment.py
```

Output JSON is saved to `experiments/quant_sql/outputs/`.

### Flags

| Flag | Effect |
|------|--------|
| `--q4-only` | Run Q4\_K\_M only |
| `--q8-only` | Run Q8\_0 only |
| `--query-id N` | Run only query N (1-14) |
| `--verbose` | Print prompts + raw inference output |

---

## Evaluating Results

```powershell
# Evaluate the most recent combined JSON
python experiments/quant_sql/evaluate_results.py

# Evaluate a specific file
python experiments/quant_sql/evaluate_results.py \
  --file experiments/quant_sql/outputs/experiment_combined_*.json

# Filter to one model
python experiments/quant_sql/evaluate_results.py --model Q4_K_M
```

The evaluator scores each query on:

1. **Valid SQL** (2pts) — is it a SELECT…FROM statement?
2. **Schema prefix** (1pt) — uses `postgres_air.` prefix
3. **Table existence** (2pts) — no hallucinated tables
4. **Column existence** (2pts) — no hallucinated qualified columns
5. **JOIN correctness** (3pts, Tier 3-5) — ON conditions follow declared FK pairs
6. **Aggregation structure** (2pts, Tier 2+5) — GROUP BY covers non-aggregated SELECTs
7. **Q9 FK path** (3pts bonus) — correct `account.frequent_flyer_id` path, not phone join

---

## OptimaX Baseline (three-way comparison)

Run the same 14 queries through the live OptimaX API for a comparison against
the production 70B model:

```powershell
# Start OptimaX first
cd sql-chat-backend
uvicorn main:app --reload --port 8000

# In another terminal
python experiments/quant_sql/run_through_optimax.py
```

Combine with local results for the full report:

```powershell
python experiments/quant_sql/evaluate_results.py \
  --file experiments/quant_sql/outputs/experiment_combined_*.json \
  --file experiments/quant_sql/outputs/experiment_optimax_*.json
```

---

## Test Query Tiers

| Tier | Description                  | Queries | Key Challenge |
|------|------------------------------|---------|---------------|
| 1    | Single table, no joins       | 1-3     | Basic projection |
| 2    | Aggregation, no joins        | 4-6     | GROUP BY structure |
| 3    | Single JOIN                  | 7-8     | FK ON condition |
| 4    | Multi-hop JOINs              | 9-11    | Correct FK path selection |
| 5    | Aggregation across JOINs     | 12-14   | Combined |

**Query 9** is the critical test: "Top 10 passengers with the most frequent
flyer points."

- **Correct path:** `passenger → account → frequent_flyer`
  via `account.frequent_flyer_id = frequent_flyer.frequent_flyer_id`
- **Wrong path:** `passenger → account → phone → frequent_flyer`
  via `phone.phone = frequent_flyer.phone` (string match, not FK)

---

## File Structure

```
experiments/quant_sql/
├── README.md                ← This file
├── schema_ddl.py            ← postgres_air DDL + FK definitions (shared)
├── test_queries.py          ← 14 test queries with gold SQL
├── run_experiment.py        ← Main: NL → llama.cpp Q4/Q8 → SQL → JSON
├── evaluate_results.py      ← Post-hoc: JSON → structural analysis → report
├── run_through_optimax.py   ← Compare: same queries via production pipeline
└── outputs/                 ← Experiment results (gitignored)
```

---

## Limitations

- The schema DDL in prompts uses representative synthetic sample rows (the
  experiment does not have live DB access). Actual row values would come from
  LlamaIndex's SQLDatabase introspection.
- Llama 3.1 8B is significantly smaller than OptimaX's production Groq
  llama-3.3-70b. Expect lower absolute SQL quality — the experiment measures
  *relative* Q4 vs Q8 degradation, not parity with 70B.
- Structural evaluation uses stdlib regex, not a full SQL parser. Complex
  aliasing patterns or CTEs may produce false positive/negative column
  violation reports.
