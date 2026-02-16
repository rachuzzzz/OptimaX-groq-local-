# OptimaX System Guarantees (v6.1.1)

## 1. Architectural Freeze Declaration

- Architecture is **frozen**. No new reasoning modules, pipeline stages, or decision authorities may be introduced.
- **Clarification authority resides exclusively in `intent_accumulator.py`**. No other module may decide whether to clarify or proceed.
- `query_pipeline.py` is pure orchestration. It contains zero decision logic.
- **SGRD (`sgrd.py`) is read-only and non-authoritative.** It projects existing decisions as deterministic template strings. It never alters control flow, reads user input, or invokes LLM calls.

## 2. What the System Guarantees

| Guarantee | Module | Enforcement Mechanism |
|---|---|---|
| No execution when entity is ambiguous | `intent_accumulator.py` | `can_proceed()` calls `is_ambiguous_entity()` for both aggregation and row-select paths. Returns `False` if entity matches `AMBIGUOUS_SEMANTIC_ALIASES`. |
| No execution when metric is missing | `intent_accumulator.py` | `can_proceed()` requires `has_metric() == True`. Invalid metric tokens (`list`, `show`, etc.) and bare aggregation primitives with ranking are rejected. |
| No silent entity rebinding | `intent_accumulator.py` | Ambiguous aliases (`customer`, `user`, `member`) always trigger clarification with explicit options derived from `AMBIGUOUS_SEMANTIC_ALIASES`. |
| No silent metric inference | `intent_accumulator.py` | `has_metric()` rejects action verbs via `INVALID_METRIC_TOKENS`. Bare aggregation primitives require an event binding or explicit user selection. |
| No execution before clarification | `query_pipeline.py` | `evaluate()` is called before any SQL path. If `decision.proceed == False`, the function returns a `PipelineResult` with `clarification_needed=True` and exits. NL-SQL, RCL, and database execution are unreachable. |
| Schema-grounded SQL generation only | `main.py` | `NLSQLTableQueryEngine` reads schema via SQLAlchemy introspection. No schema is injected into prompts. |
| SQL guard complexity enforcement | `sql_validator.py` | `QueryComplexityAnalyzer` scores structural complexity (JOINs, subqueries, missing filters). Score ≥ 10 blocks execution. |
| Preflight cost guard | `tools.py` | `_preflight_cost_check()` runs `EXPLAIN` before execution. Estimated rows exceeding `COST_THRESHOLD` (100,000) abort the query. Bounded aggregations (GROUP BY + LIMIT ≤ 100) and pure aggregates are exempt. |
| Deterministic clarification routing | `intent_accumulator.py` | `get_missing_field()` returns a fixed priority: `entity_type` → `entity_disambiguation` → `metric`. `_make_clarification_with_context()` maps each to a dedicated handler with schema-backed options. |
| Multi-turn state consistency | `intent_accumulator.py` | `IntentMerger.merge()` tracks `can_proceed()` transitions. `original_query` is updated on `False→True` transitions. FK preferences persist across `clear()` by default. |
| RCL relational ambiguity handling | `relational_corrector.py` | When multiple FK paths exist for a column, RCL emits a `RelationalAmbiguity` struct. The pipeline stores it as `PendingAmbiguity` and returns a clarification. Single FK paths are rewritten deterministically. Zero LLM involvement. |
| SQL safety validation | `tools.py` | `_validate_sql()` enforces SELECT-only, blocks dangerous keywords (`DROP`, `DELETE`, `INSERT`, etc.), and enforces LIMIT on every query. |

## 3. What the System Never Does

- **No probabilistic clarification.** Every clarify/proceed decision is a deterministic boolean from `can_proceed()`.
- **No hidden LLM decisions outside the defined pipeline.** LLM is invoked exactly twice: once for intent extraction, once for NL-SQL generation. Neither call makes clarify/proceed decisions.
- **No schema hallucination.** SQL generation is constrained to tables discovered via SQLAlchemy introspection at startup.
- **No execution without passing `can_proceed()`.** The sole path to SQL execution requires `evaluate()` to return `IntentDecision(proceed=True)`.
- **No silent override of cost guard.** Cost guard abort returns a structured error with `error_type="cost_guard"`. No fallback path bypasses it.
- **No hidden agent loops.** `ReActAgent` is retained only for non-SQL queries. All SQL queries flow through `QueryPipeline.handle()` in a single pass.
- **No automatic business logic derivation.** The system does not infer KPIs, segment definitions, or composite metrics. Metrics must be explicitly stated or selected from schema-backed suggestions.
- **No automatic metric construction.** When metric is missing or ambiguous, the system halts and presents options. It never synthesizes a metric from context.

## 4. Decision Contract

```
IF   entity is known
AND  metric is known (not action verb, not bare aggregation without binding)
AND  entity is not an ambiguous semantic alias
AND  query is not a comparison query
THEN → PROCEED to NL-SQL execution

IF   row-select detected (entity known, metric is action verb, no ranking)
AND  entity is not an ambiguous semantic alias
THEN → PROCEED as row-select (SELECT * with LIMIT)

IF   relational ambiguity detected post-execution (multiple FK paths)
THEN → HALT, store PendingAmbiguity, return clarification

IF   cost guard estimate exceeds threshold
THEN → ABORT with structured cost_guard error

OTHERWISE → CLARIFY (return missing field clarification with options)
```

## 5. Scope Limitations (Explicitly Declared)

- The system answers only questions expressible as single SQL SELECT statements over the detected schema.
- It does not perform forecasting, trend prediction, or statistical modeling.
- It does not derive business KPIs unless they correspond directly to schema columns or single-hop FK aggregations.
- It does not auto-aggregate across implicit business logic (e.g., "best customer" requires explicit metric selection).
- It does not support multi-statement transactions, CTEs composed outside NL-SQL, or cross-database queries.

## 6. Reviewer Assurance Summary

- **Clarification authority is centralized** in `intent_accumulator.py`. No other module may issue or suppress clarifications.
- **Execution path is gated** by `evaluate()` → `can_proceed()`. SQL generation and database access are unreachable when `proceed == False`.
- **Ambiguity cannot bypass the intent layer.** Row-select queries, aggregation queries, and all other query types pass through the same `is_ambiguous_entity()` check before execution.
- **SGRD is read-only.** It reads `IntentState`, `ClarificationContext`, and `PendingAmbiguity` objects. It writes nothing. It decides nothing.
