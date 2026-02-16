# Schema-Grounded Reasoning Disclosure (SGRD) — User Interaction Contract

## Purpose

SGRD exposes **decisions already made** by the system, grounded in schema, rules, or resolved intent. It never exposes speculative model reasoning, chain-of-thought, or internal heuristics.

Every disclosure is:
- **Deterministic** — same input produces same output
- **Verifiable** — traceable to a whitelisted internal object
- **Template-based** — no free-form LLM generation

---

## Delivery Mechanism

SGRD output is returned in the `agent_reasoning` field of the `ChatResponse` API model. This field was previously unused (`null`). It is now populated with a deterministic disclosure string at every pipeline exit point.

---

## What the User Sees

### A. Before Execution (Intent Acknowledgement)

**When:** The system has sufficient intent (entity + metric) and proceeds to SQL execution.

**Source:** `IntentState.entity_type`, `IntentState.metric`, `IntentState.row_select`

**Examples:**
| Intent State | Disclosure |
|---|---|
| entity=airport, metric=flight_count | `"Querying airport data to compute flight_count."` |
| entity=flight, row_select=true | `"Retrieving flight records."` |
| entity=passenger, metric=booking_value | `"Querying passenger data to compute booking_value."` |

**Guarantees:**
- Mentions only entity and metric
- Never explains how intent was inferred
- Never speculates about user meaning

---

### B. During Clarification (Semantic Ambiguity)

**When:** The system cannot proceed because entity, metric, or entity disambiguation is missing/ambiguous.

**Source:** `ClarificationContext.reason`, `.detected_value`, `.schema_source`

**Examples:**
| Reason | Disclosure |
|---|---|
| missing_entity | `"No entity type was identified in the query. Clarification required."` |
| missing_metric (entity=passenger) | `"Entity 'passenger' identified but no metric specified."` |
| invalid_metric_token (detected=list) | `"'list' is an action verb, not a measurable metric. A metric is required to proceed."` |
| bare_aggregation (detected=count) | `"'count' is an aggregation primitive without a target dimension. Clarification required to determine what to aggregate."` |
| comparison_query | `"Comparison queries require explicit segments and metrics. Clarification required."` |
| entity_rebinding (detected=customer) | `"The term 'customer' maps to multiple entities in this database."` |

**Guarantees:**
- Grounded in ClarificationReason enum (finite set)
- Schema source included when available
- Never guesses what user meant

---

### C. During Clarification (Relational / FK Ambiguity)

**When:** Multiple foreign key paths exist between tables and the system cannot auto-resolve.

**Source:** `PendingAmbiguity.column`, `.source_table`, `.target_table`, `.options`

**Example:**
```
Column 'airport_code' on 'airport' is reachable from 'flight' via 2 foreign key paths: departure_airport_id, arrival_airport_id. User clarification required to select the correct join path.
```

**Guarantees:**
- Names the ambiguous column
- Lists all valid FK paths
- Never auto-resolves
- Never uses "I think" or "probably"

---

### D. On Refusal (Cost / Safety Guard)

**When:** The cost guard blocks execution because estimated row scan exceeds the safety threshold.

**Source:** Cost guard result dict (`estimated_rows`, `error_type`)

**Example:**
```
Query not executed: estimated row scan of 1,500,000 exceeds the safety threshold. Narrow the query with filters to reduce scope.
```

**Guarantees:**
- States factual reason (row estimate)
- Frames as protection, not failure
- Suggests narrowing action
- Never exposes internal threshold constants

---

### E. On Context Resolution (Multi-Turn)

**When:** A referential phrase ("this route", "that passenger") is resolved from session context.

**Source:** `SessionContext.route_binding`, `ContextBinding`

**Examples:**
| Context | Disclosure |
|---|---|
| Route resolved | `"Interpreting route reference as JFK to ATL, based on previous query context."` |
| Entity resolved | `"Interpreting reference as passenger_id = 12345, based on previous query result."` |
| Unresolved | `"Reference could not be resolved from session context. Clarification requested."` |

**Guarantees:**
- Explicitly states what was resolved to what
- States source of resolution
- Requests clarification when unresolved

---

### F. On Success

**When:** SQL query executed successfully.

**Source:** `IntentState` (entity, metric) + SQL result (`row_count`)

**Example:**
```
Querying airport data to compute flight_count. Query executed safely. Returned 10 rows.
```

**Guarantees:**
- Confirms safe execution
- States row count
- No SQL shown (unless user requests it separately)

---

### G. On Failure

**When:** SQL execution failed (non-cost-guard).

**Source:** SQL result (`success=False`)

**Disclosure:** `"Query execution failed. No data was returned."`

**Guarantees:**
- No internal error details exposed
- No stack traces or SQL errors in disclosure

---

### H. On Greeting / Help

**When:** User sends a greeting or help request (no query).

**Disclosure:** `"No query detected. Greeting response returned."`

---

### I. On Visualization Request

**When:** User requests a chart/graph.

**Source:** Presence/absence of cached SQL result.

| State | Disclosure |
|---|---|
| Data available | `"Visualization requested. Using cached query results."` |
| No data | `"Visualization requested but no prior query data available."` |

---

## Whitelisted Data Sources

Only the following internal objects may contribute to user-facing disclosure:

| Object | Allowed Fields | Used For |
|---|---|---|
| `IntentState` | entity_type, metric, event, ranking, time_scope, row_select | Intent acknowledgement |
| `PendingAmbiguity` | source_table, target_table, column, options | FK clarification |
| `ClarificationContext` | reason, explanation, detected_value, schema_source | Semantic clarification |
| Cost guard result (dict) | estimated_rows, error_type | Cost refusal |
| `SessionContext` | route_binding.departure, route_binding.arrival | Context resolution |
| SQL result (dict) | row_count, success | Execution confirmation |

**If information is not present in these objects, it MUST NOT be shown.**

---

## Zero Behavioral Change Guarantee

This layer is **purely observational and presentational**:

- No execution paths were altered
- No decisions were added or changed
- No clarifications are triggered differently
- No queries are executed that were previously blocked
- No blocked queries are now executed
- The `reasoning_disclosure` field is a read-only projection appended to `PipelineResult`
- The `agent_reasoning` field in `ChatResponse` was previously always `null`; it now carries the disclosure string

---

## Implementation Files

| File | Change |
|---|---|
| `sgrd.py` | New file. Deterministic template functions. Zero logic. |
| `query_pipeline.py` | Added `reasoning_disclosure` field to `PipelineResult`. Populated at every exit point via SGRD read-only projections. |
| `main.py` | Wired `agent_reasoning=result.reasoning_disclosure` in chat endpoint. |
