"""
Schema-Grounded Reasoning Disclosure (SGRD)
============================================

PURPOSE:
Exposes ONLY decisions already made by the system, grounded in schema,
rules, or resolved intent. Never speculative model reasoning.

DESIGN PRINCIPLE:
    Expose decisions, not thoughts.
    Expose schema facts, not guesses.

Every user-facing explanation is:
    - Deterministic (same input -> same output)
    - Verifiable (traceable to an internal object)
    - Template-based (no free-form generation)

NON-NEGOTIABLE:
    - This module is READ-ONLY. It reads existing objects and produces strings.
    - It contains ZERO decision logic, ZERO LLM calls, ZERO heuristics.
    - It does NOT modify any object it reads.

ALLOWED DATA SOURCES (WHITELIST):
    1. IntentState         -> entity, metric, event, ranking, time_scope, row_select
    2. PendingAmbiguity    -> source_table, target_table, column, options
    3. RelationalAmbiguity -> source_table, target_table, column, options
    4. ClarificationContext -> reason, explanation, detected_value, schema_source
    5. Cost guard result   -> estimated_rows, error_type
    6. SessionContext      -> route_binding (departure, arrival)
    7. SQL execution result-> row_count, success

    If information is NOT present in these objects, it MUST NOT be shown.

Author: OptimaX Team
Version: 1.0 (Schema-Grounded Reasoning Disclosure)
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# =============================================================================
# WHITELIST: Allowed Data Sources
# =============================================================================
# Only data from these internal objects may surface to users.
# Each entry documents the object, the allowed fields, and their use.
#
# OBJECT                  | ALLOWED FIELDS                        | USED FOR
# ----------------------- | ------------------------------------- | ------------------
# IntentState             | entity_type, metric, event,           | Intent acknowledgement
#                         | ranking, time_scope, row_select       |
# PendingAmbiguity        | source_table, target_table, column,   | FK clarification
#                         | options                               |
# ClarificationContext    | reason, explanation, detected_value,  | Semantic clarification
#                         | schema_source                         |
# Cost guard result (dict)| estimated_rows, error_type            | Cost refusal
# SessionContext          | route_binding.departure,              | Context resolution
#                         | route_binding.arrival                 |
# SQL result (dict)       | row_count, success                    | Execution confirmation
# =============================================================================


# =============================================================================
# A. INTENT ACKNOWLEDGEMENT (Pre-Execution)
# =============================================================================
# Source: IntentState
# Rules:
#   - Must mention entity and metric only
#   - Must not explain how intent was inferred
#   - Must not speculate

def disclose_intent_acknowledgement(
    entity_type: Optional[str],
    metric: Optional[str],
    row_select: bool = False
) -> str:
    """
    Generate intent acknowledgement message before SQL execution.

    Source: IntentState.entity_type, IntentState.metric, IntentState.row_select

    Template rules:
        - Mentions entity and metric only
        - Does not explain how intent was inferred
        - Does not speculate

    Returns:
        Deterministic disclosure string.
    """
    if row_select and entity_type:
        return f"Retrieving {entity_type} records."

    if entity_type and metric:
        return f"Querying {entity_type} data to compute {metric}."

    if entity_type:
        return f"Querying {entity_type} data."

    return "Processing query."


# =============================================================================
# B. SEMANTIC CLARIFICATION (Entity / Metric Ambiguity)
# =============================================================================
# Source: ClarificationContext (reason, detected_value, schema_source)
# Rules:
#   - Explain why clarification is needed
#   - Ground explanation in schema ambiguity
#   - List concrete options

def disclose_semantic_clarification(
    reason: str,
    detected_value: Optional[str] = None,
    entity_type: Optional[str] = None,
    schema_source: Optional[str] = None
) -> str:
    """
    Generate reasoning disclosure for semantic clarification.

    Source: ClarificationContext fields

    Template rules:
        - States factual reason for clarification
        - Grounded in schema or intent state
        - No speculation about user intent

    Returns:
        Deterministic disclosure string.
    """
    # Map ClarificationReason values to disclosure templates
    templates = {
        "missing_entity": "No entity type was identified in the query. Clarification required.",

        "missing_metric": (
            f"Entity '{entity_type}' identified but no metric specified."
            if entity_type
            else "No metric was specified. Clarification required."
        ),

        "invalid_metric_token": (
            f"'{detected_value}' is an action verb, not a measurable metric. "
            f"A metric is required to proceed."
            if detected_value
            else "The specified term is an action verb, not a measurable metric."
        ),

        "bare_aggregation": (
            f"'{detected_value}' is an aggregation primitive without a target dimension. "
            f"Clarification required to determine what to aggregate."
            if detected_value
            else "Aggregation specified without a target dimension."
        ),

        "comparison_query": (
            "Comparison queries require explicit segments and metrics. "
            "Clarification required."
        ),

        "entity_rebinding": (
            f"The term '{detected_value}' maps to multiple entities in this database."
            if detected_value
            else "The specified term maps to multiple entities in this database."
        ),

        "relational_ambiguity": (
            f"Multiple join paths exist for '{detected_value}'. "
            f"Clarification required to select the correct relationship."
            if detected_value
            else "Multiple join paths detected. Clarification required."
        ),
    }

    disclosure = templates.get(reason, "Clarification required to proceed.")

    if schema_source:
        disclosure += f" Source: {schema_source}."

    return disclosure


# =============================================================================
# C. RELATIONAL (Foreign-Key) AMBIGUITY
# =============================================================================
# Source: PendingAmbiguity
# Rules:
#   - Name the ambiguous column
#   - List valid join targets
#   - No auto-resolution language

def disclose_relational_ambiguity(
    column: str,
    source_table: str,
    target_table: str,
    options: List[Dict[str, str]]
) -> str:
    """
    Generate reasoning disclosure for FK ambiguity.

    Source: PendingAmbiguity fields

    Template rules:
        - Names the ambiguous column
        - Lists valid join targets
        - No auto-resolution language

    Returns:
        Deterministic disclosure string.
    """
    # Strip schema prefix for readability
    source_simple = source_table.split(".")[-1]
    target_simple = target_table.split(".")[-1]

    fk_names = [opt.get("fk_column", "unknown") for opt in options]

    return (
        f"Column '{column}' on '{target_simple}' is reachable from '{source_simple}' "
        f"via {len(options)} foreign key paths: {', '.join(fk_names)}. "
        f"User clarification required to select the correct join path."
    )


# =============================================================================
# D. COST / SAFETY REFUSAL
# =============================================================================
# Source: Cost guard result dict
# Rules:
#   - State factual reason (row estimate, join count, missing filters)
#   - Frame refusal as protection, not failure
#   - Offer safe narrowing suggestions

def disclose_cost_refusal(
    estimated_rows: Optional[int] = None,
    error_type: Optional[str] = None
) -> str:
    """
    Generate reasoning disclosure for cost guard refusal.

    Source: Cost guard result dict (estimated_rows, error_type)

    Template rules:
        - States factual reason
        - Frames refusal as protection
        - Suggests narrowing

    Returns:
        Deterministic disclosure string.
    """
    if estimated_rows is not None:
        return (
            f"Query not executed: estimated row scan of {estimated_rows:,} "
            f"exceeds the safety threshold. "
            f"Narrow the query with filters to reduce scope."
        )

    if error_type == "cost_guard":
        return (
            "Query not executed: estimated cost exceeds safety threshold. "
            "Narrow the query with filters to reduce scope."
        )

    return "Query not executed due to safety constraints."


# =============================================================================
# E. CONTEXT RESOLUTION (Multi-Turn)
# =============================================================================
# Source: SessionContext (route_binding)
# Rules:
#   - Explicitly state what a reference was resolved to
#   - If unresolved, request clarification

def disclose_context_resolution(
    departure: Optional[str] = None,
    arrival: Optional[str] = None,
    entity_key: Optional[str] = None,
    entity_value: Optional[Any] = None,
    resolved: bool = True
) -> str:
    """
    Generate reasoning disclosure for multi-turn context resolution.

    Source: SessionContext.route_binding, ContextBinding

    Template rules:
        - States what reference was resolved to
        - If unresolved, states that clarification is needed

    Returns:
        Deterministic disclosure string.
    """
    if not resolved:
        return "Reference could not be resolved from session context. Clarification requested."

    if departure and arrival:
        return (
            f"Interpreting route reference as {departure} to {arrival}, "
            f"based on previous query context."
        )

    if entity_key and entity_value is not None:
        return (
            f"Interpreting reference as {entity_key} = {entity_value}, "
            f"based on previous query result."
        )

    return "Reference resolved from session context."


# =============================================================================
# F. EXECUTION SUCCESS CONFIRMATION
# =============================================================================
# Source: SQL execution result (row_count, success)
# Rules:
#   - Confirm safe execution
#   - State row count
#   - No SQL unless explicitly requested

def disclose_execution_success(row_count: int) -> str:
    """
    Generate reasoning disclosure for successful query execution.

    Source: SQL result dict (row_count)

    Template rules:
        - Confirms safe execution
        - States row count
        - No SQL shown

    Returns:
        Deterministic disclosure string.
    """
    return f"Query executed safely. Returned {row_count:,} row{'s' if row_count != 1 else ''}."


# =============================================================================
# G. EXECUTION FAILURE
# =============================================================================
# Source: SQL execution result (success=False)
# Rules:
#   - State that execution failed
#   - No internal details exposed

def disclose_execution_failure() -> str:
    """
    Generate reasoning disclosure for failed query execution.

    Source: SQL result dict (success=False)

    Returns:
        Deterministic disclosure string.
    """
    return "Query execution failed. No data was returned."


# =============================================================================
# H. GREETING (No Query)
# =============================================================================

def disclose_greeting() -> str:
    """
    Generate reasoning disclosure for greeting/help responses.

    No data source required — greeting is a routing decision.

    Returns:
        Deterministic disclosure string.
    """
    return "No query detected. Greeting response returned."


# =============================================================================
# I. VISUALIZATION REQUEST
# =============================================================================

def disclose_visualization(has_data: bool) -> str:
    """
    Generate reasoning disclosure for visualization requests.

    Source: Presence/absence of cached SQL result

    Returns:
        Deterministic disclosure string.
    """
    if has_data:
        return "Visualization requested. Using cached query results."
    return "Visualization requested but no prior query data available."


# =============================================================================
# COMPOSITE: Build disclosure from pipeline state
# =============================================================================
# These functions read pipeline-level objects and delegate to the
# appropriate template function above. They are the integration points
# called from query_pipeline.py.
# =============================================================================

def disclose_for_intent_proceed(accumulated: Any) -> str:
    """
    Build disclosure when pipeline proceeds to SQL execution.

    Reads: IntentState (entity_type, metric, row_select)

    Returns:
        Deterministic disclosure string.
    """
    if accumulated is None:
        return "Processing query."

    return disclose_intent_acknowledgement(
        entity_type=getattr(accumulated, "entity_type", None),
        metric=getattr(accumulated, "metric", None),
        row_select=getattr(accumulated, "row_select", False),
    )


def disclose_for_clarification(decision: Any) -> str:
    """
    Build disclosure when pipeline requests clarification.

    Reads: IntentDecision.context (ClarificationContext)

    Returns:
        Deterministic disclosure string.
    """
    context = getattr(decision, "context", None)
    if context is None:
        return "Clarification required to proceed."

    return disclose_semantic_clarification(
        reason=getattr(context, "reason", ""),
        detected_value=getattr(context, "detected_value", None),
        entity_type=None,  # Not always available at decision level
        schema_source=getattr(context, "schema_source", None),
    )


def disclose_for_pending_ambiguity(pending: Any) -> str:
    """
    Build disclosure when pipeline encounters relational ambiguity.

    Reads: PendingAmbiguity (column, source_table, target_table, options)

    Returns:
        Deterministic disclosure string.
    """
    if pending is None:
        return "Relational ambiguity detected. Clarification required."

    return disclose_relational_ambiguity(
        column=getattr(pending, "column", "unknown"),
        source_table=getattr(pending, "source_table", "unknown"),
        target_table=getattr(pending, "target_table", "unknown"),
        options=getattr(pending, "options", []),
    )


def disclose_for_cost_guard(result: Dict[str, Any]) -> str:
    """
    Build disclosure when cost guard blocks execution.

    Reads: Cost guard result dict (estimated_rows, error_type)

    Returns:
        Deterministic disclosure string.
    """
    return disclose_cost_refusal(
        estimated_rows=result.get("estimated_rows"),
        error_type=result.get("error_type"),
    )
