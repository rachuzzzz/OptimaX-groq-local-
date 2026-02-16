"""
QueryPipeline - Pure Orchestration Controller

Orchestrates a single user query through:
- Intent extraction and accumulation
- NL-SQL generation
- Relational correction (RCL)
- Ambiguity resolution

Contains NO decision logic. The accumulator is the sole clarification authority.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from intent_accumulator import (
    PendingAmbiguity,
    match_ambiguity_response,
    format_ambiguity_clarification,
)
from context_resolver import apply_semantic_context_binding
from sgrd import (
    disclose_greeting,
    disclose_visualization,
    disclose_for_intent_proceed,
    disclose_for_clarification,
    disclose_for_pending_ambiguity,
    disclose_for_cost_guard,
    disclose_execution_success,
    disclose_execution_failure,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class PipelineResult:
    """Result from pipeline."""
    success: bool
    response: str
    session_id: str
    execution_time: float
    sql_query: Optional[str] = None
    data: Optional[List[Dict]] = None
    query_results: Optional[List[Dict]] = None
    clarification_needed: bool = False
    chart_suggestion: Optional[Dict] = None
    error: Optional[str] = None
    # SGRD: Schema-Grounded Reasoning Disclosure (read-only projection)
    reasoning_disclosure: Optional[str] = None


# =============================================================================
# QUERY PIPELINE
# =============================================================================

class QueryPipeline:
    """
    Pure orchestration. No decision logic.

    FLOW:
    1. Check viz/greeting (routing)
    2. Extract intent (one-shot)
    3. Accumulate intent
    4. Evaluate (accumulator decides)
    5. If proceed -> NL-SQL
    6. If clarify -> return clarification

    The accumulator is the SOLE authority for clarify/proceed.
    """

    def __init__(
        self,
        llm: Any,
        intent_extractor: Any,
        intent_merger: Any,
        execute_fn: Any,
        semantic_mediator: Any = None,
        get_last_sql_result_fn: Any = None,
        set_last_sql_result_fn: Any = None,
        clear_last_sql_result_fn: Any = None,
        classify_viz_intent_fn: Any = None,
    ):
        self.llm = llm
        self.extractor = intent_extractor
        self.merger = intent_merger
        self.execute = execute_fn
        self.observer = semantic_mediator
        self.get_last_sql_result = get_last_sql_result_fn
        self.set_last_sql_result = set_last_sql_result_fn
        self.clear_last_sql_result = clear_last_sql_result_fn
        self.classify_viz_intent = classify_viz_intent_fn

        logger.info("QueryPipeline initialized (v6.1.1 pure orchestration + relational ambiguity loop)")

    async def handle(
        self,
        query: str,
        session_id: str,
        session: Dict[str, Any],
        row_limit: int = 100,
        **kwargs
    ) -> PipelineResult:
        """
        Main entry point.

        FLOW:
        1. Visualization? -> handle
        2. Greeting? -> respond
        3. Extract intent
        4. Merge into accumulated state
        5. Evaluate (accumulator decides)
        6. If proceed -> NL-SQL
        7. If clarify -> return clarification
        """
        # === DEBUG HEARTBEAT ===
        print(f"[PIPELINE] QueryPipeline.handle() invoked: {query[:60]}...", flush=True)
        logger.info(f"[PIPELINE] QueryPipeline.handle() invoked: {query[:60]}...")
        # === END DEBUG HEARTBEAT ===

        start = datetime.now()

        try:
            # =================================================================
            # ROUTING: Viz / Greeting
            # =================================================================

            # Visualization request?
            if self._is_viz_request(query):
                return self._handle_viz(query, session_id, start)

            # Greeting?
            if self._is_greeting(query):
                return self._greeting_response(session_id, start)

            # Clear cached data for new queries
            if self.clear_last_sql_result:
                self.clear_last_sql_result()

            # =================================================================
            # INTENT: Extract -> Merge -> Evaluate
            # =================================================================

            from intent_accumulator import evaluate, format_clarification, get_query_for_nlsql

            accumulated = session.get("accumulated_intent")

            # =================================================================
            # STEP 0: CHECK FOR PENDING RELATIONAL AMBIGUITY (v6.1.1)
            # =================================================================
            # If there's a pending ambiguity from a previous turn, check if the
            # user's response resolves it. If so, resolve and re-execute.
            #
            # FLOW:
            # 1. Check if accumulated has pending ambiguity
            # 2. Try to match user's response against options
            # 3. If matched -> resolve ambiguity -> re-execute with forced FK
            # 4. If not matched -> proceed normally (might be a new query)
            # =================================================================
            if accumulated and accumulated.has_pending_ambiguity():
                pending = accumulated.pending_ambiguity
                logger.info(
                    f"[PIPELINE] Pending ambiguity detected: {pending.column} via "
                    f"{[o['fk_column'] for o in pending.options]}"
                )

                # Try to match user's response
                matched_fk = match_ambiguity_response(query, pending)

                if matched_fk:
                    logger.info(f"[PIPELINE] User resolved ambiguity: {matched_fk}")

                    # Resolve the ambiguity (stores preference)
                    accumulated.resolve_ambiguity(matched_fk)
                    session["accumulated_intent"] = accumulated

                    # Re-execute with the original query and forced FK
                    query_for_nlsql = pending.original_query
                    logger.info(f"[PIPELINE] Re-executing with forced FK: {query_for_nlsql[:60]}...")

                    # Execute with intent_state (has resolved FK preferences)
                    result = await self.execute(
                        user_query=query_for_nlsql,
                        row_limit=row_limit,
                        timeout=45.0,
                        intent_state=accumulated
                    )

                    # Handle result (success or error)
                    return self._handle_execution_result(
                        result, accumulated, session_id, start
                    )

                else:
                    # Didn't match - could be a new query or unclear response
                    # Check if it looks like a new query vs confused response
                    if self._is_new_query_pattern(query):
                        logger.info("[PIPELINE] Unmatched response looks like new query - clearing ambiguity")
                        accumulated.pending_ambiguity = None
                        session["accumulated_intent"] = accumulated
                    else:
                        # Re-ask the clarification
                        logger.info("[PIPELINE] Response didn't match options - re-asking")
                        return PipelineResult(
                            success=True,
                            response=(
                                "I didn't understand your choice. " +
                                format_ambiguity_clarification(pending)
                            ),
                            session_id=session_id,
                            execution_time=self._elapsed(start),
                            clarification_needed=True,
                            reasoning_disclosure=disclose_for_pending_ambiguity(pending),
                        )

            # Extract intent (one-shot, no decisions)
            extraction = self.extractor.extract(query)

            if extraction.success and extraction.intent:
                logger.info(
                    f"[PIPELINE] Extracted: entity={extraction.intent.entity_type}, "
                    f"metric={extraction.intent.metric}"
                )

                # Merge into accumulated state
                if accumulated and self.merger:
                    accumulated = self.merger.merge(accumulated, extraction.intent, query)
                    session["accumulated_intent"] = accumulated

                    # Evaluate (SOLE DECISION POINT)
                    decision = evaluate(accumulated)

                    if not decision.proceed:
                        # CLARIFY
                        return PipelineResult(
                            success=True,
                            response=format_clarification(decision),
                            session_id=session_id,
                            execution_time=self._elapsed(start),
                            clarification_needed=True,
                            reasoning_disclosure=disclose_for_clarification(decision),
                        )

                    # PROCEED - get query for NL-SQL
                    query_for_nlsql = get_query_for_nlsql(accumulated, query)

                else:
                    # No accumulator - just proceed with original query
                    query_for_nlsql = query

            else:
                # Extraction failed - proceed with original query anyway
                query_for_nlsql = query

            # =================================================================
            # SEMANTIC CONTEXT BINDING (v6.12 - Structured Constraint Injection)
            # =================================================================
            context_disclosure = None
            route_filters = None
            session_context = session.get("session_context")
            if session_context is not None:
                query_for_nlsql, context_disclosure, route_filters = apply_semantic_context_binding(
                    query=query_for_nlsql,
                    entity_type=getattr(accumulated, 'entity_type', None) if accumulated else None,
                    session_context=session_context,
                )
                if context_disclosure:
                    logger.info("[PIPELINE] Semantic context binding applied")

            # =================================================================
            # OBSERVATION (non-blocking)
            # =================================================================

            if self.observer:
                self.observer.observe(query_for_nlsql)

            # =================================================================
            # SQL EXECUTION
            # =================================================================

            logger.info(f"[PIPELINE] Executing: {query_for_nlsql[:60]}...")

            result = await self.execute(
                user_query=query_for_nlsql,
                row_limit=row_limit,
                timeout=45.0,
                intent_state=accumulated,  # v6.1.1: Pass intent state for FK preferences
                route_filters=route_filters,  # v6.12: Structured route constraints
            )

            # =================================================================
            # RELATIONAL AMBIGUITY HANDLING (v6.1.1)
            # =================================================================
            # If RCL detected an ambiguity (multiple FK paths), we need to
            # ask the user for clarification instead of failing.
            #
            # CRITICAL: Ambiguity is NEVER an error. Always return clarification.
            #
            # FLOW:
            # 1. Detect error_type == "relational_ambiguity"
            # 2. Store ambiguity in accumulated state (if available)
            # 3. Return clarification question (ALWAYS)
            # =================================================================
            if (not result["success"] and
                    result.get("error_type") == "relational_ambiguity" and
                    result.get("ambiguity")):

                ambiguity = result["ambiguity"]
                logger.info(
                    f"[PIPELINE] Relational ambiguity detected: "
                    f"column='{ambiguity['column']}', "
                    f"options={[o['fk_column'] for o in ambiguity['options']]}"
                )

                # Convert dict options to format expected by set_pending_ambiguity
                options = [
                    {
                        "fk_column": o["fk_column"],
                        "description": o["description"]
                    }
                    for o in ambiguity["options"]
                ]

                # Create accumulated state if it doesn't exist (edge case recovery)
                if not accumulated:
                    from intent_accumulator import create_intent_state
                    accumulated = create_intent_state()
                    logger.info("[PIPELINE] Created accumulated state for ambiguity storage")

                # Store ambiguity in accumulated state for next turn
                accumulated.set_pending_ambiguity(
                    source_table=ambiguity["source_table"],
                    target_table=ambiguity["target_table"],
                    column=ambiguity["column"],
                    options=options,
                    original_query=query_for_nlsql
                )
                session["accumulated_intent"] = accumulated

                # Build clarification question
                pending = accumulated.pending_ambiguity
                clarification = format_ambiguity_clarification(pending)

                # CRITICAL: ALWAYS return clarification, NEVER fall through to error handler
                return PipelineResult(
                    success=True,  # Not a failure - just need clarification
                    response=clarification,
                    session_id=session_id,
                    execution_time=self._elapsed(start),
                    sql_query=result.get("sql"),  # Include SQL for debugging
                    clarification_needed=True,
                    reasoning_disclosure=disclose_for_pending_ambiguity(pending),
                )

            # =================================================================
            # RESULT HANDLING
            # =================================================================

            return self._handle_execution_result(
                result, accumulated, session_id, start,
                context_disclosure=context_disclosure
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return PipelineResult(
                success=False,
                response="An error occurred. Please try again.",
                session_id=session_id,
                execution_time=self._elapsed(start),
                error=str(e),
                reasoning_disclosure=disclose_execution_failure(),
            )

    # =========================================================================
    # ROUTING HELPERS
    # =========================================================================

    def _is_viz_request(self, query: str) -> bool:
        keywords = ["visualize", "graph", "chart", "plot"]
        return any(kw in query.lower() for kw in keywords)

    def _is_greeting(self, query: str) -> bool:
        greetings = ["hi", "hello", "hey", "help", "what can you do"]
        return query.lower().strip() in greetings

    def _handle_viz(self, query: str, session_id: str, start: datetime) -> PipelineResult:
        last = self.get_last_sql_result() if self.get_last_sql_result else None

        if last and last.get("success") and last.get("data"):
            data = last["data"]
            cols = last.get("columns", [])

            chart = None
            if self.classify_viz_intent and data:
                col_types = {}
                first = data[0]
                for c in cols:
                    v = first.get(c)
                    if isinstance(v, (int, float)):
                        col_types[c] = "number"
                    else:
                        col_types[c] = "string"

                chart = self.classify_viz_intent(
                    llm=self.llm,
                    data_summary=f"{len(cols)} columns: {', '.join(cols)}",
                    column_types=col_types,
                    row_count=len(data)
                )

            if self.clear_last_sql_result:
                self.clear_last_sql_result()

            return PipelineResult(
                success=True,
                response="Visualization suggestions:",
                session_id=session_id,
                execution_time=self._elapsed(start),
                sql_query=last.get("sql"),
                data=data,
                query_results=data,
                chart_suggestion=chart,
                reasoning_disclosure=disclose_visualization(has_data=True),
            )

        return PipelineResult(
            success=False,
            response="No data to visualize. Run a query first.",
            session_id=session_id,
            execution_time=self._elapsed(start),
            error="No data",
            reasoning_disclosure=disclose_visualization(has_data=False),
        )

    def _greeting_response(self, session_id: str, start: datetime) -> PipelineResult:
        return PipelineResult(
            success=True,
            response="""Hello! I'm **OptimaX** - your database query assistant.

**Try asking:**
- "Top 10 busiest airports"
- "Show me flights from JFK"
- "What's the average booking price?"

What would you like to explore?""",
            session_id=session_id,
            execution_time=self._elapsed(start),
            reasoning_disclosure=disclose_greeting(),
        )

    # =========================================================================
    # EXECUTION RESULT HELPERS (v6.1.1)
    # =========================================================================

    def _handle_execution_result(
        self,
        result: Dict[str, Any],
        accumulated: Any,
        session_id: str,
        start: datetime,
        context_disclosure: Optional[str] = None
    ) -> PipelineResult:
        """
        Handle the result from SQL execution.

        Centralizes success/failure handling to avoid code duplication.

        CRITICAL: Relational ambiguity should NEVER reach this function.
        If it does, this is a bug in the caller - but we handle it gracefully.
        """
        # Safety check: Relational ambiguity should have been handled earlier
        # This is defense-in-depth - ambiguity should NEVER appear as an error
        if (not result.get("success") and
                result.get("error_type") == "relational_ambiguity"):
            logger.warning(
                "[PIPELINE] BUG: Relational ambiguity reached _handle_execution_result. "
                "This should have been intercepted earlier."
            )
            # Return a generic message instead of "Query failed: Unknown error"
            return PipelineResult(
                success=True,  # Not a hard failure
                response=(
                    "I need more information to answer this query. "
                    "The question can be interpreted in multiple ways. "
                    "Please try rephrasing with more specific details."
                ),
                session_id=session_id,
                execution_time=self._elapsed(start),
                sql_query=result.get("sql"),
                clarification_needed=True,
                reasoning_disclosure="Multiple interpretations detected. Clarification required.",
            )

        if result["success"]:
            data = result["data"]
            sql = result["sql"]
            row_count = result["row_count"]

            # SGRD: Capture intent disclosure BEFORE clearing accumulated state
            intent_disclosure = disclose_for_intent_proceed(accumulated)

            # Cache for viz
            if self.set_last_sql_result:
                self.set_last_sql_result({
                    "success": True,
                    "data": data,
                    "columns": result["columns"],
                    "row_count": row_count,
                    "sql": sql
                })

            # Clear accumulated intent on success
            if accumulated:
                accumulated.clear()

            response = f"Query executed. Returned {row_count} row{'s' if row_count != 1 else ''}."
            if row_count > 0:
                response += "\n\n*Tip: Ask me to 'visualize this' for charts.*"

            # SGRD: Combine intent acknowledgement with execution confirmation
            exec_disclosure = disclose_execution_success(row_count)
            disclosure = f"{intent_disclosure} {exec_disclosure}"
            if context_disclosure:
                disclosure = f"{context_disclosure} {disclosure}"

            return PipelineResult(
                success=True,
                response=response,
                session_id=session_id,
                execution_time=self._elapsed(start),
                sql_query=sql,
                data=data,
                query_results=data,
                reasoning_disclosure=disclosure,
            )

        else:
            error_msg = result.get("error") or "Unknown error"

            # SGRD: Distinguish cost guard refusal from general failure
            if result.get("error_type") == "cost_guard":
                disclosure = disclose_for_cost_guard(result)
            else:
                disclosure = disclose_execution_failure()

            return PipelineResult(
                success=False,
                response=f"**Query failed:** {error_msg}",
                session_id=session_id,
                execution_time=self._elapsed(start),
                sql_query=result.get("sql"),
                error=error_msg,
                reasoning_disclosure=disclosure,
            )

    def _is_new_query_pattern(self, query: str) -> bool:
        """
        Check if query looks like a new query (vs ambiguity response).

        Used to determine if an unmatched response during ambiguity
        resolution should clear the ambiguity (user moved on) or
        re-ask the clarification (user was unclear).

        Patterns that indicate a NEW query:
        - Long queries (5+ words)
        - Starts with command words (show, list, find, etc.)
        - Contains question words (what, which, how)
        """
        q = query.lower().strip()
        words = q.split()

        # Long queries are likely new
        if len(words) >= 5:
            return True

        # Command words indicate new query
        new_query_starts = [
            "show", "list", "find", "get", "what", "which", "who", "where",
            "how", "give", "tell", "display", "fetch", "select"
        ]

        if words and words[0] in new_query_starts:
            return True

        # Very short responses (1-2 words) are likely clarifications
        if len(words) <= 2:
            return False

        return False

    def _elapsed(self, start: datetime) -> float:
        return (datetime.now() - start).total_seconds()
