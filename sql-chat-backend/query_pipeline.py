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
from sql_validator import rewrite_limit, extract_limit
from sgrd import (
    disclose_greeting,
    disclose_visualization,
    disclose_for_intent_proceed,
    disclose_for_clarification,
    disclose_for_pending_ambiguity,
    disclose_for_cost_guard,
    disclose_for_cost_limit_rewrite,
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

    # v6.16: Maximum user-specified LIMIT for cost guard refinement
    MAX_COST_GUARD_LIMIT = 1000

    def __init__(
        self,
        llm: Any,
        intent_extractor: Any,
        intent_merger: Any,
        execute_fn: Any,
        execute_sql_fn: Any = None,
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
        self.execute_sql = execute_sql_fn  # v6.16: Direct SQL execution (no LLM)
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

            # =================================================================
            # STEP 0B: CHECK FOR PENDING COST GUARD (v6.16)
            # =================================================================
            # If the previous query was blocked by the cost guard, check if
            # the user's response is a numeric LIMIT or a new query.
            #
            # FLOW:
            # 1. Check if accumulated has pending cost guard
            # 2. Parse user response as integer
            # 3. If valid integer (1..MAX_COST_GUARD_LIMIT):
            #    a. Rewrite LIMIT on stored SQL (no LLM call)
            #    b. Execute directly via execute_sql_fn
            #    c. Clear cost guard state
            # 4. If new query pattern -> clear cost guard, fall through
            # 5. If invalid -> re-ask for numeric input
            #
            # CRITICAL: This path executes STORED SQL with a rewritten LIMIT.
            # No LLM call. No NL-SQL generation. No intent extraction.
            # =================================================================
            if accumulated and accumulated.has_pending_cost_guard():
                pending_cg = accumulated.pending_cost_guard
                logger.info(
                    f"[PIPELINE] Pending cost guard detected: "
                    f"{pending_cg.estimated_rows:,} rows > {pending_cg.threshold:,}"
                )

                parsed_limit = self._parse_numeric_limit(query)

                if parsed_limit is not None:
                    # Validate range
                    if parsed_limit < 1 or parsed_limit > self.MAX_COST_GUARD_LIMIT:
                        logger.info(
                            f"[PIPELINE] Numeric limit out of range: {parsed_limit} "
                            f"(valid: 1-{self.MAX_COST_GUARD_LIMIT})"
                        )
                        return PipelineResult(
                            success=True,
                            response=(
                                f"Please enter a number between 1 and "
                                f"{self.MAX_COST_GUARD_LIMIT:,}."
                            ),
                            session_id=session_id,
                            execution_time=self._elapsed(start),
                            clarification_needed=True,
                            reasoning_disclosure=disclose_for_cost_guard({
                                "estimated_rows": pending_cg.estimated_rows,
                                "error_type": "cost_guard",
                            }),
                        )

                    # Rewrite LIMIT on stored SQL
                    rewritten_sql = rewrite_limit(
                        pending_cg.original_sql, parsed_limit
                    )
                    logger.info(
                        f"[PIPELINE] Cost guard resolved: LIMIT {parsed_limit} "
                        f"(was {pending_cg.original_limit})"
                    )

                    # Clear cost guard BEFORE execution
                    accumulated.clear_cost_guard()
                    session["accumulated_intent"] = accumulated

                    # Execute directly — no LLM, no NL-SQL
                    result = await self.execute_sql(
                        sql=rewritten_sql,
                        row_limit=parsed_limit,
                        timeout=45.0,
                    )

                    disclosure = disclose_for_cost_limit_rewrite(
                        pending_cg.estimated_rows, parsed_limit
                    )
                    return self._handle_execution_result(
                        result, accumulated, session_id, start,
                        context_disclosure=disclosure,
                    )

                elif self._is_new_query_pattern(query):
                    # User provided a new/refined query — clear cost guard, process normally
                    logger.info("[PIPELINE] New query detected — clearing cost guard")
                    accumulated.clear_cost_guard()
                    session["accumulated_intent"] = accumulated
                    # Fall through to normal extraction

                else:
                    # Response is not numeric and not a new query — re-ask
                    logger.info("[PIPELINE] Cost guard response not numeric — re-asking")
                    return PipelineResult(
                        success=True,
                        response=self._format_cost_guard_numeric_prompt(pending_cg),
                        session_id=session_id,
                        execution_time=self._elapsed(start),
                        clarification_needed=True,
                        reasoning_disclosure=disclose_for_cost_guard({
                            "estimated_rows": pending_cg.estimated_rows,
                            "error_type": "cost_guard",
                        }),
                    )

            # Extract intent (one-shot, no decisions)
            extraction = self.extractor.extract(query)

            if extraction.success and extraction.intent:
                logger.info(
                    f"[PIPELINE] Extracted: entity={extraction.intent.entity_type}, "
                    f"metric={extraction.intent.metric}"
                )

                # =============================================================
                # ALSR PHASE 1 — entity_type composite phrase decomposition (v6.22)
                # =============================================================
                # When entity_type is a composite phrase (e.g., "aircraft model"),
                # ALSR splits it into entity part ("aircraft") and attribute part
                # ("model"), resolves each against the live schema, and rewrites
                # entity_type to the canonical entity table name ("aircraft_type").
                # The attribute binding is stored for group_by_targets injection
                # AFTER merge (merge may call clear() which would wipe it).
                #
                # Guard: only runs on multi-token entity_type strings.
                # Passthrough: if the full phrase is a table, ALSR skips it
                #              and entity_resolver handles it below.
                # =============================================================
                _alsr_entity_binding = None
                if (
                    extraction.intent.entity_type
                    and " " in extraction.intent.entity_type
                    and extraction.intent.entity_type != "unknown"
                ):
                    try:
                        from semantic_attribute_resolver import resolve_attribute_phrase
                        from intent_accumulator import get_schema_reference as _get_schema
                        _alsr_schema = _get_schema()
                        if _alsr_schema:
                            _alsr_result = resolve_attribute_phrase(
                                extraction.intent.entity_type, _alsr_schema
                            )
                            if _alsr_result.status == "resolved" and _alsr_result.entity_override:
                                logger.info(
                                    f"[PIPELINE] ALSR entity decompose: "
                                    f"'{extraction.intent.entity_type}' → "
                                    f"entity='{_alsr_result.entity_override}', "
                                    f"attr='{_alsr_result.binding.qualified}'"
                                )
                                extraction.intent.entity_type = _alsr_result.entity_override
                                _alsr_entity_binding = _alsr_result.binding
                            elif _alsr_result.status == "ambiguous":
                                logger.info(
                                    f"[PIPELINE] ALSR entity ambiguous: "
                                    f"'{extraction.intent.entity_type}' → "
                                    f"{[b.qualified for b in _alsr_result.candidates]}"
                                )
                                # Leave entity_type as-is; entity_resolver handles ambiguity below
                            else:
                                logger.debug(
                                    f"[PIPELINE] ALSR entity {_alsr_result.status}: "
                                    f"'{extraction.intent.entity_type}' ({_alsr_result.reason})"
                                )
                    except ImportError:
                        logger.debug("[PIPELINE] ALSR not available (semantic_attribute_resolver not installed)")

                # =============================================================
                # ENTITY RESOLUTION — schema-driven (v6.15)
                # =============================================================
                # Resolve free-form entity names (e.g., "customer", "flights")
                # to actual schema table names BEFORE merge. This replaces
                # hardcoded AMBIGUOUS_SEMANTIC_ALIASES with live schema matching.
                # =============================================================
                if extraction.intent.entity_type and extraction.intent.entity_type != "unknown":
                    from entity_resolver import resolve_entity
                    from intent_accumulator import get_schema_reference

                    schema = get_schema_reference()
                    if schema:
                        resolution = resolve_entity(
                            extraction.intent.entity_type,
                            schema,
                        )
                        if resolution.status == "resolved":
                            logger.info(
                                f"[PIPELINE] Entity resolved: "
                                f"'{extraction.intent.entity_type}' → "
                                f"'{resolution.canonical_entity}'"
                            )
                            extraction.intent.entity_type = resolution.canonical_entity
                        elif resolution.status == "ambiguous":
                            logger.info(
                                f"[PIPELINE] Entity ambiguous: "
                                f"'{extraction.intent.entity_type}' → "
                                f"{[c.simple_name for c in resolution.candidates]}"
                            )
                            # Store candidates on accumulated state for clarification
                            if accumulated:
                                accumulated.entity_resolution_candidates = resolution.candidates
                                session["accumulated_intent"] = accumulated
                        # "unresolved" → leave entity_type as-is, let accumulator handle

                # =============================================================
                # ALSR PHASE 2 — metric / event attribute binding (v6.22)
                # =============================================================
                # When metric or event contain composite phrases (e.g., "booking
                # price"), ALSR resolves them to qualified column references
                # (e.g., "booking.price"). The original phrase is preserved for
                # the LLM (it still needs to determine the aggregation type).
                # Bindings are applied to accumulated state AFTER merge to avoid
                # being wiped by IntentMerger.merge() → accumulated.clear().
                # =============================================================
                _alsr_metric_binding = None
                _alsr_event_binding = None
                try:
                    from semantic_attribute_resolver import resolve_attribute_phrase as _alsr_resolve
                    from intent_accumulator import get_schema_reference as _get_schema2
                    _alsr_schema2 = _get_schema2()
                    if _alsr_schema2:
                        if (
                            extraction.intent.metric
                            and " " in extraction.intent.metric
                            and extraction.intent.metric != "unknown"
                        ):
                            _m_result = _alsr_resolve(extraction.intent.metric, _alsr_schema2)
                            if _m_result.status == "resolved":
                                _alsr_metric_binding = _m_result.binding
                                logger.info(
                                    f"[PIPELINE] ALSR metric binding: "
                                    f"'{extraction.intent.metric}' → "
                                    f"'{_m_result.binding.qualified}'"
                                )
                        if (
                            extraction.intent.event
                            and " " in extraction.intent.event
                            and extraction.intent.event not in ("unknown", "none", "")
                        ):
                            _e_result = _alsr_resolve(extraction.intent.event, _alsr_schema2)
                            if _e_result.status == "resolved":
                                _alsr_event_binding = _e_result.binding
                                logger.info(
                                    f"[PIPELINE] ALSR event binding: "
                                    f"'{extraction.intent.event}' → "
                                    f"'{_e_result.binding.qualified}'"
                                )
                except ImportError:
                    pass  # ALSR not available; Phase 1 guard already logged this

                # =============================================================
                # ALSR PHASE 3 — tail-position grouping phrase (v6.22+)
                # =============================================================
                # Detect structural grouping patterns (per / grouped by /
                # group by / by) in the original NL query and route the
                # extracted phrase through ALSR for schema-grounded resolution.
                #
                # This captures dimensions like "per aircraft model" that appear
                # in tail position and are syntactically distinct from the
                # entity_type / metric / event fields the LLM extractor saw.
                # ALSR Phase 1 and Phase 2 do not fire for these phrases because
                # entity_type = "flights" (no whitespace) and metric / event
                # do not contain the grouping phrase.
                #
                # Guard: detection is deterministic (no LLM calls); ALSR
                # resolution is schema-driven (no heuristic guessing).
                # No decision logic is modified. has_metric() evaluates the
                # populated group_by_targets under the existing completeness
                # contract — this is purely an extraction-layer extension.
                # =============================================================
                _alsr_grouping_binding = None
                try:
                    from grouping_phrase_detector import extract_grouping_phrase
                    _grouping_phrase = extract_grouping_phrase(query)
                    if _grouping_phrase:
                        logger.info(
                            f"[GROUPING] Detected grouping phrase: '{_grouping_phrase}'"
                        )
                        from semantic_attribute_resolver import (
                            resolve_attribute_phrase as _alsr_resolve_grp,
                        )
                        from intent_accumulator import (
                            get_schema_reference as _get_schema_grp,
                        )
                        _alsr_schema_grp = _get_schema_grp()
                        if _alsr_schema_grp:
                            _g_result = _alsr_resolve_grp(
                                _grouping_phrase, _alsr_schema_grp
                            )
                            if _g_result.status == "resolved":
                                _alsr_grouping_binding = _g_result.binding
                                logger.info(
                                    f"[GROUPING] ALSR resolved grouping → "
                                    f"{_g_result.binding.qualified}"
                                )
                            else:
                                logger.info(
                                    f"[GROUPING] Phrase detected but ALSR "
                                    f"unresolved ({_g_result.status}) → "
                                    f"no binding applied"
                                )
                except ImportError:
                    logger.debug(
                        "[GROUPING] grouping_phrase_detector not available"
                    )

                # Merge into accumulated state
                if accumulated and self.merger:
                    accumulated = self.merger.merge(accumulated, extraction.intent, query)
                    session["accumulated_intent"] = accumulated

                    # =========================================================
                    # ALSR: Apply bindings to accumulated state (v6.22)
                    # =========================================================
                    # Bindings are applied AFTER merge so that merge's optional
                    # clear() cannot wipe them. Bindings are additive — they
                    # accumulate across clarification turns until clear() is
                    # called on success.
                    # =========================================================
                    _alsr_any = False
                    if _alsr_entity_binding:
                        accumulated.attribute_bindings.append(_alsr_entity_binding)
                        accumulated.group_by_targets.append(_alsr_entity_binding.qualified)
                        _alsr_any = True
                    if _alsr_metric_binding:
                        accumulated.attribute_bindings.append(_alsr_metric_binding)
                        _alsr_any = True
                    if _alsr_event_binding:
                        accumulated.attribute_bindings.append(_alsr_event_binding)
                        _alsr_any = True
                    if _alsr_grouping_binding:
                        accumulated.attribute_bindings.append(_alsr_grouping_binding)
                        accumulated.group_by_targets.append(_alsr_grouping_binding.qualified)
                        _alsr_any = True
                    if _alsr_any:
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
            # ALSR HINT INJECTION (v6.22 audit fix — Invariant 3)
            # =================================================================
            # Enrich query_for_nlsql with ALSR-resolved column mappings so the
            # NL-SQL engine knows which specific columns were identified.
            # Without this injection, ALSR's bindings exist only in IntentState
            # and the LLM must rediscover column references independently.
            #
            # Pattern: same as semantic role hint (get_fk_hint_for_query) and
            # structured context binding — append deterministic text to query.
            #
            # Runs ONLY when ALSR produced bindings for this turn (attribute_bindings
            # non-empty). No-op on every non-ALSR path (zero overhead).
            # =================================================================
            if (
                accumulated
                and getattr(accumulated, 'attribute_bindings', None)
            ):
                try:
                    from semantic_attribute_resolver import format_alsr_query_hint
                    query_for_nlsql = format_alsr_query_hint(
                        query_for_nlsql,
                        getattr(accumulated, 'group_by_targets', []),
                        accumulated.attribute_bindings,
                    )
                    logger.info(
                        f"[PIPELINE] ALSR hint injected: "
                        f"bindings={[getattr(b, 'qualified', '?') for b in accumulated.attribute_bindings]}"
                    )
                except ImportError:
                    pass  # ALSR not available; already logged in Phase 1

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
            # COST GUARD INTERCEPT (v6.16 - Conversational Refinement)
            # =================================================================
            # If the cost guard blocked execution, enter a clarification loop
            # instead of returning a hard failure. The user is offered options:
            # add filters, apply a strict limit, or execute anyway.
            #
            # FLOW:
            # 1. Detect error_type == "cost_guard"
            # 2. Store cost guard state on accumulated intent
            # 3. Return conversational clarification (NOT an error)
            #
            # CRITICAL: Cost guard is NEVER a hard failure. It is always
            # a refinement opportunity, following the same pattern as
            # relational ambiguity.
            # =================================================================
            if (not result["success"] and
                    result.get("error_type") == "cost_guard"):

                estimated_rows = result.get("estimated_rows", 0)
                threshold = result.get("threshold", 100000)
                validated_sql = result.get("sql", "")
                original_limit = extract_limit(validated_sql)
                logger.info(
                    f"[PIPELINE] Cost guard triggered: "
                    f"{estimated_rows:,} estimated rows > {threshold:,} threshold, "
                    f"original_limit={original_limit}"
                )

                # Create accumulated state if it doesn't exist (edge case recovery)
                if not accumulated:
                    from intent_accumulator import create_intent_state
                    accumulated = create_intent_state()
                    logger.info("[PIPELINE] Created accumulated state for cost guard")

                # Store validated SQL for direct re-execution (no LLM needed)
                accumulated.set_pending_cost_guard(
                    estimated_rows=estimated_rows,
                    threshold=threshold,
                    original_sql=validated_sql,
                    original_limit=original_limit,
                )
                session["accumulated_intent"] = accumulated

                # Ask user for numeric LIMIT
                response = self._format_cost_guard_numeric_prompt(
                    accumulated.pending_cost_guard
                )

                # CRITICAL: Return as clarification, NOT as error
                return PipelineResult(
                    success=True,  # Not a failure — just need refinement
                    response=response,
                    session_id=session_id,
                    execution_time=self._elapsed(start),
                    sql_query=validated_sql,
                    clarification_needed=True,
                    reasoning_disclosure=disclose_for_cost_guard(result),
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

    # =========================================================================
    # COST GUARD HELPERS (v6.16 - Numeric LIMIT Refinement)
    # =========================================================================

    def _parse_numeric_limit(self, query: str) -> Optional[int]:
        """
        Parse a user response as a numeric LIMIT value.

        Returns:
            int if the response is a pure integer, None otherwise.

        Accepted formats:
            "10", " 25 ", "100"
        Rejected:
            "ten", "10 rows", "show 10", "abc", ""
        """
        q = query.strip()
        if q.isdigit():
            return int(q)
        return None

    def _format_cost_guard_numeric_prompt(self, pending: Any) -> str:
        """
        Format the cost guard clarification as a numeric LIMIT prompt.

        Asks the user how many rows they want to display.
        """
        return (
            f"This query may scan approximately **{pending.estimated_rows:,} rows**, "
            f"which could be slow or resource-intensive.\n\n"
            f"How many rows would you like to display?\n"
            f"Please enter a number (e.g., 10, 25, 100).\n\n"
            f"You can also rephrase your query with more specific criteria."
        )

    def _elapsed(self, start: datetime) -> float:
        return (datetime.now() - start).total_seconds()
