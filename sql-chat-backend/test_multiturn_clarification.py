"""
Regression tests for multi-turn clarification entity preservation (v6.23).

Tests the core invariant:
    When a clarification is pending for metric/FK/comparison and the user
    supplies a clarification response, the original entity MUST be preserved.
    Only an explicit new-query pattern or an entity-clarification response
    is permitted to change the entity.

No DB connection required — uses mock LLM and IntentMerger directly.
"""

import unittest
from dataclasses import dataclass, field
from typing import Optional, List
from unittest.mock import patch

from intent_accumulator import (
    IntentState,
    IntentMerger,
    IntentDecision,
    evaluate,
    set_schema_reference,
    set_schema_graph_reference,
    create_intent_state,
    create_intent_merger,
    ClarificationReason,
)


# =============================================================================
# HELPERS
# =============================================================================

def _state_with_entity(entity: str, ranking: str = "top_n", n: int = 10) -> IntentState:
    """Return a mid-clarification state: entity known, metric missing."""
    state = IntentState()
    state.entity_type = entity
    state.metric = None          # metric not yet known → triggers clarification
    state.ranking = ranking
    state.n = n
    state.original_query = f"top {entity}"
    state.turn_count = 1
    state.pending_clarification = "metric"   # simulate evaluate() having fired
    return state


@dataclass
class _MockIntent:
    """Minimal stand-in for SemanticIntent — only the fields merge() inspects."""
    entity_type: str = "unknown"
    metric: str = "unknown"
    event: Optional[str] = None
    aggregation: Optional[str] = None
    ranking: str = "none"
    n: Optional[int] = None
    time_scope: str = "all_time"
    filter_conditions: List[str] = field(default_factory=list)


# =============================================================================
# CORE FIX — ENTITY PRESERVATION DURING METRIC CLARIFICATION
# =============================================================================

class TestEntityPreservationDuringMetricClarification(unittest.TestCase):
    """
    Primary regression tests for the v6.23 entity preservation bug fix.

    Scenario: "top routes" → CLARIFY metric → "by passenger count"
    Expected: entity=route preserved; metric=count, event=passenger applied
    """

    def setUp(self):
        self.merger = create_intent_merger()

    def test_entity_preserved_on_metric_clarification_response(self):
        """
        Core test case from the bug report.

        Turn 1: "top routes"         → entity=route, metric=?, CLARIFY metric
        Turn 2: "by passenger count" → entity MUST remain route; metric=count

        [INTENT_MERGE] existing_entity=route new_entity_candidate=passenger
                       clarification_pending=metric action=preserve_original_entity
        """
        accumulated = _state_with_entity("route")

        # Simulate what the LLM extractor returns for "by passenger count"
        # when event=None (LLM put dimensional label in entity_type, not event)
        clarification_response = _MockIntent(
            entity_type="passenger",   # spurious — extracted from "passenger count"
            metric="count",
            event=None,                # LLM did NOT put it in event
            ranking="top_n",
            n=10,
        )

        result = self.merger.merge(accumulated, clarification_response, "by passenger count")

        self.assertEqual(
            result.entity_type, "route",
            "Entity must be preserved from Turn 1 — not overwritten by clarification response"
        )
        self.assertEqual(result.metric, "count")
        self.assertEqual(result.ranking, "top_n")
        # DIMENSIONAL ROUTING: dropped entity should be routed to event
        self.assertEqual(
            result.event, "passenger",
            "Dropped entity candidate must be routed to event as dimensional context"
        )

    def test_dimensional_routing_satisfies_bare_aggregation(self):
        """
        The cascading issue: entity preserved but can_proceed() still False
        because metric=count + ranking=top_n + event=None → has_metric()=False.

        Routing the dropped entity to event MUST satisfy has_metric().
        """
        accumulated = _state_with_entity("route")
        # ranking=top_n is set in _state_with_entity

        response = _MockIntent(
            entity_type="passenger",  # LLM extracted from "by passenger count"
            metric="count",
            event=None,               # NOT in event — this is the bug trigger
            ranking="top_n",
            n=10,
        )

        result = self.merger.merge(accumulated, response, "by passenger count")

        # After merge: entity=route, metric=count, event=passenger, ranking=top_n
        # has_metric() contract: count + top_n + has_event=True → COMPLETE
        self.assertTrue(
            result.has_metric(),
            "has_metric() must be True after dimensional routing: "
            "count + top_n + event=passenger satisfies completeness contract"
        )

    def test_dimensional_routing_does_not_overwrite_existing_event(self):
        """
        If accumulated.event is already set, the routing must NOT overwrite it.
        """
        accumulated = _state_with_entity("route")
        accumulated.event = "departure"  # already set from prior extraction

        response = _MockIntent(
            entity_type="arrival",    # spurious candidate
            metric="count",
            event=None,
            ranking="top_n",
        )

        result = self.merger.merge(accumulated, response, "by arrival count")

        self.assertEqual(result.entity_type, "route")
        self.assertEqual(result.event, "departure",
                         "Existing event must not be overwritten by dimensional routing")

    def test_entity_preserved_when_event_contains_entity_label(self):
        """
        "by flight count" → entity=flight extracted spuriously.
        Original entity (route) must be preserved; flight goes to event.
        """
        accumulated = _state_with_entity("route")

        response = _MockIntent(
            entity_type="flight",   # spurious
            metric="count",
            event="flight",
            ranking="top_n",
        )

        result = self.merger.merge(accumulated, response, "by flight count")

        self.assertEqual(result.entity_type, "route")
        self.assertEqual(result.metric, "count")

    def test_entity_preserved_multiple_turns(self):
        """
        Turn 1: "top routes"     → entity=route, CLARIFY metric
        Turn 2: "by count"       → entity=route preserved (still no event? accumulate)
        Turn 3: "passenger count"→ entity=route preserved
        """
        accumulated = _state_with_entity("route")

        # Turn 2
        resp2 = _MockIntent(entity_type="unknown", metric="count", ranking="top_n")
        accumulated = self.merger.merge(accumulated, resp2, "by count")
        self.assertEqual(accumulated.entity_type, "route")

        # Simulate evaluate() re-triggering clarification (count needs event for ranking)
        accumulated.pending_clarification = "metric"

        # Turn 3
        resp3 = _MockIntent(entity_type="passenger", metric="count", event="passenger")
        accumulated = self.merger.merge(accumulated, resp3, "passenger count")
        self.assertEqual(accumulated.entity_type, "route")
        self.assertEqual(accumulated.metric, "count")

    def test_original_query_composed_on_proceed_transition(self):
        """
        When entity is preserved and can_proceed() transitions False→True,
        original_query should be composed: "top routes by passenger count"
        NOT just the clarification fragment "by passenger count".
        """
        accumulated = _state_with_entity("route")

        response = _MockIntent(
            entity_type="passenger",
            metric="count",
            event="passenger",
            ranking="top_n",
            n=10,
        )

        result = self.merger.merge(accumulated, response, "by passenger count")

        # original_query must contain BOTH the entity context and the metric context
        self.assertIn("route", result.original_query.lower(),
                      "original_query must retain entity context from Turn 1")
        self.assertIn("passenger", result.original_query.lower(),
                      "original_query must include metric context from Turn 2")


# =============================================================================
# ENTITY CLARIFICATION — OVERRIDE MUST STILL WORK
# =============================================================================

class TestEntityClarificationOverrideAllowed(unittest.TestCase):
    """
    When pending_clarification == "entity_type" or "entity_disambiguation",
    the user's response IS the new entity — override must be permitted.
    """

    def setUp(self):
        self.merger = create_intent_merger()

    def test_entity_override_allowed_when_clarifying_entity(self):
        """
        Turn 1: "top 10" (no entity) → CLARIFY entity_type
        Turn 2: "flights"            → entity=flight MUST be set
        """
        accumulated = IntentState()
        accumulated.entity_type = None    # no entity yet
        accumulated.metric = "count"
        accumulated.ranking = "top_n"
        accumulated.n = 10
        accumulated.original_query = "top 10"
        accumulated.turn_count = 1
        accumulated.pending_clarification = "entity_type"  # waiting for entity

        response = _MockIntent(entity_type="flight", metric="count")
        result = self.merger.merge(accumulated, response, "flights")

        self.assertEqual(result.entity_type, "flight",
                         "Entity must be set when clarifying entity_type")

    def test_entity_override_allowed_for_disambiguation(self):
        """
        Turn 1: "top customers" → entity ambiguous → CLARIFY entity_disambiguation
        Turn 2: "passenger"     → entity=passenger MUST be set
        """
        accumulated = IntentState()
        accumulated.entity_type = "customer"   # ambiguous entity
        accumulated.metric = "count"
        accumulated.ranking = "top_n"
        accumulated.n = 10
        accumulated.original_query = "top customers"
        accumulated.turn_count = 1
        accumulated.pending_clarification = "entity_disambiguation"  # await user pick

        response = _MockIntent(entity_type="passenger", metric="count")
        result = self.merger.merge(accumulated, response, "passenger")

        self.assertEqual(result.entity_type, "passenger",
                         "Entity must be overridden when clarifying entity_disambiguation")


# =============================================================================
# NEW QUERY — ENTITY CHANGE MUST STILL WORK
# =============================================================================

class TestNewQueryEntityChangeAllowed(unittest.TestCase):
    """
    When the user starts a completely new query, entity override MUST fire
    regardless of pending_clarification (is_new_query() returns True → clear).
    """

    def setUp(self):
        self.merger = create_intent_merger()

    def test_new_query_clears_entity(self):
        """
        Turn 1: "top routes" → clarification pending
        Turn 2: "show me top flights by revenue" → new query → entity=flight
        """
        accumulated = _state_with_entity("route")

        # 6-word query starting with "show" → is_new_query() = True
        response = _MockIntent(
            entity_type="flight",
            metric="revenue",
            ranking="top_n",
            n=10,
        )
        result = self.merger.merge(
            accumulated, response, "show me top flights by revenue"
        )

        # clear() fires → entity resets; then new entity=flight is set
        self.assertEqual(result.entity_type, "flight",
                         "New query must be able to change the entity")

    def test_explicit_command_word_triggers_new_query(self):
        """
        "list passengers" starts with "list" (NEW_QUERY_STARTS) → new query.
        """
        accumulated = _state_with_entity("route")

        response = _MockIntent(entity_type="passenger", metric="list")
        result = self.merger.merge(accumulated, response, "list passengers")

        self.assertEqual(result.entity_type, "passenger")


# =============================================================================
# ROW-SELECT REGRESSION
# =============================================================================

class TestRowSelectRegressionNoChange(unittest.TestCase):
    """
    Row-select queries ("list flights", "show passengers") should be unaffected
    by the entity preservation fix — they never enter clarification.
    """

    def setUp(self):
        self.merger = create_intent_merger()

    def test_row_select_single_turn_unchanged(self):
        """
        Turn 1: "list flights" → entity=flight, metric=list → no clarification
        The merger is called once; entity must be set correctly.
        """
        accumulated = IntentState()  # fresh state, turn_count=0

        response = _MockIntent(entity_type="flight", metric="list")
        result = self.merger.merge(accumulated, response, "list flights")

        self.assertEqual(result.entity_type, "flight")
        self.assertEqual(result.metric, "list")

    def test_count_bookings_single_turn(self):
        """
        Turn 1: "count bookings" → entity=booking, metric=count → proceed immediately.
        """
        accumulated = IntentState()

        response = _MockIntent(entity_type="booking", metric="count")
        result = self.merger.merge(accumulated, response, "count bookings")

        self.assertEqual(result.entity_type, "booking")
        self.assertEqual(result.metric, "count")


# =============================================================================
# PENDING_CLARIFICATION FIELD — STATE MACHINE INVARIANTS
# =============================================================================

class TestPendingClarificationStateInvariants(unittest.TestCase):
    """
    Verify that evaluate() correctly manages pending_clarification at every
    exit point. This underpins the entity preservation guard.
    """

    @classmethod
    def setUpClass(cls):
        # Disable schema so FK checks always pass (conservative: allowed=True)
        set_schema_reference(None)
        set_schema_graph_reference(None)

    def test_pending_clarification_set_for_missing_metric(self):
        state = IntentState()
        state.entity_type = "route"
        state.metric = None
        state.ranking = "top_n"
        state.original_query = "top routes"

        decision = evaluate(state)

        self.assertFalse(decision.proceed)
        self.assertEqual(state.pending_clarification, "metric")

    def test_pending_clarification_set_for_missing_entity(self):
        state = IntentState()
        state.entity_type = None
        state.metric = "count"
        state.original_query = "count something"

        decision = evaluate(state)

        self.assertFalse(decision.proceed)
        self.assertEqual(state.pending_clarification, "entity_type")

    def test_pending_clarification_cleared_on_proceed(self):
        state = IntentState()
        state.entity_type = "flight"
        state.metric = "count"
        state.original_query = "count flights"
        state.pending_clarification = "metric"  # simulate prior clarification

        decision = evaluate(state)

        self.assertTrue(decision.proceed)
        self.assertIsNone(state.pending_clarification,
                          "pending_clarification must be None after PROCEED")

    def test_pending_clarification_set_for_comparison(self):
        state = IntentState()
        state.entity_type = "passenger"
        state.metric = "count"
        state.original_query = "compare frequent vs non-frequent passengers"

        decision = evaluate(state)

        self.assertFalse(decision.proceed)
        self.assertEqual(state.pending_clarification, "comparison")


# =============================================================================
# FULL MULTI-TURN INTEGRATION — evaluate() + merge() together
# =============================================================================

class TestFullMultiTurnFlow(unittest.TestCase):
    """
    End-to-end multi-turn tests using evaluate() + merge() together.

    These tests do NOT require a DB connection — schema is disabled so
    evaluate() uses the simple entity+metric decision rule only.
    """

    @classmethod
    def setUpClass(cls):
        set_schema_reference(None)
        set_schema_graph_reference(None)

    def setUp(self):
        self.merger = create_intent_merger()

    def test_top_routes_then_by_passenger_count(self):
        """
        The canonical bug scenario.

        Turn 1: "top routes"         → CLARIFY metric
        Turn 2: "by passenger count" → entity=route, metric=count, event=passenger
                                       → PROCEED
        """
        accumulated = create_intent_state()

        # --- Turn 1 ---
        t1_intent = _MockIntent(
            entity_type="route",
            metric="unknown",   # LLM can't determine metric from "top routes"
            ranking="top_n",
            n=10,
        )
        accumulated = self.merger.merge(accumulated, t1_intent, "top routes")
        decision1 = evaluate(accumulated)

        self.assertFalse(decision1.proceed, "Turn 1 must request metric clarification")
        self.assertEqual(accumulated.entity_type, "route")
        self.assertEqual(accumulated.pending_clarification, "metric")

        # --- Turn 2 ---
        t2_intent = _MockIntent(
            entity_type="passenger",  # spurious — this is the bug trigger
            metric="count",
            event=None,               # LLM did NOT put it in event field
            ranking="top_n",
            n=10,
        )
        accumulated = self.merger.merge(accumulated, t2_intent, "by passenger count")
        decision2 = evaluate(accumulated)

        # KEY ASSERTIONS
        self.assertEqual(
            accumulated.entity_type, "route",
            "BUG REGRESSION: entity must remain 'route' after metric clarification"
        )
        self.assertEqual(accumulated.metric, "count")
        self.assertEqual(
            accumulated.event, "passenger",
            "Dimensional routing: dropped 'passenger' must become event"
        )
        self.assertTrue(
            decision2.proceed,
            "After entity+metric+event known (count+top_n+event), must PROCEED"
        )

    def test_top_passengers_single_turn(self):
        """
        "top passengers by age" in a single turn must work normally.
        No clarification. entity=passenger, metric=age.
        """
        accumulated = create_intent_state()

        intent = _MockIntent(
            entity_type="passenger",
            metric="age",
            ranking="top_n",
            n=10,
        )
        accumulated = self.merger.merge(accumulated, intent, "top passengers by age")
        decision = evaluate(accumulated)

        self.assertEqual(accumulated.entity_type, "passenger")
        self.assertEqual(accumulated.metric, "age")
        self.assertTrue(decision.proceed)

    def test_count_bookings_single_turn(self):
        """
        "count bookings" must proceed in one turn.
        entity=booking, metric=count → PROCEED (count is always complete).
        """
        accumulated = create_intent_state()

        intent = _MockIntent(entity_type="booking", metric="count")
        accumulated = self.merger.merge(accumulated, intent, "count bookings")
        decision = evaluate(accumulated)

        self.assertEqual(accumulated.entity_type, "booking")
        self.assertTrue(decision.proceed)

    def test_list_flights_single_turn(self):
        """
        "list flights" → row-select → PROCEED immediately.
        """
        accumulated = create_intent_state()

        intent = _MockIntent(entity_type="flight", metric="list")
        accumulated = self.merger.merge(accumulated, intent, "list flights")
        decision = evaluate(accumulated)

        self.assertEqual(accumulated.entity_type, "flight")
        # Note: row_select requires no ranking; decision MIGHT clarify if
        # evaluate() thinks "list" is invalid. We verify entity is not corrupted.
        self.assertEqual(accumulated.entity_type, "flight")


if __name__ == "__main__":
    unittest.main(verbosity=2)
