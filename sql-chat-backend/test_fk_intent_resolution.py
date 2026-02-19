"""
Regression tests for FK-aware intent resolution (v6.14).

Tests _is_metric_fk_reachable(), _check_fk_projection(), and the
evaluate() FK gate. No DB connection required — uses mock schema.
"""

import unittest
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from intent_accumulator import (
    IntentState,
    IntentDecision,
    evaluate,
    set_schema_reference,
    get_schema_reference,
    set_schema_graph_reference,
    get_schema_graph_reference,
    _resolve_metric_tables,
    _is_fk_reachable_via_schema,
    _is_metric_fk_reachable,
    _detect_referenced_tables,
    _check_fk_projection,
    MAX_FK_HOPS,
    ClarificationReason,
)


# =============================================================================
# MOCK SCHEMA
# =============================================================================
# Airline schema with clear FK chains:
#   passenger <-- booking --> flight --> aircraft
#
# FK connections (directed):
#   booking.passenger_id -> passenger.passenger_id
#   booking.flight_id    -> flight.flight_id
#   flight.aircraft_id   -> aircraft.aircraft_id
#
# Reachability within MAX_FK_HOPS=2:
#   passenger <-> booking  (1 hop)  ✓
#   booking   <-> flight   (1 hop)  ✓
#   flight    <-> aircraft  (1 hop)  ✓
#   passenger <-> flight   (2 hops) ✓
#   booking   <-> aircraft  (2 hops) ✓
#   passenger <-> aircraft  (3 hops) ✗
# =============================================================================

MOCK_SCHEMA = {
    "tables": {
        "airline.passenger": {
            "columns": [
                {"name": "passenger_id", "type": "INTEGER"},
                {"name": "first_name", "type": "VARCHAR"},
                {"name": "last_name", "type": "VARCHAR"},
                {"name": "age", "type": "INTEGER"},
            ],
            "foreign_keys": [],
        },
        "airline.booking": {
            "columns": [
                {"name": "booking_id", "type": "INTEGER"},
                {"name": "passenger_id", "type": "INTEGER"},
                {"name": "flight_id", "type": "INTEGER"},
                {"name": "price", "type": "DECIMAL"},
                {"name": "seat_number", "type": "VARCHAR"},
                {"name": "booking_ref", "type": "VARCHAR"},
                {"name": "status", "type": "VARCHAR"},
            ],
            "foreign_keys": [
                {
                    "fk_column": "passenger_id",
                    "target_table": "airline.passenger",
                    "target_column": "passenger_id",
                },
                {
                    "fk_column": "flight_id",
                    "target_table": "airline.flight",
                    "target_column": "flight_id",
                },
            ],
        },
        "airline.flight": {
            "columns": [
                {"name": "flight_id", "type": "INTEGER"},
                {"name": "flight_number", "type": "VARCHAR"},
                {"name": "aircraft_id", "type": "INTEGER"},
                {"name": "departure_airport", "type": "VARCHAR"},
                {"name": "arrival_airport", "type": "VARCHAR"},
                {"name": "status", "type": "VARCHAR"},
            ],
            "foreign_keys": [
                {
                    "fk_column": "aircraft_id",
                    "target_table": "airline.aircraft",
                    "target_column": "aircraft_id",
                },
            ],
        },
        "airline.aircraft": {
            "columns": [
                {"name": "aircraft_id", "type": "INTEGER"},
                {"name": "model", "type": "VARCHAR"},
                {"name": "manufacturer", "type": "VARCHAR"},
            ],
            "foreign_keys": [],
        },
    }
}


class TestResolveMetricTables(unittest.TestCase):
    """Tests for _resolve_metric_tables()."""

    def test_exact_column_match(self):
        tables = _resolve_metric_tables("price", MOCK_SCHEMA)
        self.assertIn("airline.booking", tables)

    def test_table_prefixed_match(self):
        tables = _resolve_metric_tables("booking_price", MOCK_SCHEMA)
        self.assertIn("airline.booking", tables)

    def test_no_match(self):
        tables = _resolve_metric_tables("nonexistent_column", MOCK_SCHEMA)
        self.assertEqual(tables, [])

    def test_column_in_multiple_tables(self):
        """'status' exists in both booking and flight."""
        tables = _resolve_metric_tables("status", MOCK_SCHEMA)
        self.assertIn("airline.booking", tables)
        self.assertIn("airline.flight", tables)
        self.assertEqual(len(tables), 2)

    def test_empty_metric(self):
        self.assertEqual(_resolve_metric_tables("", MOCK_SCHEMA), [])
        self.assertEqual(_resolve_metric_tables(None, MOCK_SCHEMA), [])

    def test_empty_schema(self):
        self.assertEqual(_resolve_metric_tables("price", {}), [])
        self.assertEqual(_resolve_metric_tables("price", None), [])


class TestFKReachabilityViaSchema(unittest.TestCase):
    """Tests for _is_fk_reachable_via_schema() (BFS over FK metadata)."""

    def test_same_table(self):
        self.assertTrue(
            _is_fk_reachable_via_schema(
                "airline.passenger", "airline.passenger", MOCK_SCHEMA
            )
        )

    def test_direct_fk_forward(self):
        """booking -> flight (forward FK)."""
        self.assertTrue(
            _is_fk_reachable_via_schema(
                "airline.booking", "airline.flight", MOCK_SCHEMA
            )
        )

    def test_direct_fk_reverse(self):
        """passenger -> booking (reverse FK: booking references passenger)."""
        self.assertTrue(
            _is_fk_reachable_via_schema(
                "airline.passenger", "airline.booking", MOCK_SCHEMA
            )
        )

    def test_two_hop_reachable(self):
        """passenger -> booking -> flight (2 hops)."""
        self.assertTrue(
            _is_fk_reachable_via_schema(
                "airline.passenger", "airline.flight", MOCK_SCHEMA, max_hops=2
            )
        )

    def test_three_hop_not_reachable_with_max_2(self):
        """passenger -> booking -> flight -> aircraft (3 hops, max=2)."""
        self.assertFalse(
            _is_fk_reachable_via_schema(
                "airline.passenger", "airline.aircraft", MOCK_SCHEMA, max_hops=2
            )
        )

    def test_three_hop_reachable_with_max_3(self):
        """passenger -> aircraft reachable if max_hops=3."""
        self.assertTrue(
            _is_fk_reachable_via_schema(
                "airline.passenger", "airline.aircraft", MOCK_SCHEMA, max_hops=3
            )
        )

    def test_no_schema(self):
        """No schema → returns True (don't block)."""
        self.assertTrue(
            _is_fk_reachable_via_schema("a", "b", None)
        )


class TestIsMetricFKReachable(unittest.TestCase):
    """Tests for _is_metric_fk_reachable() (unified entry point)."""

    @classmethod
    def setUpClass(cls):
        """Set schema reference for all tests."""
        set_schema_reference(MOCK_SCHEMA)
        # Clear schema_graph so we use the schema FK fallback
        set_schema_graph_reference(None)

    def test_same_table(self):
        self.assertTrue(
            _is_metric_fk_reachable("airline.passenger", "airline.passenger")
        )

    def test_passenger_to_booking(self):
        self.assertTrue(
            _is_metric_fk_reachable("airline.passenger", "airline.booking")
        )

    def test_booking_to_flight(self):
        self.assertTrue(
            _is_metric_fk_reachable("airline.booking", "airline.flight")
        )

    def test_flight_to_aircraft(self):
        self.assertTrue(
            _is_metric_fk_reachable("airline.flight", "airline.aircraft")
        )

    def test_passenger_to_aircraft_not_reachable(self):
        """3 hops exceeds MAX_FK_HOPS=2."""
        self.assertFalse(
            _is_metric_fk_reachable("airline.passenger", "airline.aircraft")
        )


class TestDetectReferencedTables(unittest.TestCase):
    """Tests for _detect_referenced_tables()."""

    def test_single_table_reference(self):
        tables = _detect_referenced_tables("show booking details", MOCK_SCHEMA)
        self.assertIn("airline.booking", tables)

    def test_multiple_table_references(self):
        tables = _detect_referenced_tables(
            "show passenger names with booking price", MOCK_SCHEMA
        )
        self.assertIn("airline.passenger", tables)
        self.assertIn("airline.booking", tables)

    def test_plural_table_name(self):
        tables = _detect_referenced_tables("list all bookings", MOCK_SCHEMA)
        self.assertIn("airline.booking", tables)

    def test_no_table_references(self):
        tables = _detect_referenced_tables("hello world", MOCK_SCHEMA)
        self.assertEqual(tables, [])

    def test_empty_query(self):
        self.assertEqual(_detect_referenced_tables("", MOCK_SCHEMA), [])


class TestCheckFKProjection(unittest.TestCase):
    """Tests for _check_fk_projection() — the centralized FK gate."""

    @classmethod
    def setUpClass(cls):
        set_schema_reference(MOCK_SCHEMA)
        set_schema_graph_reference(None)

    def _make_state(self, entity, metric, original_query=None, row_select=False):
        state = IntentState()
        state.entity_type = entity
        state.metric = metric
        state.original_query = original_query or f"show {entity} {metric}"
        state.row_select = row_select
        return state

    def test_same_table_metric(self):
        """passenger + age → same table → proceed."""
        state = self._make_state("passenger", "age")
        result = _check_fk_projection(state)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "same_table")

    def test_fk_reachable_metric(self):
        """passenger + price → booking.price FK-reachable → proceed."""
        state = self._make_state("passenger", "price")
        result = _check_fk_projection(state)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "fk_reachable")

    def test_fk_not_reachable_metric(self):
        """passenger + model → aircraft.model NOT reachable → block."""
        state = self._make_state("passenger", "model")
        result = _check_fk_projection(state)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "no_fk_path")

    def test_booking_to_flight_reachable(self):
        """booking + departure_airport → flight FK-reachable → proceed."""
        state = self._make_state("booking", "departure_airport")
        result = _check_fk_projection(state)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "fk_reachable")

    def test_multiple_fk_paths(self):
        """'status' in both booking and flight → ambiguous → block."""
        state = self._make_state("passenger", "status")
        result = _check_fk_projection(state)
        # Both booking and flight have 'status' and are reachable from passenger
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "multiple_fk_paths")
        self.assertTrue(len(result.reachable_tables) > 1)

    def test_row_select_nl_reachable(self):
        """Row-select with NL table ref that IS reachable → proceed."""
        state = self._make_state(
            "passenger", "show",
            original_query="show passenger names with booking price",
            row_select=True,
        )
        result = _check_fk_projection(state)
        self.assertTrue(result.allowed)

    def test_row_select_nl_not_reachable(self):
        """Row-select with NL table ref that is NOT reachable → block."""
        state = self._make_state(
            "passenger", "show",
            original_query="show passenger names with aircraft model",
            row_select=True,
        )
        result = _check_fk_projection(state)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "no_fk_path")

    def test_metric_not_a_column(self):
        """Metric that doesn't match any column → pass (don't block)."""
        state = self._make_state("passenger", "booking_count")
        result = _check_fk_projection(state)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "pass")

    def test_no_schema(self):
        """No schema → pass (don't block)."""
        original = get_schema_reference()
        set_schema_reference(None)
        try:
            state = self._make_state("passenger", "price")
            result = _check_fk_projection(state)
            self.assertTrue(result.allowed)
            self.assertEqual(result.reason, "no_schema")
        finally:
            set_schema_reference(original)


class TestEvaluateWithFKGate(unittest.TestCase):
    """Integration tests: evaluate() with FK reachability gate."""

    @classmethod
    def setUpClass(cls):
        set_schema_reference(MOCK_SCHEMA)
        set_schema_graph_reference(None)

    def _make_state(self, entity, metric, original_query=None):
        state = IntentState()
        state.entity_type = entity
        state.metric = metric
        state.original_query = original_query or f"show {entity} with {metric}"
        return state

    def test_passenger_with_booking_price_proceeds(self):
        """'Show passenger names with booking price' → proceed."""
        state = self._make_state(
            "passenger", "price",
            "show passenger names with booking price",
        )
        decision = evaluate(state)
        self.assertTrue(decision.proceed)

    def test_passenger_with_seat_number_proceeds(self):
        """'Show passenger names with seat number' → proceed."""
        state = self._make_state(
            "passenger", "seat_number",
            "show passenger names with seat number",
        )
        decision = evaluate(state)
        self.assertTrue(decision.proceed)

    def test_booking_with_flight_number_proceeds(self):
        """'Show booking reference with flight number' → proceed."""
        state = self._make_state(
            "booking", "flight_number",
            "show booking reference with flight number",
        )
        decision = evaluate(state)
        self.assertTrue(decision.proceed)

    def test_flight_with_aircraft_model_proceeds(self):
        """'Show flight number with aircraft model' → proceed."""
        state = self._make_state(
            "flight", "model",
            "show flight number with aircraft model",
        )
        decision = evaluate(state)
        self.assertTrue(decision.proceed)

    def test_passenger_with_aircraft_model_clarifies(self):
        """'Show passenger names with aircraft model' → clarify."""
        state = self._make_state(
            "passenger", "model",
            "show passenger names with aircraft model",
        )
        decision = evaluate(state)
        self.assertFalse(decision.proceed)
        self.assertIsNotNone(decision.context)
        self.assertEqual(
            decision.context.reason,
            ClarificationReason.FK_NOT_REACHABLE,
        )

    def test_ambiguous_entity_still_clarifies(self):
        """Entity with 2+ resolution candidates → clarify (ambiguous entity).

        v6.15: Ambiguity is now driven by entity_resolution_candidates
        (set by entity_resolver in the pipeline), not a static dict.
        """
        state = self._make_state("customer", "count", "show customer details")
        # Simulate entity_resolver finding 2+ candidates
        from entity_resolver import EntityCandidate
        state.entity_resolution_candidates = [
            EntityCandidate("airline.passenger", "passenger", 0.7, "token_overlap"),
            EntityCandidate("airline.account", "account", 0.7, "token_overlap"),
        ]
        decision = evaluate(state)
        self.assertFalse(decision.proceed)

    def test_missing_metric_still_clarifies(self):
        """Missing metric → clarify (existing behavior)."""
        state = IntentState()
        state.entity_type = "passenger"
        state.metric = None
        state.original_query = "something about passengers"
        decision = evaluate(state)
        self.assertFalse(decision.proceed)

    def test_same_table_metric_proceeds(self):
        """passenger + age (same table) → proceed."""
        state = self._make_state("passenger", "age", "show passenger age")
        decision = evaluate(state)
        self.assertTrue(decision.proceed)


if __name__ == "__main__":
    unittest.main()
