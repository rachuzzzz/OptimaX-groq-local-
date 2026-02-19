"""
Regression tests for centralized LIMIT enforcement (v6.13).

Tests enforce_query_bounds() and detect_unbounded_intent() — no DB required.
"""

import unittest
from sql_validator import enforce_query_bounds, detect_unbounded_intent


class TestEnforceQueryBounds(unittest.TestCase):
    """Tests for enforce_query_bounds() — the canonical LIMIT enforcement."""

    # --- LIMIT injection ---

    def test_single_table_gets_limit(self):
        result = enforce_query_bounds("SELECT * FROM flight", 50)
        self.assertIn("LIMIT 50", result.sql)
        self.assertTrue(result.limit_applied)
        self.assertIsNone(result.original_limit)

    def test_join_gets_limit(self):
        sql = "SELECT f.id, b.pax FROM flight f JOIN booking b ON f.id = b.flight_id"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 50", result.sql)
        self.assertTrue(result.limit_applied)

    def test_join_order_by_gets_limit_after_order(self):
        sql = "SELECT f.id, f.price FROM flight f JOIN booking b ON f.id = b.flight_id ORDER BY f.price DESC"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 50", result.sql)
        # LIMIT should come after ORDER BY
        order_pos = result.sql.upper().index("ORDER BY")
        limit_pos = result.sql.upper().index("LIMIT")
        self.assertGreater(limit_pos, order_pos)

    def test_group_by_gets_higher_limit_when_no_max(self):
        sql = "SELECT status, COUNT(*) FROM flight GROUP BY status"
        # Without explicit max_limit, GROUP BY queries get 100
        from sql_validator import SQLLimitEnforcer
        enforcer = SQLLimitEnforcer()
        result = enforcer.enforce_limit(sql)
        self.assertIn("LIMIT 100", result.sql)
        self.assertTrue(result.limit_applied)

    def test_group_by_with_max_limit(self):
        sql = "SELECT status, COUNT(*) FROM flight GROUP BY status"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 50", result.sql)

    def test_group_by_order_by_gets_limit(self):
        sql = "SELECT status, COUNT(*) AS c FROM flight GROUP BY status ORDER BY c DESC"
        result = enforce_query_bounds(sql, 100)
        self.assertIn("LIMIT 100", result.sql)
        order_pos = result.sql.upper().index("ORDER BY")
        limit_pos = result.sql.upper().index("LIMIT")
        self.assertGreater(limit_pos, order_pos)

    # --- Existing LIMIT preserved ---

    def test_existing_limit_within_bounds_unchanged(self):
        sql = "SELECT * FROM flight LIMIT 10"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 10", result.sql)
        self.assertFalse(result.limit_applied)
        self.assertEqual(result.original_limit, 10)
        self.assertFalse(result.was_capped)

    # --- LIMIT too high gets capped ---

    def test_limit_too_high_gets_capped(self):
        sql = "SELECT * FROM flight LIMIT 500"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 50", result.sql)
        self.assertNotIn("LIMIT 500", result.sql)
        self.assertTrue(result.limit_applied)
        self.assertEqual(result.original_limit, 500)
        self.assertTrue(result.was_capped)

    # --- Trailing semicolon ---

    def test_trailing_semicolon_limit_before(self):
        sql = "SELECT * FROM flight ORDER BY price;"
        result = enforce_query_bounds(sql, 50)
        self.assertIn("LIMIT 50", result.sql)
        # LIMIT should be present, semicolon stripped during enforcement
        self.assertNotIn("; LIMIT", result.sql)

    # --- Idempotency ---

    def test_idempotency(self):
        sql = "SELECT * FROM flight"
        result1 = enforce_query_bounds(sql, 50)
        result2 = enforce_query_bounds(result1.sql, 50)
        self.assertEqual(result1.sql, result2.sql)
        # Second call should be a no-op
        self.assertFalse(result2.limit_applied)

    # --- Edge cases ---

    def test_empty_sql(self):
        result = enforce_query_bounds("", 50)
        self.assertEqual(result.sql, "")
        self.assertFalse(result.limit_applied)

    def test_whitespace_only(self):
        result = enforce_query_bounds("   ", 50)
        self.assertFalse(result.limit_applied)

    def test_limit_equal_to_max_unchanged(self):
        sql = "SELECT * FROM flight LIMIT 50"
        result = enforce_query_bounds(sql, 50)
        self.assertFalse(result.was_capped)
        self.assertFalse(result.limit_applied)
        self.assertEqual(result.original_limit, 50)


class TestDetectUnboundedIntent(unittest.TestCase):
    """Tests for detect_unbounded_intent() — NL pattern detection."""

    def test_show_all_bookings(self):
        self.assertTrue(detect_unbounded_intent("show all bookings"))

    def test_list_all_flights(self):
        self.assertTrue(detect_unbounded_intent("list all flights"))

    def test_every_route(self):
        self.assertTrue(detect_unbounded_intent("show me every route"))

    def test_everything(self):
        self.assertTrue(detect_unbounded_intent("show me everything"))

    def test_entire_table(self):
        self.assertTrue(detect_unbounded_intent("give me the entire table"))

    def test_get_all(self):
        self.assertTrue(detect_unbounded_intent("get all records"))

    def test_cheapest_flight_not_unbounded(self):
        self.assertFalse(detect_unbounded_intent("cheapest flight"))

    def test_specific_query_not_unbounded(self):
        self.assertFalse(detect_unbounded_intent("flight from JFK to LAX"))

    def test_empty_string(self):
        self.assertFalse(detect_unbounded_intent(""))

    def test_none_input(self):
        self.assertFalse(detect_unbounded_intent(None))


if __name__ == "__main__":
    unittest.main()
