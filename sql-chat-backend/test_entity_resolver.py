"""
Regression tests for schema-driven entity resolution (v6.15).

Tests resolve_entity() and internal helpers. No DB connection required
— uses mock schema.
"""

import unittest

from entity_resolver import (
    resolve_entity,
    EntityResolutionResult,
    EntityCandidate,
    _normalize,
    _depluralize,
    _extract_simple_name,
    _score_candidates,
)


# =============================================================================
# MOCK SCHEMA
# =============================================================================
# Airline schema with tables that exercise all resolution strategies:
#   - Simple names: passenger, booking, flight, aircraft
#   - Multi-token underscore: frequent_flyer
#   - Token-overlap target: customer_account (overlaps with "customer")
#
# This schema is intentionally designed to test ambiguity:
#   "customer" should match both customer_account and (via deplural overlap)
#   no exact match — triggering token overlap on customer_account.
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
            ],
            "foreign_keys": [],
        },
        "airline.frequent_flyer": {
            "columns": [
                {"name": "ff_id", "type": "INTEGER"},
                {"name": "passenger_id", "type": "INTEGER"},
                {"name": "miles", "type": "INTEGER"},
            ],
            "foreign_keys": [
                {
                    "fk_column": "passenger_id",
                    "target_table": "airline.passenger",
                    "target_column": "passenger_id",
                },
            ],
        },
        "airline.customer_account": {
            "columns": [
                {"name": "account_id", "type": "INTEGER"},
                {"name": "email", "type": "VARCHAR"},
            ],
            "foreign_keys": [],
        },
    }
}


# =============================================================================
# HELPER TESTS
# =============================================================================

class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_normalize("  Passenger  "), "passenger")

    def test_collapse_whitespace(self):
        self.assertEqual(_normalize("frequent   flyer"), "frequent flyer")


class TestDepluralize(unittest.TestCase):
    def test_simple_s(self):
        self.assertEqual(_depluralize("passengers"), "passenger")

    def test_ies(self):
        self.assertEqual(_depluralize("categories"), "category")

    def test_ses(self):
        self.assertEqual(_depluralize("buses"), "bus")

    def test_no_change(self):
        self.assertEqual(_depluralize("aircraft"), "aircraft")

    def test_short_word(self):
        self.assertEqual(_depluralize("as"), "as")

    def test_double_s(self):
        self.assertEqual(_depluralize("lass"), "lass")


class TestExtractSimpleName(unittest.TestCase):
    def test_qualified(self):
        self.assertEqual(_extract_simple_name("airline.passenger"), "passenger")

    def test_simple(self):
        self.assertEqual(_extract_simple_name("passenger"), "passenger")

    def test_multi_dot(self):
        self.assertEqual(_extract_simple_name("db.schema.table"), "table")


# =============================================================================
# RESOLVE ENTITY TESTS
# =============================================================================

class TestExactMatch(unittest.TestCase):
    """Strategy 1: Exact table name match."""

    def test_passenger_exact(self):
        result = resolve_entity("passenger", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.passenger")
        self.assertEqual(result.canonical_entity, "passenger")
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.candidates[0].match_strategy, "exact")

    def test_booking_exact(self):
        result = resolve_entity("booking", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.booking")

    def test_flight_exact(self):
        result = resolve_entity("flight", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.flight")

    def test_frequent_flyer_exact(self):
        """Multi-word table with underscores, matched exactly."""
        result = resolve_entity("frequent_flyer", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.frequent_flyer")
        self.assertEqual(result.confidence, 1.0)


class TestPluralSingular(unittest.TestCase):
    """Strategy 2: Plural/singular normalization."""

    def test_passengers_plural(self):
        result = resolve_entity("passengers", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.passenger")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.candidates[0].match_strategy, "plural_singular")

    def test_bookings_plural(self):
        result = resolve_entity("bookings", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.booking")

    def test_flights_plural(self):
        result = resolve_entity("flights", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.flight")


class TestMultiTokenUnderscore(unittest.TestCase):
    """Strategy 3: Multi-token to underscore matching."""

    def test_frequent_flyer_spaces(self):
        """'frequent flyer' → 'frequent_flyer'."""
        result = resolve_entity("frequent flyer", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.frequent_flyer")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.candidates[0].match_strategy, "multi_token_underscore")

    def test_customer_account_spaces(self):
        """'customer account' → 'customer_account'."""
        result = resolve_entity("customer account", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.customer_account")
        self.assertEqual(result.confidence, 0.9)


class TestMultiTokenPlural(unittest.TestCase):
    """Strategy 4: Multi-token + plural normalization."""

    def test_frequent_flyers_plural(self):
        """'frequent flyers' → 'frequent_flyer'."""
        result = resolve_entity("frequent flyers", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.frequent_flyer")
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.candidates[0].match_strategy, "multi_token_plural")

    def test_customer_accounts_plural(self):
        """'customer accounts' → 'customer_account'."""
        result = resolve_entity("customer accounts", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.customer_account")
        self.assertEqual(result.confidence, 0.85)


class TestTokenOverlap(unittest.TestCase):
    """Strategy 5: Token overlap matching."""

    def test_customer_overlaps_customer_account(self):
        """'customer' partially matches 'customer_account' via token overlap."""
        result = resolve_entity("customer", MOCK_SCHEMA)
        # Should find customer_account via token overlap
        self.assertGreaterEqual(len(result.candidates), 1)
        matching = [c for c in result.candidates if c.simple_name == "customer_account"]
        self.assertEqual(len(matching), 1)
        self.assertIn("token_overlap", matching[0].match_strategy)


class TestAmbiguous(unittest.TestCase):
    """Ambiguous entity resolution — multiple candidates above threshold."""

    def test_ambiguous_with_custom_schema(self):
        """Entity matching 2+ tables → ambiguous."""
        # Schema where "account" matches both account_billing and account_login
        ambig_schema = {
            "tables": {
                "db.account_billing": {
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "foreign_keys": [],
                },
                "db.account_login": {
                    "columns": [{"name": "id", "type": "INTEGER"}],
                    "foreign_keys": [],
                },
            }
        }
        result = resolve_entity("account", ambig_schema)
        self.assertEqual(result.status, "ambiguous")
        self.assertGreaterEqual(len(result.candidates), 2)
        names = {c.simple_name for c in result.candidates}
        self.assertIn("account_billing", names)
        self.assertIn("account_login", names)


class TestUnresolved(unittest.TestCase):
    """Unresolved entity — no matches."""

    def test_xyzzy_unresolved(self):
        result = resolve_entity("xyzzy", MOCK_SCHEMA)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.resolved_table)
        self.assertIsNone(result.canonical_entity)

    def test_completely_unknown(self):
        result = resolve_entity("zorblax_widget", MOCK_SCHEMA)
        self.assertEqual(result.status, "unresolved")


class TestEmptyInputs(unittest.TestCase):
    """Edge cases: empty/None inputs."""

    def test_empty_string(self):
        result = resolve_entity("", MOCK_SCHEMA)
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.raw_entity, "")

    def test_none_entity(self):
        result = resolve_entity(None, MOCK_SCHEMA)
        self.assertEqual(result.status, "unresolved")

    def test_whitespace_only(self):
        result = resolve_entity("   ", MOCK_SCHEMA)
        self.assertEqual(result.status, "unresolved")


class TestNoSchema(unittest.TestCase):
    """Edge cases: empty/None schema."""

    def test_none_schema(self):
        result = resolve_entity("passenger", None)
        self.assertEqual(result.status, "unresolved")

    def test_empty_schema(self):
        result = resolve_entity("passenger", {})
        self.assertEqual(result.status, "unresolved")

    def test_schema_no_tables(self):
        result = resolve_entity("passenger", {"tables": {}})
        self.assertEqual(result.status, "unresolved")


class TestCaseInsensitive(unittest.TestCase):
    """Case-insensitive matching."""

    def test_uppercase(self):
        result = resolve_entity("PASSENGER", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.passenger")

    def test_mixed_case(self):
        result = resolve_entity("Frequent Flyer", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.frequent_flyer")

    def test_title_case_plural(self):
        result = resolve_entity("Flights", MOCK_SCHEMA)
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_table, "airline.flight")


class TestConfidenceThreshold(unittest.TestCase):
    """Custom confidence threshold."""

    def test_high_threshold_filters_low_scores(self):
        """With threshold=0.95, only exact and plural matches qualify."""
        result = resolve_entity("passenger", MOCK_SCHEMA, confidence_threshold=0.95)
        self.assertEqual(result.status, "resolved")
        self.assertGreaterEqual(result.confidence, 0.95)

    def test_threshold_zero_accepts_all(self):
        """With threshold=0, even weak matches are accepted."""
        result = resolve_entity("customer", MOCK_SCHEMA, confidence_threshold=0.0)
        # Should find at least customer_account via token overlap
        self.assertGreater(len(result.candidates), 0)


class TestScoreCandidates(unittest.TestCase):
    """Direct tests for _score_candidates()."""

    def test_returns_sorted(self):
        candidates = _score_candidates("passenger", MOCK_SCHEMA)
        scores = [c.score for c in candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_schema(self):
        self.assertEqual(_score_candidates("passenger", {}), [])
        self.assertEqual(_score_candidates("passenger", None), [])

    def test_exact_match_score_1(self):
        candidates = _score_candidates("booking", MOCK_SCHEMA)
        exact = [c for c in candidates if c.match_strategy == "exact"]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
