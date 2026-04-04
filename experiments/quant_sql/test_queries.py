"""
Test query suite for the quantization experiment.

14 NL queries across 5 difficulty tiers, each with:
  - nl: the natural-language question sent to the model
  - tier: 1-5 difficulty tier
  - gold_sql: canonical correct SQL using declared FK paths
  - key_joins: FK pairs that MUST appear (or be equivalent) in a correct answer
  - notes: what this query tests

Tier 1  — Single table, no joins  (queries 1-3)
Tier 2  — Aggregation, no joins   (queries 4-6)
Tier 3  — Single JOIN             (queries 7-8)
Tier 4  — Multi-hop JOINs         (queries 9-11)
Tier 5  — Aggregation across JOIN (queries 12-14)

The KNOWN failure case is query 9:
  CORRECT path: passenger → account → frequent_flyer  (via account.frequent_flyer_id)
  WRONG path:   passenger → account → phone → frequent_flyer  (via phone string match)
"""

TEST_QUERIES: list = [
    # ── Tier 1: Single table, no joins ──────────────────────────────────────
    {
        "id": 1,
        "tier": 1,
        "nl": "Show 10 passengers",
        "gold_sql": (
            "SELECT passenger_id, first_name, last_name, email, account_id, age, country"
            " FROM postgres_air.passenger"
            " LIMIT 10"
        ),
        "key_joins": [],
        "notes": "Simplest possible query. Should always succeed at any quantization.",
    },
    {
        "id": 2,
        "tier": 1,
        "nl": "How many flights are there?",
        "gold_sql": (
            "SELECT COUNT(*) AS num_flights"
            " FROM postgres_air.flight"
        ),
        "key_joins": [],
        "notes": "Scalar aggregate, single table. No JOIN needed.",
    },
    {
        "id": 3,
        "tier": 1,
        "nl": "List all airports",
        "gold_sql": (
            "SELECT airport_code, airport_name, city"
            " FROM postgres_air.airport"
            " LIMIT 10"
        ),
        "key_joins": [],
        "notes": "Projection only. Tests whether model adds unnecessary schema prefix.",
    },

    # ── Tier 2: Aggregation, no joins ────────────────────────────────────────
    {
        "id": 4,
        "tier": 2,
        "nl": "Top 10 busiest departure airports by number of flights",
        "gold_sql": (
            "SELECT departure_airport, COUNT(*) AS num_flights"
            " FROM postgres_air.flight"
            " GROUP BY departure_airport"
            " ORDER BY num_flights DESC"
            " LIMIT 10"
        ),
        "key_joins": [],
        "notes": "GROUP BY + ORDER BY on single table. Tests aggregation structure.",
    },
    {
        "id": 5,
        "tier": 2,
        "nl": "What is the average booking price?",
        "gold_sql": (
            "SELECT AVG(price) AS avg_price"
            " FROM postgres_air.booking"
        ),
        "key_joins": [],
        "notes": "Simple aggregate. Model must know 'price' is in booking, not flight.",
    },
    {
        "id": 6,
        "tier": 2,
        "nl": "Top 10 busiest flight routes",
        "gold_sql": (
            "SELECT departure_airport, arrival_airport, COUNT(*) AS num_flights"
            " FROM postgres_air.flight"
            " GROUP BY departure_airport, arrival_airport"
            " ORDER BY num_flights DESC"
            " LIMIT 10"
        ),
        "key_joins": [],
        "notes": (
            "Multi-column GROUP BY. Tests whether model generates correct "
            "GROUP BY list matching the SELECT projection."
        ),
    },

    # ── Tier 3: Single JOIN ───────────────────────────────────────────────────
    {
        "id": 7,
        "tier": 3,
        "nl": "Show flights with their departure airport names",
        "gold_sql": (
            "SELECT f.flight_no, f.scheduled_departure, a.airport_name, a.city"
            " FROM postgres_air.flight f"
            " JOIN postgres_air.airport a ON f.departure_airport = a.airport_code"
            " LIMIT 10"
        ),
        "key_joins": [
            ("flight.departure_airport", "airport.airport_code"),
        ],
        "notes": (
            "Single FK join. Correct ON condition: "
            "flight.departure_airport = airport.airport_code."
        ),
    },
    {
        "id": 8,
        "tier": 3,
        "nl": "List bookings with the account login name",
        "gold_sql": (
            "SELECT b.booking_ref, b.price, a.login"
            " FROM postgres_air.booking b"
            " JOIN postgres_air.account a ON b.account_id = a.account_id"
            " LIMIT 10"
        ),
        "key_joins": [
            ("booking.account_id", "account.account_id"),
        ],
        "notes": "booking → account via account_id. Tests cross-entity single hop.",
    },

    # ── Tier 4: Multi-hop JOINs ──────────────────────────────────────────────
    {
        "id": 9,
        "tier": 4,
        "nl": "Top 10 passengers with the most frequent flyer points",
        "gold_sql": (
            "SELECT p.first_name, p.last_name, ff.award_points"
            " FROM postgres_air.passenger p"
            " JOIN postgres_air.account a ON p.account_id = a.account_id"
            " JOIN postgres_air.frequent_flyer ff ON a.frequent_flyer_id = ff.frequent_flyer_id"
            " ORDER BY ff.award_points DESC"
            " LIMIT 10"
        ),
        "key_joins": [
            ("passenger.account_id", "account.account_id"),
            ("account.frequent_flyer_id", "frequent_flyer.frequent_flyer_id"),
        ],
        "notes": (
            "KNOWN FAILURE CASE. Correct: passenger->account->frequent_flyer "
            "via account.frequent_flyer_id. "
            "WRONG: passenger->account->phone->frequent_flyer via phone string match. "
            "This is the primary test of relational reasoning degradation."
        ),
        "wrong_join_signal": "ph.phone = ff.phone",  # wrong path signature
    },
    {
        "id": 10,
        "tier": 4,
        "nl": "Show passenger names and their booked flight numbers",
        "gold_sql": (
            "SELECT p.first_name, p.last_name, f.flight_no"
            " FROM postgres_air.passenger p"
            " JOIN postgres_air.boarding_pass bp ON p.passenger_id = bp.passenger_id"
            " JOIN postgres_air.booking_leg bl ON bp.booking_leg_id = bl.booking_leg_id"
            " JOIN postgres_air.flight f ON bl.flight_id = f.flight_id"
            " LIMIT 10"
        ),
        "key_joins": [
            ("passenger.passenger_id", "boarding_pass.passenger_id"),
            ("boarding_pass.booking_leg_id", "booking_leg.booking_leg_id"),
            ("booking_leg.flight_id", "flight.flight_id"),
        ],
        "notes": (
            "3-hop chain: passenger->boarding_pass->booking_leg->flight. "
            "Tests whether model follows declared FK chain vs. inventing shortcuts."
        ),
    },
    {
        "id": 11,
        "tier": 4,
        "nl": "Which passengers have boarding passes for flights departing from JFK?",
        "gold_sql": (
            "SELECT p.first_name, p.last_name"
            " FROM postgres_air.passenger p"
            " JOIN postgres_air.boarding_pass bp ON p.passenger_id = bp.passenger_id"
            " JOIN postgres_air.booking_leg bl ON bp.booking_leg_id = bl.booking_leg_id"
            " JOIN postgres_air.flight f ON bl.flight_id = f.flight_id"
            " WHERE f.departure_airport = 'JFK'"
            " LIMIT 10"
        ),
        "key_joins": [
            ("passenger.passenger_id", "boarding_pass.passenger_id"),
            ("boarding_pass.booking_leg_id", "booking_leg.booking_leg_id"),
            ("booking_leg.flight_id", "flight.flight_id"),
        ],
        "notes": (
            "3-hop chain plus WHERE filter. Model must route through "
            "booking_leg, not attempt a direct passenger->flight shortcut."
        ),
    },

    # ── Tier 5: Aggregation across JOINs ────────────────────────────────────
    {
        "id": 12,
        "tier": 5,
        "nl": "Total booking revenue per departure airport",
        "gold_sql": (
            "SELECT f.departure_airport, SUM(b.price) AS total_revenue"
            " FROM postgres_air.booking b"
            " JOIN postgres_air.booking_leg bl ON b.booking_id = bl.booking_id"
            " JOIN postgres_air.flight f ON bl.flight_id = f.flight_id"
            " GROUP BY f.departure_airport"
            " ORDER BY total_revenue DESC"
            " LIMIT 10"
        ),
        "key_joins": [
            ("booking.booking_id", "booking_leg.booking_id"),
            ("booking_leg.flight_id", "flight.flight_id"),
        ],
        "notes": (
            "JOIN + SUM aggregate. Model must traverse booking->booking_leg->flight "
            "to reach departure_airport. GROUP BY must cover the airport column."
        ),
    },
    {
        "id": 13,
        "tier": 5,
        "nl": "Average booking price per frequent flyer level",
        "gold_sql": (
            "SELECT ff.level, AVG(b.price) AS avg_price"
            " FROM postgres_air.frequent_flyer ff"
            " JOIN postgres_air.account a ON ff.frequent_flyer_id = a.frequent_flyer_id"
            " JOIN postgres_air.booking b ON a.account_id = b.account_id"
            " GROUP BY ff.level"
            " ORDER BY ff.level"
            " LIMIT 10"
        ),
        "key_joins": [
            ("frequent_flyer.frequent_flyer_id", "account.frequent_flyer_id"),
            ("account.account_id", "booking.account_id"),
        ],
        "notes": (
            "Aggregation through the FF chain in reverse: "
            "frequent_flyer->account->booking. "
            "GROUP BY on ff.level. Tests backward FK traversal."
        ),
    },
    {
        "id": 14,
        "tier": 5,
        "nl": "How many boarding passes were issued per aircraft model?",
        "gold_sql": (
            "SELECT ac.model, COUNT(bp.pass_id) AS boarding_passes_issued"
            " FROM postgres_air.aircraft ac"
            " JOIN postgres_air.flight f ON ac.aircraft_code = f.aircraft_code"
            " JOIN postgres_air.booking_leg bl ON f.flight_id = bl.flight_id"
            " JOIN postgres_air.boarding_pass bp ON bl.booking_leg_id = bp.booking_leg_id"
            " GROUP BY ac.model"
            " ORDER BY boarding_passes_issued DESC"
            " LIMIT 10"
        ),
        "key_joins": [
            ("aircraft.aircraft_code", "flight.aircraft_code"),
            ("flight.flight_id", "booking_leg.flight_id"),
            ("booking_leg.booking_leg_id", "boarding_pass.booking_leg_id"),
        ],
        "notes": (
            "4-hop chain: aircraft->flight->booking_leg->boarding_pass. "
            "Hardest Tier 5 query. Tests deep multi-hop relational reasoning "
            "with aggregation at the top."
        ),
    },
]

# Convenience lookups
TIER_LABELS = {
    1: "Single table, no joins",
    2: "Aggregation, no joins",
    3: "Single JOIN",
    4: "Multi-hop JOINs",
    5: "Aggregation across JOINs",
}

BY_ID: dict = {q["id"]: q for q in TEST_QUERIES}


def get_queries_by_tier(tier: int) -> list:
    return [q for q in TEST_QUERIES if q["tier"] == tier]
