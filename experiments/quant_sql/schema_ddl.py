"""
postgres_air Schema DDL — shared across all experiment scripts.

DDL format mirrors what LlamaIndex's SQLDatabase.get_table_info() produces
when SQLAlchemy introspects the postgres_air PostgreSQL database.  The
CREATE TABLE bodies are derived from SQLAlchemy's CreateTable() compilation
(types as PostgreSQL would report them) plus the 3-row sample block that
LlamaIndex appends by default (sample_rows_in_table_info=3).

FK_RELATIONSHIPS, SCHEMA_TABLES, and FK_JOIN_PAIRS are shared data structures
used by both run_experiment.py (prompt building) and evaluate_results.py
(structural validation).
"""

# ---------------------------------------------------------------------------
# Schema name
# ---------------------------------------------------------------------------
SCHEMA = "postgres_air"

# ---------------------------------------------------------------------------
# Per-table column sets — used by evaluator for column-existence checks
# ---------------------------------------------------------------------------
SCHEMA_TABLES: dict = {
    "airport": {
        "airport_code", "airport_name", "city", "coordinates",
    },
    "aircraft": {
        "aircraft_code", "model", "range", "class", "velocity",
    },
    "passenger": {
        "passenger_id", "first_name", "last_name", "email",
        "account_id", "age", "country",
    },
    "account": {
        "account_id", "login", "first_name", "last_name", "frequent_flyer_id",
    },
    "frequent_flyer": {
        "frequent_flyer_id", "first_name", "last_name", "title", "card_num",
        "level", "award_points", "email", "phone",
    },
    "flight": {
        "flight_id", "flight_no", "departure_airport", "arrival_airport",
        "scheduled_departure", "scheduled_arrival",
        "actual_departure", "actual_arrival",
        "aircraft_code", "status",
    },
    "booking": {
        "booking_id", "booking_ref", "booking_name", "account_id",
        "email", "phone", "price",
    },
    "booking_leg": {
        "booking_leg_id", "booking_id", "flight_id", "leg_num",
    },
    "boarding_pass": {
        "pass_id", "passenger_id", "booking_leg_id",
        "seat", "boarding_time", "precheck",
    },
    "phone": {
        "phone_id", "account_id", "phone", "phone_type", "primary_phone",
    },
}

# ---------------------------------------------------------------------------
# Foreign key relationships — single source of truth
# (source_table, source_col, target_table, target_col)
# ---------------------------------------------------------------------------
FK_RELATIONSHIPS: list = [
    ("passenger",     "account_id",        "account",        "account_id"),
    ("account",       "frequent_flyer_id", "frequent_flyer", "frequent_flyer_id"),
    ("flight",        "departure_airport", "airport",        "airport_code"),
    ("flight",        "arrival_airport",   "airport",        "airport_code"),
    ("flight",        "aircraft_code",     "aircraft",       "aircraft_code"),
    ("booking",       "account_id",        "account",        "account_id"),
    ("booking_leg",   "booking_id",        "booking",        "booking_id"),
    ("booking_leg",   "flight_id",         "flight",         "flight_id"),
    ("boarding_pass", "passenger_id",      "passenger",      "passenger_id"),
    ("boarding_pass", "booking_leg_id",    "booking_leg",    "booking_leg_id"),
    ("phone",         "account_id",        "account",        "account_id"),
]

# Build bidirectional join-pair set for O(1) evaluator lookups.
# Each element: "table_a.col_a||table_b.col_b" (canonical form)
def _build_fk_pairs() -> set:
    pairs: set = set()
    for src_t, src_c, tgt_t, tgt_c in FK_RELATIONSHIPS:
        pairs.add(f"{src_t}.{src_c}||{tgt_t}.{tgt_c}")
        pairs.add(f"{tgt_t}.{tgt_c}||{src_t}.{src_c}")
    return pairs

FK_JOIN_PAIRS: set = _build_fk_pairs()

# ---------------------------------------------------------------------------
# postgres_air DDL — SQLAlchemy CreateTable() output style
# Each block mirrors: str(CreateTable(table).compile(dialect=postgresql))
# plus LlamaIndex's 3-row sample block.
# ---------------------------------------------------------------------------
POSTGRES_AIR_DDL = """\
CREATE TABLE postgres_air.airport (
\tairport_code VARCHAR NOT NULL,
\tairport_name TEXT,
\tcity TEXT,
\tcoordinates TEXT,
\tPRIMARY KEY (airport_code)
)

/*3 rows from airport table:
airport_code\tairport_name\tcity\tcoordinates
ATL\tHartsfield-Jackson Atlanta International\tAtlanta\t(33.636719,-84.428101)
ORD\tO'Hare International Airport\tChicago\t(41.974162,-87.907321)
LAX\tLos Angeles International Airport\tLos Angeles\t(33.942536,-118.408074)
*/

CREATE TABLE postgres_air.aircraft (
\taircraft_code INTEGER NOT NULL,
\tmodel TEXT,
\trange INTEGER,
\tclass INTEGER,
\tvelocity NUMERIC,
\tPRIMARY KEY (aircraft_code)
)

/*3 rows from aircraft table:
aircraft_code\tmodel\trange\tclass\tvelocity
1\tBoeing 787-8\t15750\t3\t903.0
2\tAirbus A380-800\t15000\t4\t945.0
3\tBoeing 737-800\t5765\t2\t842.0
*/

CREATE TABLE postgres_air.account (
\taccount_id INTEGER NOT NULL,
\tlogin TEXT,
\tfirst_name TEXT,
\tlast_name TEXT,
\tfrequent_flyer_id INTEGER,
\tPRIMARY KEY (account_id),
\tFOREIGN KEY(frequent_flyer_id) REFERENCES postgres_air.frequent_flyer (frequent_flyer_id)
)

/*3 rows from account table:
account_id\tlogin\tfirst_name\tlast_name\tfrequent_flyer_id
1\tjsmith\tJohn\tSmith\t101
2\tamiller\tAlice\tMiller\t102
3\tbjones\tBob\tJones\tNone
*/

CREATE TABLE postgres_air.frequent_flyer (
\tfrequent_flyer_id INTEGER NOT NULL,
\tfirst_name TEXT,
\tlast_name TEXT,
\ttitle TEXT,
\tcard_num TEXT,
\tlevel INTEGER,
\taward_points INTEGER,
\temail TEXT,
\tphone TEXT,
\tPRIMARY KEY (frequent_flyer_id)
)

/*3 rows from frequent_flyer table:
frequent_flyer_id\tfirst_name\tlast_name\ttitle\tcard_num\tlevel\taward_points\temail\tphone
101\tJohn\tSmith\tMr\tFF10001\t3\t45200\tjsmith@example.com\t555-0101
102\tAlice\tMiller\tMs\tFF10002\t4\t102400\tamiller@example.com\t555-0102
103\tBob\tWilliams\tMr\tFF10003\t1\t8750\tbwilliams@example.com\t555-0103
*/

CREATE TABLE postgres_air.passenger (
\tpassenger_id INTEGER NOT NULL,
\tfirst_name TEXT,
\tlast_name TEXT,
\temail TEXT,
\taccount_id INTEGER,
\tage INTEGER,
\tcountry TEXT,
\tPRIMARY KEY (passenger_id),
\tFOREIGN KEY(account_id) REFERENCES postgres_air.account (account_id)
)

/*3 rows from passenger table:
passenger_id\tfirst_name\tlast_name\temail\taccount_id\tage\tcountry
1\tJohn\tSmith\tjsmith@example.com\t1\t34\tUS
2\tAlice\tMiller\tamiller@example.com\t2\t29\tUS
3\tCarlos\tGarcia\tcgarcia@example.com\t3\t41\tMX
*/

CREATE TABLE postgres_air.flight (
\tflight_id INTEGER NOT NULL,
\tflight_no TEXT,
\tdeparture_airport VARCHAR,
\tarrival_airport VARCHAR,
\tscheduled_departure TIMESTAMP WITH TIME ZONE,
\tscheduled_arrival TIMESTAMP WITH TIME ZONE,
\tactual_departure TIMESTAMP WITH TIME ZONE,
\tactual_arrival TIMESTAMP WITH TIME ZONE,
\taircraft_code INTEGER,
\tstatus TEXT,
\tPRIMARY KEY (flight_id),
\tFOREIGN KEY(departure_airport) REFERENCES postgres_air.airport (airport_code),
\tFOREIGN KEY(arrival_airport) REFERENCES postgres_air.airport (airport_code),
\tFOREIGN KEY(aircraft_code) REFERENCES postgres_air.aircraft (aircraft_code)
)

/*3 rows from flight table:
flight_id\tflight_no\tdeparture_airport\tarrival_airport\tscheduled_departure\tscheduled_arrival\tactual_departure\tactual_arrival\taircraft_code\tstatus
1\tPA101\tATL\tORD\t2020-08-07 12:00:00+00\t2020-08-07 14:00:00+00\t2020-08-07 12:10:00+00\t2020-08-07 14:15:00+00\t1\tArrived
2\tPA102\tORD\tJFK\t2020-08-07 15:00:00+00\t2020-08-07 18:00:00+00\t2020-08-07 15:05:00+00\t2020-08-07 18:20:00+00\t2\tArrived
3\tPA103\tJFK\tLAX\t2020-08-07 19:00:00+00\t2020-08-07 22:00:00+00\tNone\tNone\t3\tScheduled
*/

CREATE TABLE postgres_air.booking (
\tbooking_id INTEGER NOT NULL,
\tbooking_ref TEXT,
\tbooking_name TEXT,
\taccount_id INTEGER,
\temail TEXT,
\tphone TEXT,
\tprice NUMERIC,
\tPRIMARY KEY (booking_id),
\tFOREIGN KEY(account_id) REFERENCES postgres_air.account (account_id)
)

/*3 rows from booking table:
booking_id\tbooking_ref\tbooking_name\taccount_id\temail\tphone\tprice
1\tBK0001\tSmith/John\t1\tjsmith@example.com\t555-0101\t350.00
2\tBK0002\tMiller/Alice\t2\tamiller@example.com\t555-0102\t520.00
3\tBK0003\tGarcia/Carlos\t3\tcgarcia@example.com\t555-0199\t210.00
*/

CREATE TABLE postgres_air.booking_leg (
\tbooking_leg_id INTEGER NOT NULL,
\tbooking_id INTEGER,
\tflight_id INTEGER,
\tleg_num INTEGER,
\tPRIMARY KEY (booking_leg_id),
\tFOREIGN KEY(booking_id) REFERENCES postgres_air.booking (booking_id),
\tFOREIGN KEY(flight_id) REFERENCES postgres_air.flight (flight_id)
)

/*3 rows from booking_leg table:
booking_leg_id\tbooking_id\tflight_id\tleg_num
1\t1\t1\t1
2\t1\t2\t2
3\t2\t3\t1
*/

CREATE TABLE postgres_air.boarding_pass (
\tpass_id INTEGER NOT NULL,
\tpassenger_id INTEGER,
\tbooking_leg_id INTEGER,
\tseat TEXT,
\tboarding_time TIMESTAMP WITH TIME ZONE,
\tprecheck BOOLEAN,
\tPRIMARY KEY (pass_id),
\tFOREIGN KEY(passenger_id) REFERENCES postgres_air.passenger (passenger_id),
\tFOREIGN KEY(booking_leg_id) REFERENCES postgres_air.booking_leg (booking_leg_id)
)

/*3 rows from boarding_pass table:
pass_id\tpassenger_id\tbooking_leg_id\tseat\tboarding_time\tprecheck
1\t1\t1\t14A\t2020-08-07 11:30:00+00\tTrue
2\t2\t3\t22C\t2020-08-07 18:30:00+00\tFalse
3\t3\t2\t7B\t2020-08-07 14:30:00+00\tTrue
*/

CREATE TABLE postgres_air.phone (
\tphone_id INTEGER NOT NULL,
\taccount_id INTEGER,
\tphone TEXT,
\tphone_type TEXT,
\tprimary_phone BOOLEAN,
\tPRIMARY KEY (phone_id),
\tFOREIGN KEY(account_id) REFERENCES postgres_air.account (account_id)
)

/*3 rows from phone table:
phone_id\taccount_id\tphone\tphone_type\tprimary_phone
1\t1\t555-0101\tmobile\tTrue
2\t2\t555-0102\tmobile\tTrue
3\t1\t555-0111\thome\tFalse
*/
"""

# ---------------------------------------------------------------------------
# LlamaIndex NLSQLTableQueryEngine prompt template
# Matches llama_index/core/prompts/default_prompts.py DEFAULT_TEXT_TO_SQL_PROMPT
# ---------------------------------------------------------------------------
_LLAMAINDEX_TEMPLATE = """\
Given an input question, first create a syntactically correct {dialect} query \
to run, then look at the results of the query and return the answer. You can \
order the results by a relevant column to return the most interesting examples \
in the database.

Never query for all the columns from a specific table, only ask for a few \
relevant columns given the question.

Pay attention to use only the column names that you can see in the schema \
description. Be careful to not query for columns that do not exist. Also, pay \
attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}

Question: {query_str}
SQLQuery: \
"""


def build_prompt(nl_query: str, dialect: str = "postgresql") -> str:
    """
    Build the LlamaIndex NLSQLTableQueryEngine prompt wrapped in the
    Llama 3.1 Instruct chat template.

    The Llama 3.1 Instruct chat template format:
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

    LlamaIndex sends the text_to_sql template as the user message when calling
    the Groq API (which wraps it in the Llama 3.1 chat format on its side).
    We replicate that format for direct llama.cpp inference.
    """
    text_to_sql = _LLAMAINDEX_TEMPLATE.format(
        dialect=dialect,
        schema=POSTGRES_AIR_DDL,
        query_str=nl_query,
    )

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + text_to_sql
        + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt
