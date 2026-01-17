"""
Query Verification Script
Cross-verifies the SQL queries from OptimaX logs
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def run_query(sql, description):
    print(f"\n{'='*60}")
    print(f"QUERY: {description}")
    print(f"{'='*60}")
    print(f"SQL: {sql}")
    print("-"*60)

    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = list(result.keys())

        print(f"Columns: {columns}")
        print(f"Row count: {len(rows)}")
        print("-"*60)
        for i, row in enumerate(rows):
            print(f"{i+1}. {dict(row._mapping)}")

    return rows

# ============================================================
# TEST 1: Top 10 busiest flight routes (from logs - this looks correct)
# ============================================================
query1 = """
SELECT departure_airport, arrival_airport, COUNT(*) as num_flights
FROM postgres_air.flight
GROUP BY departure_airport, arrival_airport
ORDER BY num_flights DESC
LIMIT 10
"""
run_query(query1, "Top 10 busiest routes (original query from logs)")

# ============================================================
# TEST 2: Top 10 passengers with most points - PROBLEMATIC QUERY
# The log shows this complex join path:
# passenger → account → booking → phone → frequent_flyer (via phone match)
# ============================================================
query2_original = """
SELECT p.passenger_id, SUM(ff.award_points) AS total_points
FROM postgres_air.passenger p
JOIN postgres_air.account a ON p.account_id = a.account_id
JOIN postgres_air.booking b ON a.account_id = b.account_id
JOIN postgres_air.phone ph ON b.account_id = ph.account_id
JOIN postgres_air.frequent_flyer ff ON ph.phone = ff.phone
GROUP BY p.passenger_id
ORDER BY total_points DESC
LIMIT 10
"""
run_query(query2_original, "Top 10 passengers (ORIGINAL from logs - uses phone join)")

# ============================================================
# TEST 3: CORRECT QUERY using direct FK relationship
# Schema shows: account.frequent_flyer_id → frequent_flyer.frequent_flyer_id
# This is the PROPER join path!
# ============================================================
query2_correct = """
SELECT p.passenger_id, p.first_name, p.last_name, ff.award_points
FROM postgres_air.passenger p
JOIN postgres_air.account a ON p.account_id = a.account_id
JOIN postgres_air.frequent_flyer ff ON a.frequent_flyer_id = ff.frequent_flyer_id
ORDER BY ff.award_points DESC
LIMIT 10
"""
run_query(query2_correct, "Top 10 passengers (CORRECT - uses frequent_flyer_id FK)")

# ============================================================
# TEST 4: Check how many accounts have frequent_flyer_id populated
# ============================================================
query_check_fk = """
SELECT
    COUNT(*) as total_accounts,
    COUNT(frequent_flyer_id) as accounts_with_ff_id,
    COUNT(DISTINCT frequent_flyer_id) as unique_ff_ids
FROM postgres_air.account
"""
run_query(query_check_fk, "Account FK linkage check")

# ============================================================
# TEST 5: Check frequent flyers directly (without joins)
# ============================================================
query_ff_direct = """
SELECT frequent_flyer_id, first_name, last_name, award_points, level
FROM postgres_air.frequent_flyer
ORDER BY award_points DESC
LIMIT 10
"""
run_query(query_ff_direct, "Top 10 frequent flyers (direct from frequent_flyer table)")

# ============================================================
# TEST 6: Verify the phone-based join is unreliable
# Check how many phone matches exist between tables
# ============================================================
query_phone_check = """
SELECT COUNT(DISTINCT ph.phone) as phone_matches
FROM postgres_air.phone ph
JOIN postgres_air.frequent_flyer ff ON ph.phone = ff.phone
"""
run_query(query_phone_check, "Phone join reliability check")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
