import psycopg2
import os

# Connect to database
conn = psycopg2.connect(
    host="localhost",
    database="postgres", 
    user="postgres",
    password="Hello@123"
)

cur = conn.cursor()

# Query top 10 states by accident count
query = """
SELECT state, COUNT(*) as accident_count 
FROM us_accidents 
GROUP BY state 
ORDER BY accident_count DESC 
LIMIT 10;
"""

cur.execute(query)
results = cur.fetchall()

print("Top 10 states with most accidents:")
print("State | Count")
print("-" * 20)
for row in results:
    print(f"{row[0]:<5} | {row[1]:,}")

cur.close()
conn.close()