"""
Test Query Governance Layer
============================

Demonstrates the rule-based analytical complexity classifier.
Run this to verify the governance logic works correctly.
"""

from main import classify_query_complexity, generate_governance_clarification


def test_simple_queries():
    """Test queries that should pass through to SQL agent (1 or 0 signal categories)"""
    print("=" * 80)
    print("SIMPLE QUERIES (Should execute SQL)")
    print("=" * 80)

    simple_queries = [
        "Show me all flights",
        "Count total bookings",
        "List airports in California",
        "What is the average ticket price?",
        "Show me the top 10 routes",  # Only ranking signal
        "Find flights in the last 30 days",  # Only time window signal
    ]

    for query in simple_queries:
        result = classify_query_complexity(query)
        print(f"\nQuery: {query}")
        print(f"  Analytical: {result['is_analytical']}")
        print(f"  Categories ({len(result['signal_categories'])}): {result['signal_categories']}")
        print(f"  Total Signals: {result['signal_count']}")
        if result['detected_signals']:
            for cat, sigs in result['detected_signals'].items():
                print(f"    - {cat}: {sigs}")


def test_analytical_queries():
    """Test queries that should be governed (2+ signal categories)"""
    print("\n" + "=" * 80)
    print("ANALYTICAL QUERIES (Should be governed - NO SQL execution)")
    print("=" * 80)

    analytical_queries = [
        # From the prompt examples
        "Identify our top 20 most valuable customers by total booking value, show their average booking frequency, preferred routes, and whether they're frequent flyers. Flag any VIPs who haven't booked in the last 90 days.",
        "Segment passengers into three groups: frequent flyers (3+ bookings), occasional travelers (1-2 bookings), and one-time customers. For each segment, show average booking value, most popular destination airports, and preferred aircraft class.",
        "Calculate the revenue loss from cancelled flights in the last quarter. Show which routes have the highest cancellation rates and estimate the passenger compensation costs.",
        "Analyze booking patterns month-over-month for the past year. Identify seasonal routes (vacation destinations) vs business routes (consistent throughout year).",
        "Compare booking behavior of frequent flyers vs non-frequent flyers: average booking value, cancellation rates, preferred booking channels, and route preferences.",
    ]

    for query in analytical_queries:
        result = classify_query_complexity(query)
        print(f"\n{'*' * 80}")
        print(f"Query: {query[:100]}...")
        print(f"  Analytical: {result['is_analytical']} [PASS]" if result['is_analytical'] else f"  Analytical: {result['is_analytical']} [FAIL]")
        print(f"  Categories ({len(result['signal_categories'])}): {result['signal_categories']}")
        print(f"  Total Signals: {result['signal_count']}")
        for cat, sigs in result['detected_signals'].items():
            print(f"    - {cat}: {sigs}")


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 80)
    print("EDGE CASES")
    print("=" * 80)

    edge_cases = [
        ("Empty query", ""),
        ("Generic question", "What can you do?"),
        ("Single word", "help"),
        ("Multiple same category", "Show top best highest routes"),  # 3 ranking signals, 1 category
    ]

    for label, query in edge_cases:
        result = classify_query_complexity(query)
        print(f"\n{label}: '{query}'")
        print(f"  Analytical: {result['is_analytical']}")
        print(f"  Categories: {result['signal_categories']}")


def test_governance_response():
    """Test the governed clarification response generation"""
    print("\n" + "=" * 80)
    print("SAMPLE GOVERNANCE RESPONSE")
    print("=" * 80)

    query = "Identify top 20 VIP customers by booking value in the last quarter, flag inactive ones"
    classification = classify_query_complexity(query)

    print(f"\nQuery: {query}\n")
    print(f"Classification: {classification}\n")

    if classification['is_analytical']:
        response = generate_governance_clarification(classification, query)
        print("GOVERNANCE RESPONSE:")
        print("-" * 80)
        print(response)
        print("-" * 80)


if __name__ == "__main__":
    test_simple_queries()
    test_analytical_queries()
    test_edge_cases()
    test_governance_response()

    print("\n" + "=" * 80)
    print("[PASS] Query Governance Tests Complete")
    print("=" * 80)
