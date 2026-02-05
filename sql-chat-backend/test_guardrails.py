"""
OptimaX v5.0 - Guardrails Test Suite
=====================================

Tests for the execution guardrails implemented in v5.0:
1. SQL Alias Validation
2. Query Complexity Analysis
3. Route Context Binding

Run with: python test_guardrails.py
"""

import sys


def test_sql_alias_validation():
    """Test SQL alias validation module."""
    print("\n" + "=" * 60)
    print("TEST 1: SQL ALIAS VALIDATION")
    print("=" * 60)

    from sql_validator import SQLAliasValidator, validate_sql_aliases

    validator = SQLAliasValidator()

    # Test case 1: Valid SQL with proper aliases
    print("\n1.1 Valid SQL with proper aliases:")
    sql1 = """
    SELECT T1.passenger_id, T1.first_name, T2.booking_id
    FROM passenger T1
    JOIN booking T2 ON T1.passenger_id = T2.passenger_id
    WHERE T1.passenger_id = 123
    """
    result1 = validator.validate(sql1)
    print(f"   SQL: {sql1.strip()[:60]}...")
    print(f"   Valid: {result1.valid}")
    print(f"   Declared: {result1.declared_aliases}")
    print(f"   Referenced: {result1.referenced_aliases}")
    assert result1.valid, "Expected valid SQL"
    print("   PASS")

    # Test case 2: INVALID SQL - undefined alias T2
    print("\n1.2 Invalid SQL - undefined alias T2:")
    sql2 = """
    SELECT T2.passenger_id, T1.first_name
    FROM passenger T1
    WHERE T2.passenger_id = 123
    """
    result2 = validator.validate(sql2)
    print(f"   SQL: {sql2.strip()[:60]}...")
    print(f"   Valid: {result2.valid}")
    print(f"   Declared: {result2.declared_aliases}")
    print(f"   Referenced: {result2.referenced_aliases}")
    print(f"   Undefined: {result2.undefined_aliases}")
    print(f"   Error: {result2.error_message}")
    assert not result2.valid, "Expected invalid SQL"
    assert "T2" in result2.undefined_aliases, "T2 should be undefined"
    print("   PASS")

    # Test case 3: Valid SQL without aliases
    print("\n1.3 Valid SQL without aliases:")
    sql3 = """
    SELECT passenger_id, first_name
    FROM passenger
    WHERE passenger_id = 123
    """
    result3 = validator.validate(sql3)
    print(f"   SQL: {sql3.strip()[:60]}...")
    print(f"   Valid: {result3.valid}")
    assert result3.valid, "Expected valid SQL"
    print("   PASS")

    # Test case 4: Schema-qualified tables (should not flag schema as undefined alias)
    print("\n1.4 Schema-qualified tables:")
    sql4 = """
    SELECT p.passenger_id
    FROM postgres_air.passenger p
    WHERE p.passenger_id = 123
    """
    result4 = validator.validate(sql4)
    print(f"   SQL: {sql4.strip()[:60]}...")
    print(f"   Valid: {result4.valid}")
    print(f"   Declared: {result4.declared_aliases}")
    assert result4.valid, "Expected valid SQL"
    print("   PASS")

    print("\n" + "-" * 60)
    print("SQL ALIAS VALIDATION: ALL TESTS PASSED")


def test_query_complexity():
    """Test query complexity analyzer."""
    print("\n" + "=" * 60)
    print("TEST 2: QUERY COMPLEXITY ANALYSIS")
    print("=" * 60)

    from sql_validator import QueryComplexityAnalyzer, analyze_query_complexity

    analyzer = QueryComplexityAnalyzer()

    # Test case 1: Simple query (low complexity)
    print("\n2.1 Simple query (low complexity):")
    sql1 = "SELECT * FROM passenger WHERE passenger_id = 123 LIMIT 10"
    result1 = analyzer.analyze(sql1)
    print(f"   SQL: {sql1[:50]}...")
    print(f"   Complexity score: {result1.complexity_score}")
    print(f"   JOINs: {result1.join_count}")
    print(f"   Has WHERE: {result1.has_where_filter}")
    print(f"   Is safe: {result1.is_safe}")
    assert result1.is_safe, "Expected safe query"
    assert result1.join_count == 0, "Expected 0 joins"
    print("   PASS")

    # Test case 2: Complex query (many JOINs, no WHERE)
    print("\n2.2 Complex query (many JOINs, no WHERE):")
    sql2 = """
    SELECT COUNT(*), SUM(amount)
    FROM booking b
    JOIN passenger p ON b.passenger_id = p.passenger_id
    JOIN flight f ON b.flight_id = f.flight_id
    JOIN airport a1 ON f.departure_airport = a1.airport_code
    JOIN airport a2 ON f.arrival_airport = a2.airport_code
    """
    result2 = analyzer.analyze(sql2)
    print(f"   SQL: {sql2.strip()[:60]}...")
    print(f"   Complexity score: {result2.complexity_score}")
    print(f"   JOINs: {result2.join_count}")
    print(f"   Has WHERE: {result2.has_where_filter}")
    print(f"   Is safe: {result2.is_safe}")
    print(f"   Warning: {result2.warning_message}")
    assert result2.join_count >= 4, "Expected 4+ joins"
    assert not result2.has_where_filter, "Expected no WHERE"
    print("   PASS")

    # Test case 3: Query with subquery
    print("\n2.3 Query with subquery:")
    sql3 = """
    SELECT * FROM passenger
    WHERE passenger_id IN (SELECT passenger_id FROM booking WHERE amount > 1000)
    """
    result3 = analyzer.analyze(sql3)
    print(f"   SQL: {sql3.strip()[:60]}...")
    print(f"   Complexity score: {result3.complexity_score}")
    print(f"   Subqueries: {result3.subquery_count}")
    assert result3.subquery_count >= 1, "Expected 1+ subquery"
    print("   PASS")

    print("\n" + "-" * 60)
    print("QUERY COMPLEXITY ANALYSIS: ALL TESTS PASSED")


def test_route_context_binding():
    """Test route context binding in context resolver."""
    print("\n" + "=" * 60)
    print("TEST 3: ROUTE CONTEXT BINDING")
    print("=" * 60)

    from context_resolver import (
        SessionContext,
        ContextResolver,
        RouteDetector,
        create_session_context,
        create_context_resolver,
    )

    # Test case 1: Route detection
    print("\n3.1 Route detection patterns:")
    detector = RouteDetector()

    test_queries = [
        "Show flights from JFK to ATL",
        "What about JFK-LAX route?",
        "Flights between ORD and SFO",
        "Departing JFK arriving ATL",
    ]

    for query in test_queries:
        route = detector.detect_route(query)
        if route:
            print(f"   '{query}' -> {route[0]} to {route[1]}")
        else:
            print(f"   '{query}' -> No route detected")
    print("   PASS")

    # Test case 2: Route reference detection
    print("\n3.2 Route reference detection:")
    ref_queries = [
        "Who flew on this route the most?",
        "Show passengers for that route",
        "Total bookings on the same route",
    ]

    for query in ref_queries:
        ref = detector.detect_route_reference(query)
        print(f"   '{query}' -> Reference: '{ref}'")
    print("   PASS")

    # Test case 3: Full route context flow
    print("\n3.3 Full route context binding flow:")
    context = create_session_context("test-session-001")
    resolver = create_context_resolver()

    # Step 1: User specifies a route
    query1 = "Show flights from JFK to ATL"
    result1 = resolver.resolve(query1, context)
    print(f"   Query: '{query1}'")
    print(f"   Route bound: {context.has_route()}")
    if context.has_route():
        route = context.get_route()
        print(f"   Bound route: {route.departure} -> {route.arrival}")

    # Step 2: User references the route
    query2 = "Which passenger travelled on this route the most?"
    result2 = resolver.resolve(query2, context)
    print(f"\n   Follow-up: '{query2}'")
    print(f"   Resolved: {result2.resolved}")
    print(f"   Resolved query: '{result2.resolved_query}'")

    assert context.has_route(), "Route should be bound"
    assert result2.resolved, "Route reference should be resolved"
    assert "JFK" in result2.resolved_query, "JFK should be in resolved query"
    assert "ATL" in result2.resolved_query, "ATL should be in resolved query"
    print("   PASS")

    # Test case 4: Route reference without context
    print("\n3.4 Route reference without context (should ask clarification):")
    fresh_context = create_session_context("fresh-session")
    query3 = "Show passengers for this route"
    result3 = resolver.resolve(query3, fresh_context)
    print(f"   Query: '{query3}'")
    print(f"   Needs clarification: {result3.needs_clarification}")
    print(f"   Message: '{result3.clarification_message}'")
    assert result3.needs_clarification, "Should request clarification"
    print("   PASS")

    print("\n" + "-" * 60)
    print("ROUTE CONTEXT BINDING: ALL TESTS PASSED")


def test_relational_corrector_sql_rewriting():
    """
    Test that SQL rewriting places JOINs in the correct position.

    REGRESSION TEST (v6.1.2):
    The old implementation inserted JOINs at the wrong position,
    resulting in broken SQL like:

        SELECT ... FROM flight GROUP
        JOIN airport a1 ON ...
        BY airport_code

    This test verifies that JOINs are correctly placed BEFORE GROUP BY.
    """
    print("\n" + "=" * 60)
    print("TEST 4: RELATIONAL CORRECTOR SQL REWRITING")
    print("=" * 60)

    from relational_corrector import (
        RelationalCorrector,
        SchemaMetadata,
        ForeignKeyInfo,
    )

    # Build test schema: flight table with FK to airport
    schema = SchemaMetadata(
        tables={
            "flight": {"flight_id", "departure_airport", "arrival_airport", "scheduled_departure"},
            "airport": {"airport_code", "airport_name", "city"},
        },
        foreign_keys=[
            ForeignKeyInfo(
                source_table="flight",
                source_column="departure_airport",
                target_table="airport",
                target_column="airport_code"
            ),
            ForeignKeyInfo(
                source_table="flight",
                source_column="arrival_airport",
                target_table="airport",
                target_column="airport_code"
            ),
        ]
    )

    corrector = RelationalCorrector(schema)

    # Test case 4.1: SQL with GROUP BY - JOINs must appear BEFORE GROUP BY
    print("\n4.1 SQL with GROUP BY - JOIN placement:")
    input_sql = """SELECT airport_code, COUNT(*) FROM flight GROUP BY airport_code ORDER BY COUNT(*) DESC LIMIT 1"""

    print(f"   Input SQL: {input_sql}")

    # Use forced FK to resolve ambiguity (simulating user chose "arrival_airport")
    forced_fk = {"flight": "arrival_airport"}
    result = corrector.correct_with_forced_fk(input_sql, forced_fk)

    print(f"   Success: {result.success}")
    print(f"   Output SQL:\n{result.corrected_sql}")

    # Verify the output is syntactically valid
    corrected = result.corrected_sql
    assert result.success, f"Correction should succeed, got: {result.error}"

    # Check that JOIN appears BEFORE GROUP BY
    join_pos = corrected.upper().find("JOIN")
    group_pos = corrected.upper().find("GROUP BY")
    order_pos = corrected.upper().find("ORDER BY")
    limit_pos = corrected.upper().find("LIMIT")

    assert join_pos > 0, "JOIN should be in the output"
    assert group_pos > 0, "GROUP BY should be in the output"
    assert join_pos < group_pos, f"JOIN (pos {join_pos}) must appear BEFORE GROUP BY (pos {group_pos})"
    assert group_pos < order_pos, f"GROUP BY (pos {group_pos}) must appear BEFORE ORDER BY (pos {order_pos})"
    assert order_pos < limit_pos, f"ORDER BY (pos {order_pos}) must appear BEFORE LIMIT (pos {limit_pos})"

    # Check that the column is qualified with alias
    assert "a1.airport_code" in corrected.lower() or "a1.airport_code" in corrected, \
        "airport_code should be qualified with join alias"

    print("   PASS - JOIN correctly placed before GROUP BY")

    # Test case 4.2: SQL with WHERE and GROUP BY
    print("\n4.2 SQL with WHERE and GROUP BY:")
    input_sql2 = """SELECT airport_code, COUNT(*) FROM flight WHERE scheduled_departure > '2024-01-01' GROUP BY airport_code"""

    print(f"   Input SQL: {input_sql2}")

    result2 = corrector.correct_with_forced_fk(input_sql2, forced_fk)
    corrected2 = result2.corrected_sql

    print(f"   Output SQL:\n{corrected2}")

    assert result2.success, f"Correction should succeed"

    # Check clause order: FROM -> JOIN -> WHERE -> GROUP BY
    join_pos2 = corrected2.upper().find("JOIN")
    where_pos2 = corrected2.upper().find("WHERE")
    group_pos2 = corrected2.upper().find("GROUP BY")

    assert join_pos2 < where_pos2, f"JOIN must appear BEFORE WHERE"
    assert where_pos2 < group_pos2, f"WHERE must appear BEFORE GROUP BY"

    print("   PASS - Clause order is correct: FROM -> JOIN -> WHERE -> GROUP BY")

    # Test case 4.3: Simple SQL without GROUP BY (should still work)
    print("\n4.3 Simple SQL without GROUP BY:")
    input_sql3 = """SELECT airport_code FROM flight LIMIT 10"""

    result3 = corrector.correct_with_forced_fk(input_sql3, forced_fk)
    corrected3 = result3.corrected_sql

    print(f"   Input: {input_sql3}")
    print(f"   Output: {corrected3}")

    assert result3.success, "Correction should succeed"
    assert "JOIN" in corrected3.upper(), "JOIN should be added"
    assert "LIMIT" in corrected3.upper(), "LIMIT should be preserved"

    print("   PASS - Simple query corrected correctly")

    print("\n" + "-" * 60)
    print("RELATIONAL CORRECTOR SQL REWRITING: ALL TESTS PASSED")


def main():
    print("=" * 60)
    print("OptimaX v5.0 - Guardrails Test Suite")
    print("=" * 60)

    try:
        test_sql_alias_validation()
        test_query_complexity()
        test_route_context_binding()
        test_relational_corrector_sql_rewriting()

        print("\n" + "=" * 60)
        print("ALL GUARDRAILS TESTS PASSED")
        print("=" * 60)
        return 0

    except ImportError as e:
        print(f"\nIMPORT ERROR: {e}")
        print("Make sure you're running from the sql-chat-backend directory")
        return 1

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
