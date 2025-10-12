"""
Test script for OptimaX optimizations
Tests heuristic routing, async inference, and caching
"""

import asyncio
import time
from heuristic_router import get_heuristic_router
from async_inference import get_inference_engine
from query_cache import get_query_cache

# Test queries
TEST_QUERIES = {
    "sql_queries": [
        "Show me the top 10 states with most accidents",
        "Find severe accidents during snow weather",
        "What times of day have the most accidents?",
        "Count accidents near traffic signals in California",
        "Which weather conditions cause the most accidents?",
        "accidents in CA state by city",
        "how many crashes in Texas",
        "list accidents by severity",
    ],
    "chat_queries": [
        "Hello, how are you?",
        "What can you do?",
        "Help me understand your capabilities",
        "Good morning",
        "Thanks for your help",
    ]
}

def test_heuristic_router():
    """Test heuristic routing accuracy and speed"""
    print("\n" + "="*60)
    print("TEST 1: Heuristic Router")
    print("="*60)

    router = get_heuristic_router(fallback_to_llm=True)

    print("\n📊 Testing SQL Queries:")
    print("-" * 60)
    sql_correct = 0
    total_time = 0

    for query in TEST_QUERIES["sql_queries"]:
        start = time.time()
        result = router.route(query)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        confidence = router.get_confidence(query)
        is_correct = result == "sql" or result is None  # None means LLM fallback (acceptable)

        if is_correct:
            sql_correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} '{query[:50]}...'")
        print(f"   Result: {result}, Confidence: {confidence:.2f}, Time: {elapsed:.2f}ms")

    print(f"\n✅ SQL Query Accuracy: {sql_correct}/{len(TEST_QUERIES['sql_queries'])} ({sql_correct/len(TEST_QUERIES['sql_queries'])*100:.1f}%)")
    print(f"⚡ Average routing time: {total_time/len(TEST_QUERIES['sql_queries']):.2f}ms")

    print("\n📊 Testing Chat Queries:")
    print("-" * 60)
    chat_correct = 0
    total_time = 0

    for query in TEST_QUERIES["chat_queries"]:
        start = time.time()
        result = router.route(query)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        confidence = router.get_confidence(query)
        is_correct = result == "chat"

        if is_correct:
            chat_correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} '{query}'")
        print(f"   Result: {result}, Confidence: {confidence:.2f}, Time: {elapsed:.2f}ms")

    print(f"\n✅ Chat Query Accuracy: {chat_correct}/{len(TEST_QUERIES['chat_queries'])} ({chat_correct/len(TEST_QUERIES['chat_queries'])*100:.1f}%)")
    print(f"⚡ Average routing time: {total_time/len(TEST_QUERIES['chat_queries']):.2f}ms")

    # Overall accuracy
    total_correct = sql_correct + chat_correct
    total_queries = len(TEST_QUERIES["sql_queries"]) + len(TEST_QUERIES["chat_queries"])
    print(f"\n🎯 Overall Accuracy: {total_correct}/{total_queries} ({total_correct/total_queries*100:.1f}%)")

async def test_async_inference():
    """Test async inference speed and caching"""
    print("\n" + "="*60)
    print("TEST 2: Async Inference Engine")
    print("="*60)

    engine = get_inference_engine()

    test_prompt = """Classify the user's intent.

SQL_INTENT: Questions about traffic accident data
CHAT_INTENT: Greetings, general chat

User question: Show me accidents in California
Intent:"""

    print("\n📊 Testing Single Inference (Cold):")
    print("-" * 60)
    start = time.time()
    try:
        response = await engine.generate_async(
            model="phi3:mini",
            prompt=test_prompt,
            temperature=0.1,
            max_tokens=50,
            use_cache=True
        )
        elapsed = (time.time() - start) * 1000
        print(f"✅ Response: {response[:100]}...")
        print(f"⚡ Time: {elapsed:.0f}ms")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("⚠️  Note: Make sure Ollama is running with phi3:mini model")
        return

    print("\n📊 Testing Single Inference (Cached):")
    print("-" * 60)
    start = time.time()
    try:
        response = await engine.generate_async(
            model="phi3:mini",
            prompt=test_prompt,
            temperature=0.1,
            max_tokens=50,
            use_cache=True
        )
        elapsed = (time.time() - start) * 1000
        print(f"✅ Response: {response[:100]}...")
        print(f"⚡ Time: {elapsed:.0f}ms (should be <5ms)")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    print("\n📊 Testing Concurrent Inference:")
    print("-" * 60)
    tasks = [
        {"model": "phi3:mini", "prompt": test_prompt, "temperature": 0.1, "max_tokens": 50}
        for _ in range(3)
    ]

    start = time.time()
    try:
        results = await engine.generate_concurrent(tasks)
        elapsed = (time.time() - start) * 1000
        print(f"✅ Generated {len(results)} responses concurrently")
        print(f"⚡ Total time: {elapsed:.0f}ms ({elapsed/len(results):.0f}ms per request)")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

    # Cache stats
    cache_stats = engine.get_cache_stats()
    print(f"\n📈 Cache Stats:")
    print(f"   Total entries: {cache_stats['total_entries']}")
    print(f"   Valid entries: {cache_stats['valid_entries']}")

    await engine.close()

def test_query_cache():
    """Test query cache functionality"""
    print("\n" + "="*60)
    print("TEST 3: Query Cache")
    print("="*60)

    cache = get_query_cache(max_size=100, default_ttl=3600)

    print("\n📊 Testing SQL Query Caching:")
    print("-" * 60)

    # Cache a SQL query
    test_question = "Show top 10 states with accidents"
    test_sql = "SELECT state, COUNT(*) as count FROM us_accidents GROUP BY state ORDER BY count DESC LIMIT 10;"
    test_results = [
        {"state": "CA", "count": 1741433},
        {"state": "FL", "count": 880192},
        {"state": "TX", "count": 582837}
    ]

    cache.cache_sql_query(test_question, test_sql, test_results)
    print("✅ Cached SQL query")

    # Retrieve from cache
    start = time.time()
    cached_result = cache.get_sql_query(test_question)
    elapsed = (time.time() - start) * 1000

    if cached_result:
        print(f"✅ Retrieved from cache in {elapsed:.2f}ms")
        print(f"   SQL: {cached_result['sql_query'][:50]}...")
        print(f"   Results: {len(cached_result['query_results'])} rows")
    else:
        print("❌ Failed to retrieve from cache")

    print("\n📊 Testing Chat Response Caching:")
    print("-" * 60)

    test_message = "Hello, how are you?"
    test_response = "Hello! I'm OptimaX, ready to help you analyze traffic accident data."

    cache.cache_chat_response(test_message, test_response)
    print("✅ Cached chat response")

    start = time.time()
    cached_chat = cache.get_chat_response(test_message)
    elapsed = (time.time() - start) * 1000

    if cached_chat:
        print(f"✅ Retrieved from cache in {elapsed:.2f}ms")
        print(f"   Response: {cached_chat[:60]}...")
    else:
        print("❌ Failed to retrieve from cache")

    print("\n📊 Testing Route Decision Caching:")
    print("-" * 60)

    test_query = "Show accidents in California"
    cache.cache_route_decision(test_query, "sql")
    print("✅ Cached route decision")

    start = time.time()
    cached_route = cache.get_route_decision(test_query)
    elapsed = (time.time() - start) * 1000

    if cached_route:
        print(f"✅ Retrieved from cache in {elapsed:.2f}ms")
        print(f"   Route: {cached_route}")
    else:
        print("❌ Failed to retrieve from cache")

    # Cache stats
    stats = cache.get_stats()
    print(f"\n📈 Cache Stats:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Utilization: {stats['utilization']:.1%}")

def test_end_to_end_performance():
    """Test end-to-end performance comparison"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Performance")
    print("="*60)

    router = get_heuristic_router()
    cache = get_query_cache()

    test_queries = TEST_QUERIES["sql_queries"][:5]

    print("\n📊 Testing Query Processing Pipeline:")
    print("-" * 60)

    total_time = 0
    heuristic_hits = 0
    cache_hits = 0

    for query in test_queries:
        start = time.time()

        # Step 1: Check cache
        cached_route = cache.get_route_decision(query)
        if cached_route:
            cache_hits += 1
            route = cached_route
        else:
            # Step 2: Heuristic routing
            route = router.route(query)
            if route is not None:
                heuristic_hits += 1
                cache.cache_route_decision(query, route)

        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        print(f"✅ '{query[:50]}...'")
        print(f"   Route: {route}, Time: {elapsed:.2f}ms")

    print(f"\n📊 Pipeline Performance:")
    print(f"   Average time: {total_time/len(test_queries):.2f}ms")
    print(f"   Cache hits: {cache_hits}/{len(test_queries)} ({cache_hits/len(test_queries)*100:.0f}%)")
    print(f"   Heuristic hits: {heuristic_hits}/{len(test_queries)} ({heuristic_hits/len(test_queries)*100:.0f}%)")

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 OptimaX Optimization Tests")
    print("="*60)

    # Test 1: Heuristic Router
    test_heuristic_router()

    # Test 2: Async Inference
    await test_async_inference()

    # Test 3: Query Cache
    test_query_cache()

    # Test 4: End-to-End
    test_end_to_end_performance()

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\n💡 Tips:")
    print("   - Heuristic routing should be 60-80% accurate")
    print("   - Cache hits should reduce response time to <10ms")
    print("   - Async inference should be faster than synchronous")
    print("   - Monitor these metrics in production via /performance endpoint")

if __name__ == "__main__":
    asyncio.run(main())
