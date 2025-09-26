#!/usr/bin/env python3
"""
Test script for the LLM-only router functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sql-chat-backend'))

from main import route_query_with_llm, generate_chat_response

def test_llm_only_routing():
    """Test the LLM-only routing functionality"""
    print("Testing LLM-Only Router Functionality")
    print("=" * 40)

    # Test cases for different types of queries
    test_cases = [
        # SQL queries
        ("How many accidents happened in California?", "sql"),
        ("Show me the top 10 states with most accidents", "sql"),
        ("What is the average severity of accidents?", "sql"),
        ("Find accidents with high temperature", "sql"),
        ("Count accidents by weather condition", "sql"),

        # Chat queries
        ("Hello, how are you?", "chat"),
        ("Hi there!", "chat"),
        ("Thank you for your help", "chat"),
        ("What is your name?", "chat"),
        ("Who are you?", "chat"),
        ("Goodbye", "chat"),
    ]

    print("Testing LLM routing (requires model to be loaded):")
    print("-" * 50)

    for query, expected in test_cases:
        try:
            result = route_query_with_llm(query)
            status = "✓" if result == expected else "✗"
            print(f"{status} '{query}' -> {result} (expected: {expected})")
        except RuntimeError as e:
            print(f"✗ '{query}' -> ERROR: {str(e)}")

    print("\nTesting chat response generation:")
    print("-" * 40)

    chat_queries = [
        "Hello, how are you?",
        "What can you help me with?",
        "Thank you for your assistance"
    ]

    for query in chat_queries:
        try:
            response = generate_chat_response(query)
            print(f"✓ '{query}' -> '{response[:50]}...'")
        except RuntimeError as e:
            print(f"✗ '{query}' -> ERROR: {str(e)}")

if __name__ == "__main__":
    test_llm_only_routing()