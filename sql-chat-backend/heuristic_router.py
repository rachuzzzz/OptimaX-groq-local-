"""
Heuristic-based Query Router for OptimaX
Reduces LLM calls by using pattern matching and keyword detection
"""

import re
from typing import Literal
from functools import lru_cache

QueryIntent = Literal["sql", "chat"]

class HeuristicRouter:
    """Fast heuristic routing to classify queries without LLM calls"""

    # SQL intent keywords (data query related)
    SQL_KEYWORDS = {
        # Query verbs
        'show', 'find', 'get', 'list', 'display', 'count', 'how many', 'what',
        'which', 'where', 'when', 'who', 'total', 'sum', 'average', 'avg',
        'maximum', 'minimum', 'max', 'min', 'top', 'bottom', 'most', 'least',
        'analyze', 'compare', 'group', 'filter', 'search',

        # Domain-specific keywords
        'accident', 'accidents', 'crash', 'crashes', 'collision',
        'state', 'city', 'county', 'location', 'weather', 'severe', 'severity',
        'traffic', 'signal', 'junction', 'highway', 'road', 'street',
        'temperature', 'rain', 'snow', 'fog', 'wind', 'visibility',
        'california', 'texas', 'florida', 'york',

        # SQL-like terms
        'by state', 'by city', 'by weather', 'by severity', 'per state',
        'in california', 'during', 'between', 'greater than', 'less than'
    }

    # Chat intent keywords (conversational)
    CHAT_KEYWORDS = {
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon',
        'good evening', 'thanks', 'thank you', 'bye', 'goodbye',
        'help', 'what can you do', 'who are you', 'what are you',
        'introduce yourself', 'capabilities', 'how do you work'
    }

    # SQL patterns (regex)
    SQL_PATTERNS = [
        r'\b(how many|count)\b.*\b(accident|crash)',
        r'\b(show|list|display|find)\b.*\b(accident|state|city|weather)',
        r'\b(top|bottom|most|least)\b.*\b\d+',
        r'\b(which|what)\b.*(state|city|location|weather).*\b(most|highest|lowest)',
        r'\b(severe|severity)\b.*\b(accident|crash)',
        r'\b(weather|condition).*\b(accident|crash)',
        r'\b(traffic signal|junction|crossing)',
        r'\b(during|between|in)\b.*\b(20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)',
        r'\b(average|avg|mean|median)\b',
        r'\bCA\b|\bTX\b|\bFL\b|\bNY\b',  # State codes
    ]

    def __init__(self, fallback_to_llm: bool = True):
        """
        Initialize heuristic router

        Args:
            fallback_to_llm: If True, returns None for ambiguous cases (triggering LLM fallback)
                           If False, makes best guess without LLM
        """
        self.fallback_to_llm = fallback_to_llm
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.SQL_PATTERNS]

    @lru_cache(maxsize=1000)
    def route(self, query: str) -> QueryIntent | None:
        """
        Route query using heuristics

        Returns:
            "sql" for SQL queries
            "chat" for conversational queries
            None if uncertain (fallback to LLM)
        """
        query_lower = query.lower().strip()

        # Empty or very short queries are likely chat
        if len(query_lower) < 3:
            return "chat"

        # Check for exact chat keyword matches (high confidence)
        for keyword in self.CHAT_KEYWORDS:
            if keyword in query_lower:
                # Exception: "help me find accidents" is SQL, not chat
                if any(sql_kw in query_lower for sql_kw in ['find', 'show', 'accident', 'crash']):
                    if keyword in ['help', 'can you']:
                        continue  # Don't immediately classify as chat
                return "chat"

        # Check SQL patterns (regex)
        pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(query_lower))
        if pattern_matches >= 2:
            return "sql"  # High confidence SQL

        # Check SQL keywords
        sql_keyword_count = sum(1 for keyword in self.SQL_KEYWORDS if keyword in query_lower)

        # Decision logic
        if sql_keyword_count >= 3:
            return "sql"  # High confidence SQL
        elif sql_keyword_count >= 1 and pattern_matches >= 1:
            return "sql"  # Medium confidence SQL
        elif sql_keyword_count == 0 and pattern_matches == 0:
            return "chat"  # Likely conversational

        # Ambiguous case
        if self.fallback_to_llm:
            return None  # Trigger LLM routing
        else:
            # Make best guess: more SQL keywords = SQL, otherwise chat
            return "sql" if sql_keyword_count >= 1 else "chat"

    def get_confidence(self, query: str) -> float:
        """
        Get confidence score for routing decision (0.0 to 1.0)

        Returns:
            1.0 = very confident
            0.0 = not confident (should use LLM)
        """
        query_lower = query.lower().strip()

        # Chat keyword match = high confidence
        for keyword in self.CHAT_KEYWORDS:
            if keyword in query_lower:
                return 0.95

        # Pattern matches
        pattern_matches = sum(1 for pattern in self.compiled_patterns if pattern.search(query_lower))
        sql_keyword_count = sum(1 for keyword in self.SQL_KEYWORDS if keyword in query_lower)

        # Confidence calculation
        if pattern_matches >= 2 or sql_keyword_count >= 3:
            return 0.9
        elif pattern_matches >= 1 and sql_keyword_count >= 2:
            return 0.75
        elif sql_keyword_count >= 1 or pattern_matches >= 1:
            return 0.5
        else:
            return 0.3

    def explain_decision(self, query: str) -> dict:
        """
        Explain routing decision for debugging

        Returns:
            Dictionary with decision reasoning
        """
        query_lower = query.lower().strip()

        matched_sql_keywords = [kw for kw in self.SQL_KEYWORDS if kw in query_lower]
        matched_chat_keywords = [kw for kw in self.CHAT_KEYWORDS if kw in query_lower]
        matched_patterns = [i for i, pattern in enumerate(self.compiled_patterns) if pattern.search(query_lower)]

        intent = self.route(query)
        confidence = self.get_confidence(query)

        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "matched_sql_keywords": matched_sql_keywords,
            "matched_chat_keywords": matched_chat_keywords,
            "matched_pattern_indices": matched_patterns,
            "used_llm_fallback": intent is None
        }


# Singleton instance
_router_instance = None

def get_heuristic_router(fallback_to_llm: bool = True) -> HeuristicRouter:
    """Get singleton heuristic router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = HeuristicRouter(fallback_to_llm=fallback_to_llm)
    return _router_instance
