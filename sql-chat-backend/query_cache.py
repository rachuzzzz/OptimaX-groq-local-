"""
Intelligent Query Cache for OptimaX
Caches SQL queries, results, and chat responses
"""

import hashlib
import json
import logging
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    entry_type: str  # "sql_query", "chat_response", "route_decision"

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data


class QueryCache:
    """LRU cache with TTL for query results"""

    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        """
        Initialize query cache

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        content = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats["misses"] += 1
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            logger.debug(f"Cache entry expired: {key[:16]}...")
            del self._cache[key]
            self._stats["misses"] += 1
            self._stats["expirations"] += 1
            return None

        # Update access metadata
        entry.last_accessed = datetime.now()
        entry.access_count += 1

        # Move to end (LRU)
        self._cache.move_to_end(key)

        self._stats["hits"] += 1
        logger.debug(f"Cache hit: {key[:16]}... (access count: {entry.access_count})")
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        entry_type: str = "general"
    ):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            entry_type: Type of entry for stats
        """
        # Evict if at max size
        if key not in self._cache and len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Evicted cache entry: {evicted_key[:16]}...")

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
            entry_type=entry_type
        )

        self._cache[key] = entry
        self._cache.move_to_end(key)
        logger.debug(f"Cached entry: {key[:16]}... (type: {entry_type}, ttl: {entry.ttl_seconds}s)")

    def cache_sql_query(
        self,
        user_question: str,
        sql_query: str,
        query_results: List[Dict],
        ttl: int = 1800  # 30 minutes
    ):
        """Cache SQL query and results"""
        key = self._generate_key(user_question)
        value = {
            "sql_query": sql_query,
            "query_results": query_results,
            "user_question": user_question
        }
        self.set(key, value, ttl=ttl, entry_type="sql_query")

    def get_sql_query(self, user_question: str) -> Optional[Dict[str, Any]]:
        """Get cached SQL query and results"""
        key = self._generate_key(user_question)
        return self.get(key)

    def cache_chat_response(
        self,
        user_message: str,
        chat_response: str,
        ttl: int = 3600  # 1 hour
    ):
        """Cache chat response"""
        key = self._generate_key(user_message)
        value = {
            "chat_response": chat_response,
            "user_message": user_message
        }
        self.set(key, value, ttl=ttl, entry_type="chat_response")

    def get_chat_response(self, user_message: str) -> Optional[str]:
        """Get cached chat response"""
        key = self._generate_key(user_message)
        cached = self.get(key)
        return cached["chat_response"] if cached else None

    def cache_route_decision(
        self,
        user_message: str,
        route: str,
        ttl: int = 7200  # 2 hours
    ):
        """Cache routing decision"""
        key = self._generate_key("route", user_message)
        value = {
            "route": route,
            "user_message": user_message
        }
        self.set(key, value, ttl=ttl, entry_type="route_decision")

    def get_route_decision(self, user_message: str) -> Optional[str]:
        """Get cached routing decision"""
        key = self._generate_key("route", user_message)
        cached = self.get(key)
        return cached["route"] if cached else None

    def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Invalidated cache entry: {key[:16]}...")

    def clear(self):
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")

    def cleanup_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            del self._cache[key]
            self._stats["expirations"] += 1

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        # Count by entry type
        type_counts = {}
        for entry in self._cache.values():
            type_counts[entry.entry_type] = type_counts.get(entry.entry_type, 0) + 1

        # Most accessed entries
        top_entries = sorted(
            self._cache.values(),
            key=lambda e: e.access_count,
            reverse=True
        )[:5]

        return {
            "total_entries": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "entries_by_type": type_counts,
            "top_accessed": [
                {
                    "key": entry.key[:16] + "...",
                    "type": entry.entry_type,
                    "access_count": entry.access_count,
                    "age_seconds": (datetime.now() - entry.created_at).total_seconds()
                }
                for entry in top_entries
            ]
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
        logger.info("Cache statistics reset")


# Singleton instance
_query_cache: Optional[QueryCache] = None

def get_query_cache(max_size: int = 500, default_ttl: int = 3600) -> QueryCache:
    """Get singleton query cache instance"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(max_size=max_size, default_ttl=default_ttl)
    return _query_cache
