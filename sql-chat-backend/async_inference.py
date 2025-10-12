"""
Asynchronous Inference Engine for OptimaX
Handles concurrent LLM calls and async SQL generation
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class AsyncInferenceEngine:
    """Async LLM inference with concurrent execution and caching"""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._call_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 3600  # 1 hour cache TTL

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_cache_key(self, model: str, prompt: str, temperature: float) -> str:
        """Generate cache key for prompt"""
        content = f"{model}:{prompt}:{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False

        age = (datetime.now() - self._cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_ttl_seconds

    async def generate_async(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 200,
        use_cache: bool = True
    ) -> str:
        """
        Generate text asynchronously using Ollama

        Args:
            model: Model name (e.g., "phi3:mini", "qwen2.5-coder:3b")
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching

        Returns:
            Generated text
        """
        # Check cache first
        cache_key = self._get_cache_key(model, prompt, temperature)
        if use_cache and cache_key in self._call_cache and self._is_cache_valid(cache_key):
            logger.info(f"Cache hit for model {model}")
            return self._call_cache[cache_key]

        await self._ensure_session()

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            start_time = datetime.now()
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

            generated_text = result.get("response", "").strip()

            if not generated_text:
                raise RuntimeError(f"Empty response from Ollama model {model}")

            # Cache the result
            if use_cache:
                self._call_cache[cache_key] = generated_text
                self._cache_timestamps[cache_key] = datetime.now()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Generated response from {model} in {elapsed:.2f}s")

            return generated_text

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling Ollama: {str(e)}")
            raise RuntimeError(f"Ollama API error: {str(e)}")
        except Exception as e:
            logger.error(f"Error during async generation: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")

    async def generate_with_timeout(
        self,
        model: str,
        prompt: str,
        timeout_seconds: float = 30.0,
        **kwargs
    ) -> str:
        """
        Generate text with timeout

        Args:
            model: Model name
            prompt: Input prompt
            timeout_seconds: Timeout in seconds
            **kwargs: Additional arguments for generate_async

        Returns:
            Generated text

        Raises:
            asyncio.TimeoutError: If generation exceeds timeout
        """
        try:
            return await asyncio.wait_for(
                self.generate_async(model, prompt, **kwargs),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Generation timeout after {timeout_seconds}s for model {model}")
            raise

    async def generate_concurrent(
        self,
        tasks: list[Dict[str, Any]]
    ) -> list[str]:
        """
        Generate multiple completions concurrently

        Args:
            tasks: List of dicts with keys: model, prompt, temperature, max_tokens

        Returns:
            List of generated texts in same order as tasks
        """
        coroutines = [
            self.generate_async(
                model=task["model"],
                prompt=task["prompt"],
                temperature=task.get("temperature", 0.1),
                max_tokens=task.get("max_tokens", 200),
                use_cache=task.get("use_cache", True)
            )
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to error strings
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {str(result)}")
                processed_results.append(f"ERROR: {str(result)}")
            else:
                processed_results.append(result)

        return processed_results

    def clear_cache(self):
        """Clear all cached results"""
        self._call_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Inference cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(1 for key in self._call_cache if self._is_cache_valid(key))

        return {
            "total_entries": len(self._call_cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._call_cache) - valid_entries,
            "cache_ttl_seconds": self.cache_ttl_seconds
        }


# Singleton instance
_inference_engine: Optional[AsyncInferenceEngine] = None

def get_inference_engine(ollama_base_url: str = "http://localhost:11434") -> AsyncInferenceEngine:
    """Get singleton inference engine instance"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = AsyncInferenceEngine(ollama_base_url)
    return _inference_engine

async def cleanup_inference_engine():
    """Cleanup inference engine (call on shutdown)"""
    global _inference_engine
    if _inference_engine is not None:
        await _inference_engine.close()
        _inference_engine = None
