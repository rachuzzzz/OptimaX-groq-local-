"""
OptimaX v6.1 - Semantic Intent Extractor (ONE-SHOT ONLY)
=========================================================

PURPOSE:
This module provides LLM-based semantic intent extraction.
It produces a BEST-GUESS interpretation of user intent.

v6.1 CHANGES (Architecture Simplification):
- ONE-SHOT extraction only
- NO confidence gating (removed)
- NO clarification decisions (removed)
- NO should_proceed logic (removed)
- Just extracts intent and returns it

WHAT THIS MODULE DOES:
- Uses LLM to interpret user queries conceptually
- Extracts structured intent (entity, metric, event, etc.)
- Returns best-guess interpretation

WHAT THIS MODULE DOES NOT DO (v6.1):
- Decide whether to proceed or clarify (that's intent_accumulator's job)
- Gate on confidence thresholds
- Generate clarification questions
- Block execution

INVARIANTS PRESERVED:
- Schema-blindness (no table/column access)
- No SQL generation
- Original query preserved

Author: OptimaX Team
Version: 6.1 (One-Shot Extractor)
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC INTENT (Structured Output)
# =============================================================================

@dataclass
class SemanticIntent:
    """
    Structured representation of user intent.

    This is a CONCEPTUAL representation extracted by LLM.
    It describes WHAT the user wants, not HOW to get it.
    """
    # Core intent fields
    entity_type: str = "unknown"
    metric: str = "unknown"
    event: Optional[str] = None

    # Scope fields
    time_scope: str = "all_time"
    ranking: str = "none"
    n: Optional[int] = None
    filter_conditions: List[str] = field(default_factory=list)
    aggregation: Optional[str] = None

    # Original query (preserved)
    raw_query: str = ""

    # LLM reasoning (for debugging)
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "metric": self.metric,
            "event": self.event,
            "time_scope": self.time_scope,
            "ranking": self.ranking,
            "n": self.n,
            "filter_conditions": self.filter_conditions,
            "aggregation": self.aggregation,
            "raw_query": self.raw_query,
            "reasoning": self.reasoning,
        }


@dataclass
class ExtractionResult:
    """
    Result of semantic intent extraction.

    v6.1: Removed should_proceed - accumulator decides that.
    """
    success: bool
    intent: Optional[SemanticIntent] = None
    error: Optional[str] = None


# =============================================================================
# SEMANTIC INTENT EXTRACTOR (v6.1 - One-Shot Only)
# =============================================================================

class SemanticIntentExtractor:
    """
    LLM-based semantic intent extraction.

    v6.1 SIMPLIFICATION:
    - Extracts intent, returns it
    - NO decision logic
    - NO confidence gating
    - NO clarification generation

    The intent_accumulator decides whether to proceed or clarify.
    """

    def __init__(self, llm, **kwargs):
        """
        Initialize the extractor.

        Args:
            llm: LLM instance for intent extraction
            **kwargs: Ignored (for backward compatibility)
        """
        self.llm = llm
        logger.info("SemanticIntentExtractor initialized (v6.1 one-shot)")

    def extract(self, user_query: str) -> ExtractionResult:
        """
        Extract semantic intent from a user query.

        Returns best-guess interpretation. Does NOT decide proceed/clarify.
        """
        if not user_query or not user_query.strip():
            return ExtractionResult(success=False, error="Empty query")

        try:
            intent = self._extract_with_llm(user_query)

            logger.info(
                f"[EXTRACT] entity={intent.entity_type}, "
                f"metric={intent.metric}, event={intent.event}"
            )

            return ExtractionResult(success=True, intent=intent)

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return partial intent on error
            return ExtractionResult(
                success=True,  # Still success - we have partial info
                intent=SemanticIntent(raw_query=user_query),
                error=str(e)
            )

    def _extract_with_llm(self, user_query: str) -> SemanticIntent:
        """Use LLM to extract semantic intent."""

        prompt = f"""Analyze this query and extract the user's intent. Return JSON only.

QUERY: "{user_query}"

Return this JSON structure:
{{
    "entity_type": "main entity mentioned in the query (use the exact term the user used) or 'unknown'",
    "metric": "measurement type (count, sum, average, list, total, etc.) or 'unknown'",
    "event": "event type (arrivals, departures, bookings, etc.) or null",
    "time_scope": "all_time, last_year, last_month, last_week, or 'unknown'",
    "ranking": "top_n, bottom_n, or none",
    "n": number for top/bottom N or null,
    "filter_conditions": ["conceptual filters like 'from JFK', 'VIP customers'"],
    "aggregation": "group_by_entity, sum_all, average_all, none, or null",
    "reasoning": "brief explanation"
}}

Be generous in interpretation. If "busiest" likely means "by count", use "count".
If time scope isn't mentioned, use "all_time".
Return ONLY the JSON object."""

        response = self.llm.complete(prompt)
        response_text = str(response).strip()

        # Extract JSON
        json_text = self._extract_json(response_text)
        parsed = json.loads(json_text)

        return SemanticIntent(
            entity_type=parsed.get("entity_type", "unknown"),
            metric=parsed.get("metric", "unknown"),
            event=parsed.get("event"),
            time_scope=parsed.get("time_scope", "all_time"),
            ranking=parsed.get("ranking", "none"),
            n=parsed.get("n"),
            filter_conditions=parsed.get("filter_conditions", []),
            aggregation=parsed.get("aggregation"),
            raw_query=user_query,
            reasoning=parsed.get("reasoning"),
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        if "```json" in text:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
            if match:
                return match.group(1).strip()

        if "```" in text:
            match = re.search(r'```\s*([\s\S]*?)\s*```', text)
            if match:
                return match.group(1).strip()

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0)

        return text


# =============================================================================
# BACKWARD COMPATIBILITY (will be removed in future)
# =============================================================================

def format_intent_clarification(result: ExtractionResult) -> str:
    """Legacy function - returns empty (accumulator handles clarification)."""
    return ""
