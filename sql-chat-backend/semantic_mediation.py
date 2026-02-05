"""
OptimaX v6.1 - Semantic Mediation (OBSERVATIONAL ONLY)
=======================================================

PURPOSE:
Detects semantic concepts in queries for LOGGING ONLY.
Does NOT block. Does NOT clarify. Does NOT influence control flow.

v6.1 CHANGES:
- Stripped to minimal observer
- Removed all blocking logic
- Removed backward compatibility cruft
- Removed MediationAction, MediationResult, etc.

WHAT THIS MODULE DOES:
- Logs detected concepts (for analytics)
- That's it

WHAT THIS MODULE DOES NOT DO:
- Block execution
- Clarify
- Influence decisions

Author: OptimaX Team
Version: 6.1 (Minimal Observer)
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# CONCEPT PATTERNS (For Logging)
# =============================================================================

CONCEPTS = {
    "route": [r"\broutes?\b", r"\bconnections?\b", r"\bfrom\s+\w+\s+to\s+\w+"],
    "traffic": [r"\btraffic\b", r"\bbusiest\b", r"\bbusy\b", r"\bvolume\b"],
    "ranking": [r"\btop\s+\d+\b", r"\bbottom\s+\d+\b", r"\bbest\b", r"\bworst\b"],
    "entity": [r"\bairports?\b", r"\bflights?\b", r"\bcustom(?:er)?s?\b", r"\bpassengers?\b"],
}


# =============================================================================
# SEMANTIC OBSERVER
# =============================================================================

@dataclass
class Observation:
    """What we observed (for logging)."""
    concepts: List[str] = field(default_factory=list)


class SemanticMediator:
    """
    Observes concepts in queries. Does NOT block or clarify.
    """

    def __init__(self, **kwargs):
        self._patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in CONCEPTS.items()
        }
        logger.info("SemanticMediator initialized (v6.1 observer only)")

    def observe(self, query: str) -> Observation:
        """Observe concepts in query (logging only)."""
        detected = []
        q = query.lower()

        for concept, patterns in self._patterns.items():
            for p in patterns:
                if p.search(q):
                    detected.append(concept)
                    break

        if detected:
            logger.info(f"[OBS] Concepts: {detected}")

        return Observation(concepts=detected)

    def mediate(self, query: str, **kwargs) -> Observation:
        """Alias for observe() (backward compat)."""
        return self.observe(query)


# =============================================================================
# FACTORY
# =============================================================================

def create_semantic_mediator(**kwargs) -> SemanticMediator:
    return SemanticMediator(**kwargs)
