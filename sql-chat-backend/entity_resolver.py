"""
Entity Resolver — Schema-Driven Entity Resolution (v6.15)

Pure-function module. No side effects, no state, no LLM calls.

Resolves free-form entity names emitted by the LLM extractor (e.g.,
"customer", "frequent flyers") to actual schema table names using
deterministic string-matching strategies.

INVARIANTS:
- Pure functions only (deterministic, no side effects)
- Schema-driven: all matching is against the live schema
- No hardcoded domain knowledge (airline, e-commerce, etc.)
- No LLM calls
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class EntityCandidate:
    """
    A single candidate table match for an entity name.

    Attributes:
        table_name: Fully-qualified table name (e.g., "airline.passenger")
        simple_name: Table name without schema prefix (e.g., "passenger")
        score: Match confidence (0.0-1.0)
        match_strategy: How this match was found
    """
    table_name: str
    simple_name: str
    score: float
    match_strategy: str


@dataclass
class EntityResolutionResult:
    """
    Result of entity resolution.

    Attributes:
        status: "resolved" | "ambiguous" | "unresolved"
        raw_entity: Original LLM output
        resolved_table: Fully-qualified table name (if resolved)
        canonical_entity: Simple table name without schema prefix (if resolved)
        candidates: All matches with scores
        confidence: Top candidate confidence (0.0-1.0)
    """
    status: str
    raw_entity: str
    resolved_table: Optional[str] = None
    canonical_entity: Optional[str] = None
    candidates: List[EntityCandidate] = field(default_factory=list)
    confidence: float = 0.0


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _extract_simple_name(table_name: str) -> str:
    """Extract simple table name from fully-qualified name."""
    return table_name.split(".")[-1].lower()


def _depluralize(word: str) -> str:
    """
    Simple English depluralization.

    Handles common patterns:
    - "ies" → "y" (e.g., "categories" → "category")
    - "ses" / "xes" / "zes" / "ches" / "shes" → remove "es"
    - "s" → remove "s"
    """
    w = word.lower()
    if len(w) <= 2:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes")):
        return w[:-2]
    if w.endswith(("ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _score_candidates(
    raw_entity: str,
    schema: Dict[str, Any],
) -> List[EntityCandidate]:
    """
    Score all schema tables against the raw entity string.

    Applies strategies in priority order. Each table gets the score
    from the highest-priority strategy that matches.

    Strategies (applied in order):
    1. Exact match (score=1.0)
    2. Plural/singular normalization (score=0.95)
    3. Multi-token underscore (score=0.9)
    4. Multi-token + plural (score=0.85)
    5. Token overlap (score=0.7 * overlap ratio)
    """
    if not schema or "tables" not in schema:
        return []

    entity_norm = _normalize(raw_entity)
    entity_deplural = _depluralize(entity_norm)
    entity_underscored = entity_norm.replace(" ", "_")
    entity_underscored_deplural = _depluralize(entity_underscored)

    candidates = []

    for table_name in schema["tables"]:
        simple = _extract_simple_name(table_name)

        # Strategy 1: Exact match
        if entity_norm == simple:
            candidates.append(EntityCandidate(
                table_name=table_name,
                simple_name=simple,
                score=1.0,
                match_strategy="exact",
            ))
            continue

        # Strategy 2: Plural/singular normalization
        simple_deplural = _depluralize(simple)
        if entity_deplural == simple or entity_norm == simple_deplural or entity_deplural == simple_deplural:
            candidates.append(EntityCandidate(
                table_name=table_name,
                simple_name=simple,
                score=0.95,
                match_strategy="plural_singular",
            ))
            continue

        # Strategy 3: Multi-token underscore match
        # "frequent flyer" → "frequent_flyer"
        if entity_underscored == simple:
            candidates.append(EntityCandidate(
                table_name=table_name,
                simple_name=simple,
                score=0.9,
                match_strategy="multi_token_underscore",
            ))
            continue

        # Strategy 4: Multi-token + plural
        # "frequent flyers" → "frequent_flyer"
        if entity_underscored_deplural == simple or entity_underscored == simple_deplural:
            candidates.append(EntityCandidate(
                table_name=table_name,
                simple_name=simple,
                score=0.85,
                match_strategy="multi_token_plural",
            ))
            continue

        # Strategy 5: Token overlap
        # "customer" vs "customer_account" → partial match
        entity_tokens = set(entity_norm.replace("_", " ").split())
        table_tokens = set(simple.replace("_", " ").split())

        if entity_tokens and table_tokens:
            overlap = entity_tokens & table_tokens
            if overlap:
                # Score based on how much of the entity is covered
                ratio = len(overlap) / max(len(entity_tokens), len(table_tokens))
                score = round(0.7 * ratio, 3)
                if score > 0:
                    candidates.append(EntityCandidate(
                        table_name=table_name,
                        simple_name=simple,
                        score=score,
                        match_strategy="token_overlap",
                    ))
                    continue

            # Also try depluralized token overlap
            entity_tokens_deplural = {_depluralize(t) for t in entity_tokens}
            table_tokens_deplural = {_depluralize(t) for t in table_tokens}
            overlap_deplural = entity_tokens_deplural & table_tokens_deplural
            if overlap_deplural:
                ratio = len(overlap_deplural) / max(len(entity_tokens_deplural), len(table_tokens_deplural))
                score = round(0.7 * ratio, 3)
                if score > 0:
                    candidates.append(EntityCandidate(
                        table_name=table_name,
                        simple_name=simple,
                        score=score,
                        match_strategy="token_overlap_deplural",
                    ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# =============================================================================
# PUBLIC API
# =============================================================================

def resolve_entity(
    raw_entity: str,
    schema: Dict[str, Any],
    confidence_threshold: float = 0.6,
) -> EntityResolutionResult:
    """
    Resolve a free-form entity name to a schema table.

    Pure function. No side effects, no state, no LLM calls.

    Args:
        raw_entity: Entity name from LLM extractor (e.g., "customer", "flights")
        schema: Database schema dict with "tables" key
        confidence_threshold: Minimum score for a candidate to be considered

    Returns:
        EntityResolutionResult with status, resolved table, and candidates
    """
    # Guard: empty/None entity
    if not raw_entity or not raw_entity.strip():
        logger.debug("[ENTITY-RESOLVER] Empty entity → unresolved")
        return EntityResolutionResult(
            status="unresolved",
            raw_entity=raw_entity or "",
        )

    # Guard: empty/None schema
    if not schema or "tables" not in schema or not schema["tables"]:
        logger.debug("[ENTITY-RESOLVER] No schema → unresolved")
        return EntityResolutionResult(
            status="unresolved",
            raw_entity=raw_entity,
        )

    # Score all candidates
    all_candidates = _score_candidates(raw_entity, schema)

    # Filter by threshold
    qualified = [c for c in all_candidates if c.score >= confidence_threshold]

    if not qualified:
        logger.info(
            f"[ENTITY-RESOLVER] '{raw_entity}' → unresolved "
            f"(no candidates above threshold {confidence_threshold})"
        )
        return EntityResolutionResult(
            status="unresolved",
            raw_entity=raw_entity,
            candidates=all_candidates,
            confidence=all_candidates[0].score if all_candidates else 0.0,
        )

    if len(qualified) == 1:
        winner = qualified[0]
        logger.info(
            f"[ENTITY-RESOLVER] '{raw_entity}' → resolved to "
            f"'{winner.table_name}' (score={winner.score}, "
            f"strategy={winner.match_strategy})"
        )
        return EntityResolutionResult(
            status="resolved",
            raw_entity=raw_entity,
            resolved_table=winner.table_name,
            canonical_entity=winner.simple_name,
            candidates=qualified,
            confidence=winner.score,
        )

    # Multiple qualified candidates → ambiguous
    logger.info(
        f"[ENTITY-RESOLVER] '{raw_entity}' → ambiguous "
        f"({len(qualified)} candidates: "
        f"{[c.simple_name for c in qualified]})"
    )
    return EntityResolutionResult(
        status="ambiguous",
        raw_entity=raw_entity,
        candidates=qualified,
        confidence=qualified[0].score,
    )
