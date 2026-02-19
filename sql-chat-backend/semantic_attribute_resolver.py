"""
Attribute-Level Semantic Resolution (ALSR) — v6.22

Pure-function module. No side effects, no state, no LLM calls.

Resolves composite attribute phrases emitted by the LLM extractor
(e.g., "aircraft model", "booking price", "flight status") into
schema-grounded qualified column references.

BACKGROUND
----------
The LLM extractor sometimes conflates an entity name with one of its
attributes into a single field value, e.g.:

    entity_type = "aircraft model"          (should be entity="aircraft_type",
                                              attribute="model")
    metric      = "booking price"           (column = booking.price or similar)
    event       = "flight departure time"   (column = flight.departure_time)

ALSR decomposes such composite phrases into:
    entity part   → resolved via entity_resolver
    attribute part → matched against schema columns

The resolved binding surfaces as:
    AttributeBinding.qualified   e.g. "aircraft_type.model"
    entity_override              e.g. "aircraft_type"
    group_by_targets entry       e.g. "aircraft_type.model"

PIPELINE POSITION
-----------------
Extract → **ALSR** → Entity Resolver → Intent Accumulator → Decide → Execute

Phase 1: entity_type decomposition   (before Entity Resolver)
Phase 2: metric / event binding      (before merge into accumulated state)

INVARIANTS
----------
- Pure functions only (deterministic, no side effects)
- Schema-driven: all matching is against the live schema
- No hardcoded domain knowledge (airline, e-commerce, etc.)
- No LLM calls
- Does NOT modify SQL
- Does NOT import from intent_accumulator (avoids circular dependency)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import pure, stateless helpers from entity_resolver.
# We rely on _depluralize and _normalize being stable pure functions;
# they carry no state and have no side effects.
from entity_resolver import resolve_entity, _depluralize, _normalize


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class AttributeBinding:
    """
    A resolved binding from a composite phrase to a qualified column reference.

    Attributes:
        phrase:           Original composite phrase (e.g., "aircraft model")
        table:            Column's bare table name  (e.g., "aircraft_type" if same-table,
                          or "payment" if column found in FK-reachable table)
        column:           Matched column name       (e.g., "model")
        qualified:        Fully qualified reference (e.g., "aircraft_type.model")
        entity_part:      Entity portion of phrase  (e.g., "aircraft")
        attribute_part:   Attribute portion         (e.g., "model")
        entity_table:     ALWAYS the entity part's resolved table name (e.g., "aircraft_type").
                          This is the value propagated back to intent.entity_type.
                          Invariant: entity_table == table for same-table bindings;
                                     entity_table != table for FK-reachable bindings.
        from_entity_table: True if the column was found in the entity table itself.
                           False if it was found in an FK-reachable table.
                           Used for pool-priority ambiguity resolution.
        strategy:         Column match strategy used (e.g., "entity:exact+col:exact")
        confidence:       Overall confidence [0.0, 1.0]
    """
    phrase: str
    table: str
    column: str
    qualified: str
    entity_part: str
    attribute_part: str
    entity_table: str       # Always the entity part's resolved table (audit fix v6.22)
    from_entity_table: bool # True iff column found in entity table itself (audit fix v6.22)
    strategy: str
    confidence: float


@dataclass
class AttributeResolutionResult:
    """
    Result of attribute phrase resolution.

    Attributes:
        status:          "resolved" | "ambiguous" | "unresolved" | "passthrough"
        phrase:          Original phrase
        binding:         Set when status="resolved"
        candidates:      Non-empty when status="ambiguous" (multiple close matches)
        entity_override: Canonical entity name for back-propagation to entity_type
                         (i.e., the value to write back to intent.entity_type).
                         Set when status="resolved" or status="passthrough" with table match.
        reason:          Diagnostic reason for this result
    """
    status: str
    phrase: str
    binding: Optional[AttributeBinding] = None
    candidates: List[AttributeBinding] = field(default_factory=list)
    entity_override: Optional[str] = None
    reason: str = ""


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_attribute_index(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build a {bare_table_name -> {lowercase column names}} index.

    Schema keys may be schema-qualified ("public.booking_leg") or bare.
    The returned index always uses bare names as keys.

    Returns empty dict on invalid schema.
    """
    index: Dict[str, Set[str]] = {}
    if not schema or "tables" not in schema:
        return index

    for table_name, table_info in schema["tables"].items():
        bare = table_name.split(".")[-1].lower()
        cols = {c["name"].lower() for c in table_info.get("columns", [])}
        index[bare] = cols

    return index


def _match_column(
    attr_part: str,
    columns: Set[str],
    entity_bare: Optional[str] = None,
) -> Optional[Tuple[str, str, float]]:
    """
    Match an attribute string against a set of column names.

    Returns (matched_column_name, strategy, confidence) or None if no match.

    Strategies (applied in priority order, with early return on match):
    1. Exact match                         (1.00)
    2. Plural/singular normalization       (0.95)
    3. Multi-token → underscore            (0.90)  "total amount" → "total_amount"
    4. Multi-token underscore + depluralize(0.85)
    5. Table-prefix suffix match           (0.82)  entity_bare + "_" + attr_part
    6. Bare suffix match                   (0.80)  col.endswith("_" + attr_part)
    7. Token overlap (best ratio)          (0.70 * ratio)

    Args:
        attr_part:   Attribute string to match (e.g., "model", "total amount")
        columns:     Set of lowercase column names from the target table
        entity_bare: Optional bare table name for table-prefix suffix matching
    """
    if not attr_part or not columns:
        return None

    a_norm = _normalize(attr_part)                          # lowercase, strip
    a_deplural = _depluralize(a_norm)
    a_underscore = a_norm.replace(" ", "_")
    a_underscore_deplural = _depluralize(a_underscore)

    # Pass 1: Exact match
    for col in columns:
        if a_norm == col:
            return (col, "exact", 1.0)

    # Pass 2: Plural/singular normalization
    for col in columns:
        c_deplural = _depluralize(col)
        if a_deplural == col or a_norm == c_deplural or a_deplural == c_deplural:
            return (col, "plural_singular", 0.95)

    # Pass 3: Multi-token underscore ("total amount" → "total_amount")
    for col in columns:
        if a_underscore == col:
            return (col, "multi_token_underscore", 0.9)

    # Pass 4: Multi-token underscore + depluralize
    for col in columns:
        c_deplural = _depluralize(col)
        if a_underscore_deplural == col or a_underscore == c_deplural:
            return (col, "multi_token_plural", 0.85)

    # Pass 5: Table-prefix suffix match ("booking" + "price" → "booking_price")
    if entity_bare:
        prefixed = entity_bare + "_" + a_underscore
        prefixed_deplural = entity_bare + "_" + a_underscore_deplural
        for col in columns:
            if col in (prefixed, prefixed_deplural, _depluralize(prefixed)):
                return (col, "table_prefix_suffix", 0.82)

    # Pass 6: Bare suffix match (col ends with "_<attr>")
    a_suffix = "_" + a_underscore
    a_suffix_deplural = "_" + a_underscore_deplural
    for col in columns:
        if col.endswith(a_suffix) or col.endswith(a_suffix_deplural):
            return (col, "suffix_match", 0.80)

    # Pass 7: Token overlap — find the best ratio across all columns
    a_tokens = set(a_norm.replace("_", " ").split())
    if not a_tokens:
        return None

    best: Optional[Tuple[str, str, float]] = None
    for col in columns:
        c_tokens = set(col.replace("_", " ").split())
        if not c_tokens:
            continue
        overlap = a_tokens & c_tokens
        if overlap:
            ratio = len(overlap) / max(len(a_tokens), len(c_tokens))
            score = round(0.7 * ratio, 3)
            if score > 0 and (best is None or score > best[2]):
                best = (col, "token_overlap", score)

    return best


def _is_fk_reachable_simple(
    source_bare: str,
    target_bare: str,
    schema: Dict[str, Any],
    max_depth: int = 4,
) -> bool:
    """
    Check if target_bare table is FK-reachable from source_bare via BFS.

    Uses bare table names (no schema prefix).
    Conservative: if source table is unknown, returns True (don't block).

    This is an internal reachability guard for scoping the column search.
    Definitive FK gating is deferred to intent_accumulator._check_fk_projection.

    Args:
        source_bare: Source bare table name (e.g., "booking")
        target_bare: Target bare table name (e.g., "passenger")
        schema:      Database schema dict with "tables" key
        max_depth:   Maximum BFS traversal depth

    Returns:
        True if target is reachable from source, False otherwise
    """
    if source_bare == target_bare:
        return True
    if not schema or "tables" not in schema:
        return True  # No schema → conservative: don't block

    # Build bare-name → fully-qualified name lookup
    bare_to_fq: Dict[str, str] = {}
    for fq_name in schema["tables"]:
        bare = fq_name.split(".")[-1].lower()
        bare_to_fq[bare] = fq_name

    source_fq = bare_to_fq.get(source_bare)
    target_fq = bare_to_fq.get(target_bare)

    if not source_fq:
        return True   # Unknown source → conservative: don't block
    if not target_fq:
        return False  # Unknown target → cannot be reachable

    visited: Set[str] = {source_fq}
    queue = [(source_fq, 0)]

    while queue:
        current_fq, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        table_info = schema["tables"].get(current_fq)
        if not table_info:
            continue

        # Forward FKs: current table → other tables
        for fk in table_info.get("foreign_keys", []):
            fk_target = fk.get("target_table", "")
            if fk_target == target_fq:
                return True
            if fk_target and fk_target not in visited:
                visited.add(fk_target)
                queue.append((fk_target, depth + 1))

        # Reverse FKs: other tables → current table
        for other_fq, other_info in schema["tables"].items():
            if other_fq in visited:
                continue
            for fk in other_info.get("foreign_keys", []):
                if fk.get("target_table") == current_fq:
                    if other_fq == target_fq:
                        return True
                    visited.add(other_fq)
                    queue.append((other_fq, depth + 1))
                    break  # One FK per table is enough for reachability

    return False


def _split_phrase(phrase: str) -> List[Tuple[List[str], List[str]]]:
    """
    Generate all (entity_tokens, attr_tokens) splits for a phrase.

    For phrase with K tokens, produces K-1 splits in left-to-right order
    (entity is typically the leading token(s)):

        "aircraft type model" → [
            (["aircraft"], ["type", "model"]),
            (["aircraft", "type"], ["model"]),
        ]

    Returns empty list for single-token or empty phrases.
    """
    tokens = phrase.strip().split()
    n = len(tokens)
    if n < 2:
        return []
    return [(tokens[:i], tokens[i:]) for i in range(1, n)]


# =============================================================================
# PUBLIC API
# =============================================================================

def build_attribute_index(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Build a {bare_table_name -> {col_names}} index for fast column lookup.

    Pre-computing this index and passing it to resolve_attribute_phrase()
    avoids rebuilding the index on every call in hot paths.

    Args:
        schema: Database schema dict with "tables" key

    Returns:
        Dict mapping bare table names to sets of lowercase column names.
        Empty dict if schema is invalid.
    """
    return _build_attribute_index(schema)


def resolve_attribute_phrase(
    phrase: str,
    schema: Dict[str, Any],
    attribute_index: Optional[Dict[str, Set[str]]] = None,
) -> AttributeResolutionResult:
    """
    Resolve a composite attribute phrase to a schema-grounded column reference.

    Handles composite phrases where the LLM fuses entity + attribute into a
    single field value (e.g., entity_type="aircraft model" instead of
    entity_type="aircraft" with attribute "model").

    ALGORITHM
    ---------
    1. Guard: empty phrase → unresolved
    2. Guard: single token → not a composite phrase → passthrough
    3. Full-phrase entity check: if phrase resolves to a table with confidence
       ≥ 0.85, return "passthrough" (entity_resolver handles it better)
    4. Split search (left-to-right priority):
       a. For each (entity_part, attr_part) split:
          i.  Resolve entity_part via entity_resolver (threshold 0.70)
          ii. Match attr_part in entity table's columns
          iii.If not found in entity table, search FK-reachable tables
       b. Collect all valid AttributeBinding candidates
    5. Deduplication: keep highest-confidence binding per (table, column) pair
    6. Return: resolved (single best) / ambiguous (multiple close) / unresolved

    Args:
        phrase:          Composite phrase from LLM (e.g., "aircraft model")
        schema:          Database schema dict with "tables" key
        attribute_index: Optional pre-built {bare_table -> {cols}} index

    Returns:
        AttributeResolutionResult with status and binding (if resolved)
    """
    # Guard: empty/None phrase
    if not phrase or not phrase.strip():
        return AttributeResolutionResult(
            status="unresolved",
            phrase=phrase or "",
            reason="empty_phrase",
        )

    phrase = phrase.strip()

    # Guard: no schema
    if not schema or "tables" not in schema:
        return AttributeResolutionResult(
            status="passthrough",
            phrase=phrase,
            reason="no_schema",
        )

    # Guard: single token → not a composite phrase
    tokens = phrase.split()
    if len(tokens) < 2:
        return AttributeResolutionResult(
            status="passthrough",
            phrase=phrase,
            reason="single_token",
        )

    # -------------------------------------------------------------------------
    # Passthrough guard: if the full phrase itself resolves to a table with
    # high confidence, entity_resolver handles it. ALSR only applies to phrases
    # that the entity resolver cannot resolve cleanly.
    # -------------------------------------------------------------------------
    full_resolution = resolve_entity(phrase, schema)
    if full_resolution.status == "resolved" and full_resolution.confidence >= 0.85:
        logger.debug(
            f"[ALSR] '{phrase}' passthrough — full phrase resolves to table "
            f"'{full_resolution.canonical_entity}' (score={full_resolution.confidence})"
        )
        return AttributeResolutionResult(
            status="passthrough",
            phrase=phrase,
            entity_override=full_resolution.canonical_entity,
            reason="full_phrase_table_match",
        )

    # Build attribute index if not provided
    if attribute_index is None:
        attribute_index = _build_attribute_index(schema)

    # -------------------------------------------------------------------------
    # Split search: try all (entity_part, attr_part) splits left-to-right.
    # Collect all valid AttributeBinding candidates.
    # -------------------------------------------------------------------------
    candidates: List[AttributeBinding] = []

    for entity_tokens, attr_tokens in _split_phrase(phrase):
        entity_part_str = " ".join(entity_tokens)
        attr_part_str = " ".join(attr_tokens)

        # Resolve entity part via entity_resolver (lower threshold for partial phrases)
        entity_res = resolve_entity(entity_part_str, schema, confidence_threshold=0.70)

        if entity_res.status not in ("resolved", "ambiguous"):
            logger.debug(
                f"[ALSR] '{phrase}' split [{entity_part_str}|{attr_part_str}] "
                f"— entity part unresolved, skipping"
            )
            continue

        # For ambiguous entity, try all candidate tables.
        # For resolved entity, try the single winner.
        entity_candidates = (
            entity_res.candidates
            if entity_res.status == "ambiguous"
            else entity_res.candidates[:1]
        )

        for entity_cand in entity_candidates:
            entity_bare = entity_cand.simple_name
            entity_cols = attribute_index.get(entity_bare, set())

            # ── Primary: search within the entity table's own columns ──────
            col_match = _match_column(attr_part_str, entity_cols, entity_bare)
            if col_match:
                matched_col, strategy, col_conf = col_match
                overall_conf = round(entity_cand.score * col_conf, 3)
                binding = AttributeBinding(
                    phrase=phrase,
                    table=entity_bare,
                    column=matched_col,
                    qualified=f"{entity_bare}.{matched_col}",
                    entity_part=entity_part_str,
                    attribute_part=attr_part_str,
                    entity_table=entity_bare,     # same table → entity_table == table
                    from_entity_table=True,
                    strategy=f"entity:{entity_cand.match_strategy}+col:{strategy}",
                    confidence=overall_conf,
                )
                candidates.append(binding)
                logger.debug(
                    f"[ALSR] '{phrase}' split [{entity_part_str}|{attr_part_str}] "
                    f"→ entity={entity_bare}, col={matched_col} "
                    f"(conf={overall_conf})"
                )
                continue  # Entity table matched — don't also search FK tables

            # ── Secondary: search FK-reachable tables ─────────────────────
            # Apply a hop penalty (×0.9) to prefer same-table matches.
            for other_bare, other_cols in attribute_index.items():
                if other_bare == entity_bare:
                    continue
                if not _is_fk_reachable_simple(entity_bare, other_bare, schema):
                    continue
                col_match = _match_column(attr_part_str, other_cols, other_bare)
                if col_match:
                    matched_col, strategy, col_conf = col_match
                    overall_conf = round(entity_cand.score * col_conf * 0.9, 3)
                    binding = AttributeBinding(
                        phrase=phrase,
                        table=other_bare,           # column's FK table
                        column=matched_col,
                        qualified=f"{other_bare}.{matched_col}",
                        entity_part=entity_part_str,
                        attribute_part=attr_part_str,
                        entity_table=entity_bare,   # always the entity part's table
                        from_entity_table=False,    # column is in a FK-reachable table
                        strategy=f"entity:{entity_cand.match_strategy}+fk_col:{strategy}",
                        confidence=overall_conf,
                    )
                    candidates.append(binding)
                    logger.debug(
                        f"[ALSR] '{phrase}' split [{entity_part_str}|{attr_part_str}] "
                        f"→ entity={entity_bare}, fk_table={other_bare}, col={matched_col} "
                        f"(conf={overall_conf})"
                    )

    if not candidates:
        logger.info(f"[ALSR] '{phrase}' → unresolved (no entity+column split found)")
        return AttributeResolutionResult(
            status="unresolved",
            phrase=phrase,
            reason="no_split_found",
        )

    # -------------------------------------------------------------------------
    # Deduplication: keep highest-confidence binding per (table, column) pair.
    # -------------------------------------------------------------------------
    seen: Dict[Tuple[str, str], AttributeBinding] = {}
    for b in candidates:
        key = (b.table, b.column)
        if key not in seen or b.confidence > seen[key].confidence:
            seen[key] = b

    unique = sorted(seen.values(), key=lambda b: b.confidence, reverse=True)

    # Single unique binding → unambiguous resolved
    if len(unique) == 1:
        winner = unique[0]
        logger.info(
            f"[ALSR] '{phrase}' → resolved: {winner.qualified} "
            f"(conf={winner.confidence}, strategy={winner.strategy})"
        )
        return AttributeResolutionResult(
            status="resolved",
            phrase=phrase,
            binding=winner,
            candidates=unique,
            entity_override=winner.entity_table,   # audit fix: always entity part's table
            reason="single_binding",
        )

    # -------------------------------------------------------------------------
    # Pool-priority ambiguity resolution (audit fix v6.22):
    #
    # Entity-table bindings (from_entity_table=True) structurally outrank
    # FK-table bindings (from_entity_table=False) regardless of confidence gap.
    # The entity's own attributes are unambiguously primary.
    #
    # Confidence-gap ambiguity is applied ONLY within the winning pool.
    # This prevents the 0.9-vs-1.0 case (entity table exact, FK table exact
    # with hop penalty) from incorrectly triggering user clarification.
    # -------------------------------------------------------------------------
    entity_table_pool = [b for b in unique if b.from_entity_table]
    primary_pool = entity_table_pool if entity_table_pool else unique

    top_conf = primary_pool[0].confidence
    close = [b for b in primary_pool if top_conf - b.confidence <= 0.15]

    if len(close) > 1:
        logger.info(
            f"[ALSR] '{phrase}' → ambiguous within {'entity' if entity_table_pool else 'fk'} pool "
            f"({len(close)} candidates: {[b.qualified for b in close]})"
        )
        return AttributeResolutionResult(
            status="ambiguous",
            phrase=phrase,
            candidates=close,
            reason="multiple_close_bindings",
        )

    # Single winner from the primary pool
    winner = primary_pool[0]
    logger.info(
        f"[ALSR] '{phrase}' → resolved: {winner.qualified} "
        f"(conf={winner.confidence}, strategy={winner.strategy}, "
        f"entity_table={winner.entity_table})"
    )
    return AttributeResolutionResult(
        status="resolved",
        phrase=phrase,
        binding=winner,
        candidates=unique,
        entity_override=winner.entity_table,   # audit fix: always entity part's table
        reason="highest_confidence_winner",
    )


# =============================================================================
# QUERY HINT FORMATTER (v6.22 audit fix — Invariant 3)
# =============================================================================

def format_alsr_query_hint(
    query: str,
    group_by_targets: List[str],
    attribute_bindings: List[Any],
) -> str:
    """
    Append ALSR-resolved column references to the NL query as a structural hint.

    This function is the synchronization bridge between ALSR's schema-grounded
    resolution and the NL-SQL engine. Without it, ALSR's bindings exist only
    in IntentState and are never communicated to the LLM, causing the LLM to
    rediscover column references independently (potential semantic drift).

    Follows the same query-enrichment pattern as semantic role hints in main.py
    (get_fk_hint_for_query appends structured context to the query string).
    The LLM sees the enriched query as its full input; the hint is deterministic,
    schema-derived, and database-agnostic.

    DESIGN INVARIANTS
    -----------------
    - Deterministic: same bindings always produce the same hint
    - Database-agnostic: uses bare table.column format, not schema-qualified
    - Non-modifying: returns original query unchanged if no bindings
    - No SQL generation: hint describes column mappings only
    - Pure function: no side effects, no state

    EXAMPLE OUTPUT
    --------------
    Input query:  "Count passengers per aircraft model"
    Bindings:     [AttributeBinding(phrase='aircraft model', qualified='aircraft.model')]
    Output query: "Count passengers per aircraft model
                   [Column mappings: 'aircraft model' → aircraft.model]"

    Args:
        query:              NL query string (from get_query_for_nlsql)
        group_by_targets:   List of qualified column refs for GROUP BY hints
                            (e.g., ["aircraft.model"])
        attribute_bindings: List of AttributeBinding objects (may be List[Any])

    Returns:
        Enriched query string, or original query if nothing to inject.
    """
    if not attribute_bindings and not group_by_targets:
        return query

    # Build phrase → qualified column mapping from resolved bindings.
    # Each mapping tells the LLM what specific column a composite phrase refers to.
    mappings: List[str] = []
    seen_phrases: set = set()

    for b in attribute_bindings:
        phrase    = getattr(b, 'phrase', None)
        qualified = getattr(b, 'qualified', None)
        if phrase and qualified and phrase not in seen_phrases:
            mappings.append(f"'{phrase}' → {qualified}")
            seen_phrases.add(phrase)

    if not mappings:
        # No phrase→column mappings — use group_by_targets as a bare hint
        if group_by_targets:
            return query + f"\n[Group by column(s): {', '.join(group_by_targets)}]"
        return query

    hint = "\n[Column mappings: " + "; ".join(mappings) + "]"
    logger.debug(f"[ALSR] Query hint injected: {hint.strip()}")
    return query + hint
