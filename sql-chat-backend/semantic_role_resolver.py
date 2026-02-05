"""
SemanticRoleResolver - Pre-NL-SQL FK Hint Layer

Detects semantic roles (arrival vs departure, etc.) and provides FK hints
to guide NL-SQL toward the correct column.

CONSERVATIVE: Only emits hints when role is unambiguous.
If conflicting roles detected -> no hint.
RCL still validates everything downstream.
"""

import re
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SemanticRole:
    """
    Represents a semantic role that maps to a specific FK preference.

    Attributes:
        name: Human-readable role name (e.g., "arrival", "departure")
        patterns: Regex patterns that indicate this role
        fk_preferences: Dict mapping table -> preferred FK column
    """
    name: str
    patterns: List[str]  # Regex patterns (case-insensitive)
    fk_preferences: Dict[str, str]  # {table_name: fk_column}


@dataclass
class RoleResolutionResult:
    """
    Result of semantic role resolution.

    Attributes:
        detected_roles: Set of role names detected in query
        preferred_fks: Dict of table -> preferred FK column (empty if ambiguous)
        is_ambiguous: True if conflicting roles detected
        hint_applied: True if a hint was successfully resolved
        reason: Explanation of resolution
    """
    detected_roles: Set[str] = field(default_factory=set)
    preferred_fks: Dict[str, str] = field(default_factory=dict)
    is_ambiguous: bool = False
    hint_applied: bool = False
    reason: str = ""


# =============================================================================
# ROLE DEFINITIONS (Schema-Driven, Extensible)
# =============================================================================
# These define semantic roles and their FK mappings.
# Each role has:
#   - patterns: words/phrases that indicate this role
#   - fk_preferences: which FK to prefer when this role is detected
#
# IMPORTANT: These are NOT hardcoded table names in the sense of being
# globally assumed. They are mappings that only activate when:
#   1. The role pattern is detected in the user query
#   2. The table exists in the schema
#   3. The FK column exists on that table
# =============================================================================

# Aviation domain roles (arrival vs departure)
ARRIVAL_ROLE = SemanticRole(
    name="arrival",
    patterns=[
        r'\barrival[s]?\b',
        r'\barriving\b',
        r'\barrived\b',
        r'\binbound\b',
        r'\binto\b',           # "flights into JFK"
        r'\bdestination\b',
        r'\bto\s+(?:airport|city)\b',
        r'\bland(?:ing|ed|s)?\b',
    ],
    fk_preferences={
        "flight": "arrival_airport",
        "postgres_air.flight": "arrival_airport",
    }
)

DEPARTURE_ROLE = SemanticRole(
    name="departure",
    patterns=[
        r'\bdeparture[s]?\b',
        r'\bdeparting\b',
        r'\bdeparted\b',
        r'\boutbound\b',
        r'\bfrom\s+(?:airport|city)\b',
        r'\borigin\b',
        r'\boriginating\b',
        r'\bleaving\b',
        r'\btake\s*off[s]?\b',
    ],
    fk_preferences={
        "flight": "departure_airport",
        "postgres_air.flight": "departure_airport",
    }
)

# Passenger domain roles (could extend for other domains)
PASSENGER_ORIGIN_ROLE = SemanticRole(
    name="passenger_origin",
    patterns=[
        r'\bpassenger[s]?\s+from\b',
        r'\btraveler[s]?\s+from\b',
    ],
    fk_preferences={
        "booking": "departure_airport",
    }
)

PASSENGER_DESTINATION_ROLE = SemanticRole(
    name="passenger_destination",
    patterns=[
        r'\bpassenger[s]?\s+to\b',
        r'\btraveler[s]?\s+to\b',
    ],
    fk_preferences={
        "booking": "arrival_airport",
    }
)

# Default role registry
DEFAULT_ROLES = [
    ARRIVAL_ROLE,
    DEPARTURE_ROLE,
    PASSENGER_ORIGIN_ROLE,
    PASSENGER_DESTINATION_ROLE,
]

# Define conflicting role pairs (roles that cannot coexist)
CONFLICTING_ROLE_PAIRS = [
    ("arrival", "departure"),
    ("passenger_origin", "passenger_destination"),
]


# =============================================================================
# SEMANTIC ROLE RESOLVER
# =============================================================================

class SemanticRoleResolver:
    """
    Resolves semantic roles from user queries to provide FK hints.

    BEHAVIOR:
    1. Scan query for role patterns
    2. If exactly ONE role detected for a table -> emit hint
    3. If MULTIPLE conflicting roles -> emit NO hint (conservative)
    4. If NO role detected -> emit NO hint (passthrough)

    CRITICAL GUARANTEES:
    - NEVER generates SQL
    - NEVER guesses when ambiguous
    - NEVER bypasses downstream validation
    - Only provides ADVISORY hints
    """

    def __init__(
        self,
        roles: Optional[List[SemanticRole]] = None,
        conflicting_pairs: Optional[List[tuple]] = None
    ):
        """
        Initialize with role definitions.

        Args:
            roles: List of SemanticRole definitions (uses defaults if None)
            conflicting_pairs: Pairs of role names that conflict
        """
        self.roles = roles or DEFAULT_ROLES
        self.conflicting_pairs = conflicting_pairs or CONFLICTING_ROLE_PAIRS

        # Compile patterns for efficiency
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for role in self.roles:
            self._compiled_patterns[role.name] = [
                re.compile(p, re.IGNORECASE) for p in role.patterns
            ]

        logger.info(f"[SRR] Initialized with {len(self.roles)} semantic roles")

    def resolve(self, user_query: str) -> RoleResolutionResult:
        """
        Resolve semantic roles from a user query.

        Args:
            user_query: The natural language query

        Returns:
            RoleResolutionResult with detected roles and FK preferences
        """
        if not user_query or not user_query.strip():
            return RoleResolutionResult(reason="Empty query")

        query_lower = user_query.lower()

        # Step 1: Detect all matching roles
        detected_roles: Set[str] = set()
        role_objects: Dict[str, SemanticRole] = {}

        for role in self.roles:
            patterns = self._compiled_patterns[role.name]
            for pattern in patterns:
                if pattern.search(query_lower):
                    detected_roles.add(role.name)
                    role_objects[role.name] = role
                    logger.debug(f"[SRR] Detected role '{role.name}' via pattern '{pattern.pattern}'")
                    break  # One match per role is enough

        if not detected_roles:
            logger.debug("[SRR] No semantic roles detected")
            return RoleResolutionResult(
                reason="No semantic roles detected in query"
            )

        # Step 2: Check for conflicting roles
        for role_a, role_b in self.conflicting_pairs:
            if role_a in detected_roles and role_b in detected_roles:
                logger.info(
                    f"[SRR] Conflicting roles detected: {role_a} vs {role_b}. "
                    "No hint will be applied (conservative)."
                )
                return RoleResolutionResult(
                    detected_roles=detected_roles,
                    is_ambiguous=True,
                    reason=f"Conflicting roles: {role_a} and {role_b}"
                )

        # Step 3: Build FK preferences (merge non-conflicting roles)
        preferred_fks: Dict[str, str] = {}
        table_conflicts: Dict[str, Set[str]] = {}  # table -> set of preferred FKs

        for role_name in detected_roles:
            role = role_objects[role_name]
            for table, fk_col in role.fk_preferences.items():
                if table not in table_conflicts:
                    table_conflicts[table] = set()
                table_conflicts[table].add(fk_col)

        # Only include tables with exactly ONE preferred FK
        for table, fk_set in table_conflicts.items():
            if len(fk_set) == 1:
                preferred_fks[table] = list(fk_set)[0]
            else:
                logger.warning(
                    f"[SRR] Multiple FK preferences for table '{table}': {fk_set}. "
                    "Skipping hint for this table."
                )

        if not preferred_fks:
            return RoleResolutionResult(
                detected_roles=detected_roles,
                is_ambiguous=True,
                reason="All detected roles produced conflicting FK preferences"
            )

        logger.info(
            f"[SRR] Resolved semantic roles: {detected_roles} -> "
            f"FK preferences: {preferred_fks}"
        )

        return RoleResolutionResult(
            detected_roles=detected_roles,
            preferred_fks=preferred_fks,
            hint_applied=True,
            reason=f"Detected roles: {', '.join(detected_roles)}"
        )


# =============================================================================
# HINT FORMATTING FOR NL-SQL
# =============================================================================

def format_fk_hint_for_prompt(
    result: RoleResolutionResult,
    schema_tables: Optional[Set[str]] = None
) -> str:
    """
    Format FK preferences as a hint string for NL-SQL prompts.

    This creates a human-readable hint that nudges the LLM toward
    using the correct FK column.

    Args:
        result: The role resolution result
        schema_tables: Optional set of actual table names in schema
                      (used to filter hints to existing tables)

    Returns:
        Hint string to append to NL-SQL prompt, or empty string

    IMPORTANT:
    This hint is ADVISORY. It does not force the LLM to do anything.
    RCL will still validate the generated SQL.
    """
    if not result.hint_applied or not result.preferred_fks:
        return ""

    # Filter to tables that exist in schema (if provided)
    relevant_fks = result.preferred_fks
    if schema_tables:
        relevant_fks = {
            t: fk for t, fk in result.preferred_fks.items()
            if t.lower() in {s.lower() for s in schema_tables} or
               t.split(".")[-1].lower() in {s.split(".")[-1].lower() for s in schema_tables}
        }

    if not relevant_fks:
        return ""

    # Build hint text
    hints = []
    for table, fk_col in relevant_fks.items():
        table_name = table.split(".")[-1]  # Remove schema prefix for readability
        hints.append(f"- For {table_name}, prefer using '{fk_col}' based on query context")

    hint_text = (
        "\n\nSEMANTIC HINT (based on user query analysis):\n"
        "The user's query suggests the following column preferences:\n" +
        "\n".join(hints) +
        "\nUse these columns when joining to related tables if applicable."
    )

    return hint_text


def format_fk_hint_as_dict(result: RoleResolutionResult) -> Dict:
    """
    Format FK preferences as a structured dict.

    This can be passed to systems that accept structured hints.

    Returns:
        {
            "preferred_fk": {"table": "column", ...},
            "detected_roles": ["role1", "role2"],
            "reason": "..."
        }
    """
    return {
        "preferred_fk": result.preferred_fks,
        "detected_roles": list(result.detected_roles),
        "hint_applied": result.hint_applied,
        "is_ambiguous": result.is_ambiguous,
        "reason": result.reason
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global resolver instance (lazy initialization)
_resolver: Optional[SemanticRoleResolver] = None


def get_resolver() -> SemanticRoleResolver:
    """Get or create the global resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = SemanticRoleResolver()
    return _resolver


def resolve_semantic_roles(user_query: str) -> RoleResolutionResult:
    """
    Convenience function to resolve semantic roles.

    Args:
        user_query: Natural language query

    Returns:
        RoleResolutionResult
    """
    return get_resolver().resolve(user_query)


def get_fk_hint_for_query(
    user_query: str,
    schema_tables: Optional[Set[str]] = None
) -> str:
    """
    Get FK hint string for a user query.

    Args:
        user_query: Natural language query
        schema_tables: Optional set of table names in schema

    Returns:
        Hint string for NL-SQL prompt (empty if no hint)
    """
    result = resolve_semantic_roles(user_query)
    return format_fk_hint_for_prompt(result, schema_tables)
