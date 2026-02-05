"""
OptimaX v6.1 - NL-SQL Query Interface

ARCHITECTURE:
    Query -> Intent Extraction -> Accumulation -> NL-SQL -> RCL -> Execution

KEY MODULES:
    - intent_accumulator.py: Sole clarification authority (entity + metric -> proceed)
    - query_pipeline.py: Pure orchestration
    - relational_corrector.py: FK-based SQL correction
    - semantic_role_resolver.py: Pre-NL-SQL hints

INVARIANTS:
    - Database agnostic (uses runtime FK metadata)
    - Fail-fast for invalid SQL
    - No confidence scores or magic numbers
"""

# =============================================================================
# CRITICAL: ENVIRONMENT GUARD MUST BE FIRST
# =============================================================================
# This validates the Python environment before ANY other imports.
# Ensures we're in the correct venv with all dependencies available.
# =============================================================================
from env_guard import validate_environment, quick_verify_llamaindex

# Run validation (strict=False allows startup to continue with warnings)
# Set strict=True for production to hard-fail on environment issues
_env_valid = validate_environment(strict=False)
if not _env_valid:
    import sys
    print("\n[WARNING] Environment validation failed. See errors above.", flush=True)
    print("[WARNING] Continuing anyway - some features may not work.\n", flush=True)

# Quick LlamaIndex verification
print("\n[LLAMAINDEX] Quick verification...", flush=True)
_llama_ok, _llama_msgs = quick_verify_llamaindex()
for _msg in _llama_msgs:
    print(f"  {_msg}", flush=True)
if not _llama_ok:
    print("[WARNING] LlamaIndex verification failed. Run: python verify_llamaindex.py --verbose", flush=True)

# =============================================================================
# CRITICAL: LOGGING CONFIGURATION MUST BE FIRST
# =============================================================================
# This MUST happen before ANY other imports that create loggers.
# Uses force=True to ensure logging works even after Uvicorn reload.
# =============================================================================
import logging
import sys
import os

# Force completely unbuffered output (critical for uvicorn --reload)
os.environ['PYTHONUNBUFFERED'] = '1'

# Force unbuffered stdout/stderr for real-time logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()


# Configure root logger with force=True (clears existing handlers)
# Use custom flushing handler for immediate output
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Add our flushing handler
_handler = FlushingStreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
root_logger.addHandler(_handler)

# Immediately verify it works
print("[LOGGING] === Logging system initializing ===", flush=True)
logging.getLogger("main").info("[LOGGING] Root logger configured with FlushingStreamHandler")

# Configure all our module loggers
_MODULE_LOGGERS = [
    "main", "tools", "sql_validator", "query_pipeline",
    "intent_accumulator", "semantic_intent_extractor",
    "relational_corrector", "semantic_role_resolver",
    "join_path_inference", "semantic_mediation", "context_resolver"
]

for _name in _MODULE_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    _lg.propagate = True

# Reduce noise from third-party libraries
for _name in ["httpx", "httpcore", "urllib3", "sqlalchemy.engine"]:
    logging.getLogger(_name).setLevel(logging.WARNING)

print("[LOGGING] === All module loggers configured ===", flush=True)

# =============================================================================
# END LOGGING CONFIGURATION
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
import asyncio
from contextlib import asynccontextmanager

from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent

# === NL-SQL Engine ===
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

# Disable embeddings (we use NL-SQL, not RAG)
Settings.embed_model = None

from tools import (
    initialize_tools,
    get_last_sql_result,
    clear_last_sql_result,
    set_last_sql_result,
    classify_visualization_intent,
    classify_query_intent,
    DatabaseManager,
)

from join_path_inference import (
    SchemaGraph,
    format_join_guidance,
    identify_tables_for_query,
)

# === Query Pipeline ===
from query_pipeline import QueryPipeline, PipelineResult

# === Optional Modules (graceful degradation if missing) ===
try:
    from semantic_mediation import SemanticMediator, create_semantic_mediator
    SEMANTIC_MEDIATION_AVAILABLE = True
except ImportError:
    SEMANTIC_MEDIATION_AVAILABLE = False

try:
    from semantic_intent_extractor import SemanticIntentExtractor, SemanticIntent, ExtractionResult
    SEMANTIC_INTENT_AVAILABLE = True
except ImportError:
    SEMANTIC_INTENT_AVAILABLE = False

try:
    from intent_accumulator import (
        IntentState,
        IntentMerger,
        PendingAmbiguity,
        create_intent_state,
        create_intent_merger,
        evaluate,
        format_clarification,
        get_query_for_nlsql,
        # v6.1.1: Relational ambiguity handling
        generate_ambiguity_clarification,
        match_ambiguity_response,
        format_ambiguity_clarification,
        # v6.3: Schema-backed clarification prompts
        set_schema_reference,
    )
    INTENT_ACCUMULATION_AVAILABLE = True
except ImportError:
    INTENT_ACCUMULATION_AVAILABLE = False

try:
    from context_resolver import (
        SessionContext,
        ContextResolver,
        ResultContextBinder,
        RouteBinding,
        RouteDetector,
        create_session_context,
        create_context_resolver,
        create_result_binder,
        create_route_detector,
    )
    CONTEXT_RESOLUTION_AVAILABLE = True
except ImportError:
    CONTEXT_RESOLUTION_AVAILABLE = False
    logger.warning("Context resolution module not available - proceeding without it")

try:
    from sql_validator import (
        SQLAliasValidator,
        QueryComplexityAnalyzer,
        SQLOutputSanitizer,  # v5.0.2: Strict SQL output contract
        ColumnExistenceValidator,  # v5.0.2: Pre-execution column validation
        AliasValidationResult,
        QueryComplexityResult,
        SanitizationResult,  # v5.0.2: Sanitization result type
        ColumnValidationResult,  # v5.0.2: Column validation result type
        validate_sql_aliases,
        analyze_query_complexity,
        sanitize_sql_output,  # v5.0.2: Convenience function
        validate_column_existence,  # v5.0.2: Column validation function
        create_alias_validator,
        create_complexity_analyzer,
        create_sql_sanitizer,  # v5.0.2: Factory function
        create_column_validator,  # v5.0.2: Column validator factory
    )
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    logger.warning("SQL validation module not available - proceeding without it")

try:
    from relational_corrector import (
        RelationalCorrector,
        CorrectionResult,
        RelationalAmbiguity,
        AmbiguityOption,
        create_relational_corrector,
        correct_sql,
    )
    RELATIONAL_CORRECTION_AVAILABLE = True
except ImportError:
    RELATIONAL_CORRECTION_AVAILABLE = False
    logger.warning("Relational corrector not available - proceeding without it")

try:
    from semantic_role_resolver import (
        SemanticRoleResolver,
        RoleResolutionResult,
        resolve_semantic_roles,
        get_fk_hint_for_query,
        format_fk_hint_as_dict,
    )
    SEMANTIC_ROLE_RESOLVER_AVAILABLE = True
except ImportError:
    SEMANTIC_ROLE_RESOLVER_AVAILABLE = False
    logger.warning("Semantic role resolver not available - proceeding without it")

load_dotenv()

# Logger for this module (uses centralized config from top of file)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize system on startup, cleanup on shutdown.

    v5.0 CHANGES:
    - NL-SQL engine is now the PRIMARY path for SQL generation
    - Schema is NOT injected into prompts (NL-SQL reads it via SQLAlchemy)
    - DJPI is initialized but used for ADVISORY purposes only
    - ReActAgent is kept for non-SQL queries and clarifications
    """
    global agent, db_manager, llm, custom_prompt_config, schema_graph
    global sql_database, nl_sql_engine  # v5.0: NL-SQL components
    global semantic_mediator  # v5.0: POLICY LAYER - must be global for chat endpoint
    global intent_extractor  # v5.1: LLM-based intent understanding
    global intent_merger  # v5.3: Phase 3.5 intent accumulation
    global context_resolver, result_binder  # v5.0: Context resolution
    global alias_validator, complexity_analyzer, sql_sanitizer  # v5.0: SQL guardrails
    global relational_corrector  # v6.1: FK-based SQL correction
    global query_pipeline  # v5.3.2: Query orchestration pipeline

    try:
        # === LOGGING VERIFICATION ===
        print("\n" + "="*60, flush=True)
        print("[STARTUP] OptimaX backend starting", flush=True)
        print("="*60, flush=True)
        logger.info("[STARTUP] Logging system active - starting initialization...")
        # === END LOGGING VERIFICATION ===

        logger.info("Initializing OptimaX v5.3 (NL-SQL + Intent Accumulation - Phase 3.5)...")

        # Verify API key
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found! Set it in your environment. "
                "You can get one at: https://console.groq.com/keys"
            )

        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not found in environment variables!")

        # === LLM Initialization ===
        llm = Groq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.1,  # v5.0: Lower temperature for more deterministic SQL
            max_output_tokens=1024,  # v5.0: Increased for complex SQL queries
        )

        # Set LLM globally in LlamaIndex Settings (ensures no OpenAI fallback)
        Settings.llm = llm

        logger.info(f"[OK] Groq LLM initialized: {GROQ_MODEL}")
        logger.info(f"[OK] Embeddings DISABLED (NL-SQL does not require vector similarity)")

        # Initialize tools and get database manager (for validation & schema info)
        tools, db_manager = initialize_tools(DATABASE_URL)
        logger.info(f"[OK] Tools initialized: {len(tools)} tools")

        # === NL-SQL Engine Initialization ===
        engine = create_engine(DATABASE_URL)

        # Get separated schema and table names for NL-SQL initialization
        nl_sql_config = db_manager.get_tables_for_nl_sql()
        raw_table_names = nl_sql_config["tables"]
        detected_schema = nl_sql_config["schema"]

        logger.info(f"[OK] Discovered {len(raw_table_names)} tables" +
                   (f" in schema '{detected_schema}'" if detected_schema else ""))
        logger.info(f"  Tables: {raw_table_names[:5]}{'...' if len(raw_table_names) > 5 else ''}")

        # SQLDatabase: schema passed separately from table names
        sql_database = SQLDatabase(
            engine,
            schema=detected_schema,         # Schema passed separately (or None)
            include_tables=raw_table_names  # Raw table names only (no schema prefix)
        )

        nl_sql_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=raw_table_names,
            llm=llm,
            synthesize_response=False,
            verbose=True,
        )
        logger.info(f"[OK] NL-SQL Engine initialized with {len(raw_table_names)} table constraints")

        # === DJPI Schema Graph (advisory mode - for explainability) ===
        schema_graph = SchemaGraph()
        schema_dict = {}
        for table_name, table_info in db_manager.schema["tables"].items():
            schema_dict[table_name] = [
                {"name": col["name"], "type": str(col["type"])}
                for col in table_info["columns"]
            ]
        schema_graph.build_from_schema(schema_dict)
        logger.info(f"[OK] DJPI schema graph initialized (ADVISORY mode)")

        # === Semantic Mediation (observational) ===
        if SEMANTIC_MEDIATION_AVAILABLE:
            semantic_mediator = SemanticMediator(strict_mode=False)
            logger.info(f"[OK] Semantic Mediator initialized (pre-NL-SQL concept detection)")
        else:
            semantic_mediator = None
            logger.info("[OK] Semantic Mediation skipped (module not available)")

        # === Intent Extraction ===
        if SEMANTIC_INTENT_AVAILABLE:
            intent_extractor = SemanticIntentExtractor(llm=llm)
            logger.info(f"[OK] Semantic Intent Extractor initialized (v6.1 one-shot)")
        else:
            intent_extractor = None
            logger.info("[OK] Semantic Intent Extraction skipped (module not available)")

        # === Intent Accumulation ===
        if INTENT_ACCUMULATION_AVAILABLE:
            intent_merger = create_intent_merger()
            # v6.3: Set schema reference for schema-backed clarification prompts
            set_schema_reference(db_manager.schema)
            logger.info(f"[OK] Intent Merger initialized (Phase 3.5 multi-turn accumulation)")
            logger.info(f"[OK] Schema reference set for clarification suggestions")
        else:
            intent_merger = None
            logger.info("[OK] Intent Accumulation skipped (module not available)")

        # === Context Resolution (multi-turn) ===
        if CONTEXT_RESOLUTION_AVAILABLE:
            context_resolver = create_context_resolver()
            result_binder = create_result_binder()
            logger.info(f"[OK] Context Resolver initialized (multi-turn reference resolution)")
        else:
            context_resolver = None
            result_binder = None
            logger.info("[OK] Context Resolution skipped (module not available)")

        # === SQL Validation & Guardrails ===
        if SQL_VALIDATION_AVAILABLE:
            sql_sanitizer = create_sql_sanitizer()  # v5.0.2: Strict SQL output contract
            alias_validator = create_alias_validator()
            complexity_analyzer = create_complexity_analyzer()
            logger.info(f"[OK] SQL Guardrails initialized (sanitizer + alias validation + complexity checks)")
        else:
            sql_sanitizer = None
            alias_validator = None
            complexity_analyzer = None
            logger.info("[OK] SQL Guardrails skipped (module not available)")

        # === Relational Correctness Layer ===
        if RELATIONAL_CORRECTION_AVAILABLE:
            relational_corrector = create_relational_corrector(db_manager.schema)
            if relational_corrector:
                fk_count = len(relational_corrector.schema.foreign_keys)
                logger.info(f"[OK] Relational Corrector initialized ({fk_count} FK relationships)")
            else:
                logger.info("[OK] Relational Corrector skipped (no schema)")
        else:
            relational_corrector = None
            logger.info("[OK] Relational Corrector skipped (module not available)")

        # === System Prompt (behavioral rules only, no schema injection) ===
        final_prompt = SYSTEM_PROMPT_TEMPLATE

        # Load custom prompt if exists (for behavioral customization only)
        if os.path.exists(CUSTOM_PROMPT_FILE):
            try:
                with open(CUSTOM_PROMPT_FILE, 'r') as f:
                    import json
                    config = json.load(f)
                    custom_prompt_config["enabled"] = config.get("enabled", False)
                    custom_prompt_config["prompt"] = config.get("prompt")
                    custom_prompt_config["use_dynamic_schema"] = False  # v5.0: Always False

                if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
                    # Use custom prompt (schema injection disabled in v5.0)
                    final_prompt = custom_prompt_config["prompt"]
                    # Remove any legacy schema placeholders
                    final_prompt = final_prompt.replace("{SCHEMA_SECTION}", "")
                    logger.info("[OK] Using CUSTOM prompt (schema injection disabled in v5.0)")
            except Exception as e:
                logger.warning(f"Failed to load custom prompt: {e}")
                logger.info("[OK] Falling back to DEFAULT prompt")

        # Initialize ReActAgent (kept for non-SQL queries, clarifications)
        # NOTE: LlamaIndex 0.14+ uses direct constructor
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=True,
            system_prompt=final_prompt,
        )
        logger.info("[OK] ReActAgent initialized (for non-SQL queries)")

        # ====================================================================
        # QUERY PIPELINE (v6.1 - Pure Orchestration)
        # ====================================================================
        # Simplified pipeline with SOLE clarification authority in accumulator.
        # No feature flags. No decision logic in pipeline.
        # ====================================================================
        query_pipeline = QueryPipeline(
            llm=llm,
            intent_extractor=intent_extractor,
            intent_merger=intent_merger,
            execute_fn=execute_nl_sql_query_async,
            semantic_mediator=semantic_mediator,
            get_last_sql_result_fn=get_last_sql_result,
            set_last_sql_result_fn=set_last_sql_result,
            clear_last_sql_result_fn=clear_last_sql_result,
            classify_viz_intent_fn=classify_visualization_intent,
        )
        logger.info("[OK] QueryPipeline initialized (v6.1.1 pure orchestration + ambiguity loop)")

        logger.info("=" * 60)
        logger.info("OptimaX v6.1.1 (Simplified Architecture + RCL + SRR + Ambiguity Loop) Ready!")
        logger.info("Architecture: Extract -> Accumulate -> Decide -> Execute -> [Ambiguity -> Clarify -> Resolve]")
        logger.info("Clarification Authority: intent_accumulator.py ONLY (+ RCL ambiguity)")
        logger.info("Decision Rule: entity AND metric known -> proceed")
        logger.info(f"Database: {len(db_manager.schema['tables'])} tables detected")
        if relational_corrector:
            fk_count = len(relational_corrector.schema.foreign_keys)
            logger.info(f"Relational Correction (RCL): {fk_count} FK relationships loaded")
        if SEMANTIC_ROLE_RESOLVER_AVAILABLE:
            from semantic_role_resolver import DEFAULT_ROLES
            logger.info(f"Semantic Role Resolver (SRR): {len(DEFAULT_ROLES)} roles defined")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    yield  # Server is running

    # Cleanup on shutdown (if needed)
    logger.info("Shutting down OptimaX...")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="OptimaX SQL Chat API",
    description="v5.3 Phase 3.5: Intent Accumulation + Multi-turn Semantic Convergence for NL-SQL queries",
    version="5.3",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================================================
# GLOBAL INSTANCES (v5.3 - NL-SQL Architecture + Intent Accumulation)
# ============================================================================
agent = None           # ReActAgent (kept for non-SQL queries, clarifications)
db_manager = None      # DatabaseManager (schema info, validation)
llm = None             # Groq LLM instance
schema_graph = None    # DJPI SchemaGraph (ADVISORY ONLY in v5.0)
semantic_mediator = None  # SemanticMediator (v5.0 - pre-NL-SQL concept detection)
intent_extractor = None   # SemanticIntentExtractor (v5.1 - LLM-based intent understanding)
intent_merger = None      # IntentMerger (v5.3 - Phase 3.5 intent accumulation)
context_resolver = None   # ContextResolver (v5.0 - multi-turn reference resolution)
result_binder = None      # ResultContextBinder (v5.0 - context binding from results)

# SQL Validation & Safety Guardrails (v5.0 - Execution Guardrails)
alias_validator = None    # SQLAliasValidator (v5.0 - alias reference validation)
complexity_analyzer = None  # QueryComplexityAnalyzer (v5.0 - query safety checks)
sql_sanitizer = None      # SQLOutputSanitizer (v5.0.2 - strict SQL output contract)
relational_corrector = None  # RelationalCorrector (v6.1 - FK-based SQL correction)

# NL-SQL Engine Components (v5.0 - Primary SQL execution path)
sql_database = None    # LlamaIndex SQLDatabase (SQLAlchemy wrapper)
nl_sql_engine = None   # NLSQLTableQueryEngine (handles SQL generation)

# Query Pipeline (v5.3.2 - Orchestration Controller)
query_pipeline = None  # QueryPipeline (handles all query orchestration)

sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    include_sql: Optional[bool] = True
    include_tasks: Optional[bool] = False
    row_limit: Optional[int] = 50


class ChartSuggestion(BaseModel):
    analysis_type: str  # 'comparison', 'time_series', 'proportion', 'correlation', 'distribution'
    reasoning: str
    recommended_charts: List[Dict[str, Any]]  # [{"type": "bar", "label": "Bar Chart", "description": "...", "recommended": true}]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sql_query: Optional[str] = None
    error: Optional[str] = None
    query_results: Optional[List[Dict[str, Any]]] = None  # Frontend expects this
    data: Optional[List[Dict[str, Any]]] = None
    execution_time: float
    tasks: Optional[List[Dict[str, Any]]] = None
    clarification_needed: Optional[bool] = False
    agent_reasoning: Optional[str] = None
    chart_suggestion: Optional[Dict[str, Any]] = None  # New: LLM suggests chart types


# ============================================================================
# SYSTEM PROMPT (v5.0 - NO SCHEMA INJECTION)
# ============================================================================
# ARCHITECTURAL CHANGE: Schema is NO LONGER injected into the system prompt.
# SQL generation is now handled by LlamaIndex's NLSQLTableQueryEngine which
# reads schema directly via SQLAlchemy introspection.
#
# This prompt now focuses ONLY on:
# - Behavioral rules (when to respond, when to stop)
# - Safety constraints (read-only, no dangerous operations)
# - Response formatting guidelines
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are OptimaX, an AI assistant for database analysis.

IMPORTANT BEHAVIORAL RULES:
1. NEVER run queries to "demonstrate capabilities" or "show examples"
2. If user asks "what can you do", describe capabilities - DO NOT run queries
3. Once you have useful data, formulate your answer and STOP
4. Don't run multiple queries unless absolutely necessary

SAFETY CONSTRAINTS (ENFORCED BY SYSTEM):
- You are READ-ONLY: No INSERT, UPDATE, DELETE, ALTER, DROP, CREATE, TRUNCATE
- All queries automatically include LIMIT for safety
- Only SELECT statements are allowed

SEMANTIC INTENT CLARIFICATION:
When users say:
- "add [column]" -> Include in SELECT projection (NOT ALTER TABLE)
- "include [column]" -> Include in SELECT projection
- "show [column]" -> Include in SELECT projection

RESPONSE GUIDELINES:
- Be concise and accurate
- Present data clearly
- Explain insights from the results
"""

# Legacy prompt variable (kept for backwards compatibility with custom prompts)
SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE

# Custom prompt storage
CUSTOM_PROMPT_FILE = "custom_system_prompt.txt"
custom_prompt_config = {
    "enabled": False,
    "prompt": None,
    "use_dynamic_schema": True
}




# === SQL Execution (Hard-Gated) ===

def execute_nl_sql_query(
    user_query: str,
    row_limit: int = 50,
    intent_state: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute NL-SQL query with hard-gated error handling.

    Returns structured result: success/sql/data/columns/error.
    Intent_state provides resolved FK preferences from prior clarifications.
    """
    global nl_sql_engine, db_manager, schema_graph

    # Semantic role resolution (optional FK hints for NL-SQL)
    query_with_hint = user_query
    semantic_hint_applied = False
    semantic_hint_info = None

    if SEMANTIC_ROLE_RESOLVER_AVAILABLE:
        try:
            # Get schema tables for filtering hints to relevant tables
            schema_tables = set(db_manager.get_table_names_without_schema()) if db_manager else None

            # Resolve semantic roles and get hint
            role_result = resolve_semantic_roles(user_query)
            semantic_hint_info = format_fk_hint_as_dict(role_result)

            if role_result.hint_applied:
                # Append hint to query for NL-SQL
                hint_text = get_fk_hint_for_query(user_query, schema_tables)
                if hint_text:
                    query_with_hint = user_query + hint_text
                    semantic_hint_applied = True
                    logger.info(
                        f"[STRICT] Semantic role hint applied: "
                        f"roles={role_result.detected_roles}, "
                        f"fk_prefs={role_result.preferred_fks}"
                    )
            elif role_result.is_ambiguous:
                logger.info(
                    f"[STRICT] Semantic roles ambiguous (no hint): {role_result.reason}"
                )
            # else: no roles detected, proceed without hint

        except Exception as e:
            # Non-fatal: semantic hint failure should not block query
            logger.warning(f"[STRICT] Semantic role resolution failed (non-blocking): {e}")

    # Generate SQL via NL-SQL Engine
    try:
        print(f"\n[QUERY] Processing: {user_query[:80]}...", flush=True)
        logger.info(f"[STRICT] Executing NL-SQL query: {user_query[:80]}...")

        # Call NL-SQL engine with potentially hinted query
        nl_response = nl_sql_engine.query(query_with_hint)

        # Extract SQL from response metadata
        # CRITICAL: We DISCARD the LLM's natural language response (str(nl_response))
        # This prevents schema guessing, error explanations, or SQL repair prose
        sql_query = None
        if hasattr(nl_response, 'metadata') and nl_response.metadata:
            sql_query = nl_response.metadata.get('sql_query')

        if not sql_query:
            # NL-SQL engine failed to generate SQL - this is a hard failure
            logger.warning(f"[STRICT] NL-SQL engine did not generate SQL")
            # Get raw table names for user hint (without schema prefix for clarity)
            available = db_manager.get_table_names_without_schema() if db_manager else None
            return {
                "success": False,
                "sql": None,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": "Could not generate SQL for this query. Please rephrase using specific table or column names.",
                "error_type": "sql_generation_failed",
                "available_tables": available,
            }

        print(f"[SQL] Generated: {sql_query[:100]}...", flush=True)
        logger.info(f"[STRICT] Generated SQL (raw): {sql_query[:100]}...")

        # SQL sanitization (extract single statement, strip commentary)
        if SQL_VALIDATION_AVAILABLE and sql_sanitizer:
            sanitization_result = sql_sanitizer.sanitize(sql_query)

            if not sanitization_result.valid:
                # Sanitization failed - return structured error
                logger.warning(f"[STRICT] SQL sanitization FAILED: {sanitization_result.error_message}")
                return {
                    "success": False,
                    "sql": sql_query,  # Include raw SQL for debugging
                    "data": None,
                    "columns": None,
                    "row_count": 0,
                    "error": sanitization_result.error_message,
                    "error_type": "sql_sanitization_failed",
                    "available_tables": None,
                }

            # Use sanitized SQL from here on
            sql_query = sanitization_result.sql

            if sanitization_result.had_commentary:
                logger.info(f"[STRICT] Stripped commentary from SQL output")

            logger.info(f"[STRICT] Sanitized SQL: {sql_query[:100]}...")

        # Schema normalization (add schema prefix if needed)
        if db_manager and db_manager.active_schema:
            sql_query = db_manager._normalize_sql_schema(sql_query)
            logger.info(f"[STRICT] Normalized SQL: {sql_query[:100]}...")

    except Exception as e:
        # NL-SQL engine threw an exception - return structured error
        logger.error(f"[STRICT] NL-SQL engine exception: {str(e)}")
        available = db_manager.get_table_names_without_schema() if db_manager else None
        return {
            "success": False,
            "sql": None,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": f"SQL generation failed: {str(e)}",
            "error_type": "sql_generation_exception",
            "available_tables": available,
        }

    # Relational Correctness Layer (FK-based SQL correction)
    if RELATIONAL_CORRECTION_AVAILABLE and relational_corrector:
        # Check if we have resolved FK preferences from prior clarification
        forced_fk = None
        if intent_state and hasattr(intent_state, 'get_resolved_fk_preferences'):
            forced_fk = intent_state.get_resolved_fk_preferences()

        # Apply correction (with forced FK if available)
        if forced_fk:
            logger.info(f"[STRICT] Applying resolved FK preferences: {forced_fk}")
            correction_result = relational_corrector.correct_with_forced_fk(sql_query, forced_fk)
        else:
            correction_result = relational_corrector.correct(sql_query)

        if not correction_result.success:
            # Check if this is a STRUCTURED AMBIGUITY (needs clarification)
            if correction_result.needs_clarification() and correction_result.ambiguity:
                ambiguity = correction_result.ambiguity
                logger.info(
                    f"[RCL_AMBIGUITY] Structured ambiguity detected - needs clarification: "
                    f"column='{ambiguity.column}', options={[o.fk_column for o in ambiguity.options]}"
                )

                # Return special result indicating clarification needed
                return {
                    "success": False,
                    "sql": sql_query,
                    "data": None,
                    "columns": None,
                    "row_count": 0,
                    "error": None,  # NOT an error - it's an ambiguity
                    "error_type": "relational_ambiguity",
                    "ambiguity": ambiguity.to_dict(),  # Structured ambiguity for caller
                    "available_tables": None,
                }

            # Hard error (not ambiguity)
            error_info = correction_result.error or {}
            error_msg = error_info.get("message", "Relational correction failed")
            logger.warning(f"[STRICT] Relational correction FAILED: {error_msg}")
            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": error_msg,
                "error_type": "relational_correction_failed",
                "available_tables": None,
            }

        # Use corrected SQL if fixes were applied
        if correction_result.applied_fixes:
            logger.info(
                f"[STRICT] Relational correction applied {len(correction_result.applied_fixes)} fix(es)"
            )
            for fix in correction_result.applied_fixes:
                logger.info(f"  - Added JOIN: {fix['join_condition']}")
            sql_query = correction_result.corrected_sql

    # Column existence validation
    if SQL_VALIDATION_AVAILABLE and db_manager and db_manager.schema:
        column_result = validate_column_existence(sql_query, db_manager.schema)

        if not column_result.valid:
            logger.warning(f"[STRICT] Column validation FAILED: {column_result.error_message}")
            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": column_result.error_message,
                "error_type": "column_validation_failed",
                "available_tables": None,
            }

        logger.debug(f"[STRICT] Column validation passed")

    # Alias validation (reject undefined table aliases)
    if SQL_VALIDATION_AVAILABLE and alias_validator:
        alias_result = alias_validator.validate(sql_query)

        if not alias_result.valid:
            logger.warning(f"[STRICT] Alias validation FAILED: {alias_result.error_message}")
            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": alias_result.error_message,
                "error_type": "alias_validation_failed",
                "available_tables": None,
            }

        logger.debug(f"[STRICT] Alias validation passed: {alias_result.declared_aliases}")

    # Complexity check (block overly complex queries)
    # v6.9: BUG 2 FIX - Analyzer now accepts row_limit to know LIMIT will be enforced
    if SQL_VALIDATION_AVAILABLE and complexity_analyzer:
        complexity_result = complexity_analyzer.analyze(sql_query, row_limit=row_limit)

        if not complexity_result.is_safe:
            # Query too complex - block execution
            logger.warning(
                f"[STRICT] Query complexity BLOCKED: "
                f"score={complexity_result.complexity_score}, "
                f"joins={complexity_result.join_count}, "
                f"has_where={complexity_result.has_where_filter}"
            )
            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": complexity_result.warning_message,
                "error_type": "complexity_blocked",
                "available_tables": None,
            }

        if complexity_result.warning_message:
            # Warning but not blocking - log for observability
            logger.info(f"[STRICT] Query complexity warning: {complexity_result.warning_message}")

    # Safety validation (SELECT-only, LIMIT enforcement)
    try:
        is_valid, validated_sql = db_manager._validate_sql(sql_query, row_limit)

        if not is_valid:
            logger.warning(f"[STRICT] SQL validation failed: {validated_sql}")
            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": f"Query blocked by safety validation: {validated_sql}",
                "error_type": "validation_failed",
                "available_tables": None,
            }

        sql_query = validated_sql  # Use validated (possibly modified) SQL

    except Exception as e:
        logger.error(f"[STRICT] Validation exception: {str(e)}")
        return {
            "success": False,
            "sql": sql_query,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": f"SQL validation error: {str(e)}",
            "error_type": "validation_exception",
            "available_tables": None,
        }

    # Execute against database
    try:
        print(f"[EXEC] Executing query against database...", flush=True)
        result = db_manager.execute_query(sql_query, row_limit)

        if result.get("success"):
            data = result.get("data", [])
            columns = result.get("columns", [])

            logger.info(f"[STRICT] Query successful: {len(data)} rows returned")

            return {
                "success": True,
                "sql": sql_query,
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "error": None,
                "error_type": None,
                "available_tables": None,
            }
        else:
            # Database returned an error - pass it through RAW
            db_error = result.get("error", "Unknown database error")
            logger.warning(f"[STRICT] Database error: {db_error}")
            available = db_manager.get_table_names_without_schema() if db_manager else None

            return {
                "success": False,
                "sql": sql_query,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": db_error,
                "error_type": "database_error",
                "available_tables": available,
            }

    except Exception as e:
        logger.error(f"[STRICT] Database exception: {str(e)}")
        available = db_manager.get_table_names_without_schema() if db_manager else None
        return {
            "success": False,
            "sql": sql_query,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": f"Database execution failed: {str(e)}",
            "error_type": "database_exception",
            "available_tables": available,
        }


async def execute_nl_sql_query_async(
    user_query: str,
    row_limit: int = 50,
    timeout: float = 45.0,
    intent_state: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Async wrapper for execute_nl_sql_query with timeout.

    v6.1.1: Added intent_state parameter for resolved FK preferences.

    Returns structured result or timeout error.
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(execute_nl_sql_query, user_query, row_limit, intent_state),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[STRICT] Query timeout after {timeout}s")
        return {
            "success": False,
            "sql": None,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": (
                f"Query timed out after {timeout} seconds. "
                f"Please narrow your query by specifying:\n"
                f"- A specific route (e.g., 'JFK to LAX')\n"
                f"- A date range (e.g., 'in January 2024')\n"
                f"- A specific entity (e.g., 'passenger 12345')"
            ),
            "error_type": "timeout",
            "available_tables": None,
        }


def get_or_create_session(session_id: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Get existing session or create new one with optional custom prompt"""
    global agent, llm

    if session_id not in sessions:
        session_agent = agent

        if custom_prompt:
            logger.info(f"Creating session with custom system prompt: {session_id}")
            tools, _ = initialize_tools(DATABASE_URL)
            session_agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True,
                context=custom_prompt,
            )

        # Create session context for multi-turn reference resolution
        session_context = None
        if CONTEXT_RESOLUTION_AVAILABLE:
            session_context = create_session_context(session_id)

        # Create intent state for multi-turn intent accumulation (Phase 3.5)
        accumulated_intent = None
        if INTENT_ACCUMULATION_AVAILABLE:
            accumulated_intent = create_intent_state()

        sessions[session_id] = {
            "agent": session_agent,
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "message_count": 0,
            "custom_prompt": custom_prompt is not None,
            # Analytical context tracking for Query Governance
            "analytical_context": {
                "objective_type": None,  # e.g., ranking, aggregation, classification
                "entity": None,          # e.g., customer, route, flight
                "status": "pending",     # pending, completed
                "last_sql_result": None, # Cache last valid SQL result for viz reuse
            },
            # v5.0: Session context for multi-turn reference resolution
            # Stores entity identifiers from single-row results
            "session_context": session_context,
            # v5.3 Phase 3.5: Accumulated intent state for multi-turn convergence
            # Allows follow-up fragments to contribute to a complete intent
            "accumulated_intent": accumulated_intent,
        }

        prompt_type = "custom" if custom_prompt else "default"
        logger.info(f"Created new session ({prompt_type} prompt): {session_id}")

    sessions[session_id]["last_active"] = datetime.now()
    return sessions[session_id]


@app.get("/")
async def root():
    return {
        "message": "OptimaX SQL Chat API v5.3",
        "version": "5.3",
        "architecture": "Intent Extraction -> Intent Accumulation -> Context Resolution -> Semantic Mediation -> NL-SQL -> Guardrails -> Database",
        "features": [
            "Intent Accumulation (v5.3 Phase 3.5 - multi-turn semantic convergence)",
            "Semantic Intent Extraction (v5.1 - LLM-based conceptual understanding)",
            "Session Context Resolution (v5.0 - multi-turn + route context)",
            "Route Context Binding (v5.0 - 'JFK to ATL' -> 'this route')",
            "Semantic Mediation Layer (v5.0 - pre-NL-SQL concept detection)",
            "SQL Alias Validation (v5.0 - reject undefined aliases)",
            "Query Complexity Guardrails (v5.0 - block unsafe queries)",
            "SQL generation via LlamaIndex NLSQLTableQueryEngine",
            "DJPI v3 - Advisory mode (explainability only)",
            "Query Governance Layer (preserved)",
            "One-shot chart classification",
            "Session-based memory",
            "SQL safety validation (read-only, LIMIT enforcement)",
        ],
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.

    Validates:
    - NL-SQL engine initialization
    - Database connection (live test)
    - Foreign key relationships loaded
    - Python environment
    """
    import sys

    health_status = {
        "status": "healthy",
        "version": "6.1",
        "checks": {},
    }

    # Check 1: NL-SQL Engine
    if nl_sql_engine is None:
        health_status["status"] = "unhealthy"
        health_status["checks"]["nl_sql_engine"] = {"status": "fail", "error": "Not initialized"}
    else:
        health_status["checks"]["nl_sql_engine"] = {"status": "ok", "type": "NLSQLTableQueryEngine"}

    # Check 2: Database Connection (live test)
    db_check = {"status": "unknown"}
    if db_manager:
        try:
            # Perform actual query to verify connection
            result = db_manager.execute_query("SELECT 1 as health_check", row_limit=1)
            if result.get("success"):
                db_check["status"] = "ok"
                db_check["tables"] = len(db_manager.get_all_table_names())
                db_check["schema"] = db_manager.active_schema
            else:
                db_check["status"] = "fail"
                db_check["error"] = result.get("error", "Query failed")
                health_status["status"] = "unhealthy"
        except Exception as e:
            db_check["status"] = "fail"
            db_check["error"] = str(e)
            health_status["status"] = "unhealthy"
    else:
        db_check["status"] = "fail"
        db_check["error"] = "DatabaseManager not initialized"
        health_status["status"] = "unhealthy"
    health_status["checks"]["database"] = db_check

    # Check 3: Foreign Keys (for relational correction)
    fk_check = {"status": "unknown"}
    if RELATIONAL_CORRECTION_AVAILABLE and relational_corrector:
        fk_count = len(relational_corrector.schema.foreign_keys)
        fk_check["status"] = "ok"
        fk_check["foreign_keys_loaded"] = fk_count
    else:
        fk_check["status"] = "warn"
        fk_check["message"] = "Relational correction not available"
    health_status["checks"]["foreign_keys"] = fk_check

    # Check 4: Python Environment
    env_check = {
        "status": "ok",
        "interpreter": sys.executable,
        "version": sys.version.split()[0],
        "in_venv": hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix,
    }
    health_status["checks"]["python_environment"] = env_check

    # Check 5: LlamaIndex imports
    llama_check = {"status": "unknown"}
    try:
        import llama_index
        llama_check["status"] = "ok"
        llama_check["version"] = getattr(llama_index, "__version__", "unknown")
    except ImportError as e:
        llama_check["status"] = "fail"
        llama_check["error"] = str(e)
        health_status["status"] = "unhealthy"
    health_status["checks"]["llama_index"] = llama_check

    # Summary
    health_status["model"] = GROQ_MODEL
    health_status["active_sessions"] = len(sessions)
    health_status["layers"] = {
        "semantic_intent": "enabled" if SEMANTIC_INTENT_AVAILABLE else "disabled",
        "intent_accumulation": "enabled" if INTENT_ACCUMULATION_AVAILABLE else "disabled",
        "context_resolution": "enabled" if CONTEXT_RESOLUTION_AVAILABLE else "disabled",
        "semantic_mediation": "enabled" if SEMANTIC_MEDIATION_AVAILABLE else "disabled",
        "sql_guardrails": "enabled" if SQL_VALIDATION_AVAILABLE else "disabled",
        "relational_correction": "enabled" if RELATIONAL_CORRECTION_AVAILABLE and relational_corrector else "disabled",
        "semantic_role_resolver": "enabled" if SEMANTIC_ROLE_RESOLVER_AVAILABLE else "disabled",
    }

    return health_status


@app.get("/verify/llamaindex")
async def verify_llamaindex():
    """
    Detailed LlamaIndex verification endpoint.

    Returns comprehensive verification results including:
    - Import status for all required modules
    - Version consistency check
    - Module path verification (inside venv)
    - Runtime class availability

    For full verification with smoke test, run:
        python verify_llamaindex.py --smoke-test --verbose
    """
    import importlib
    import importlib.metadata

    result = {
        "status": "unknown",
        "imports": {},
        "versions": {},
        "paths": {},
        "warnings": [],
        "errors": [],
    }

    # Required imports
    required_imports = [
        ("llama_index", "Meta-package"),
        ("llama_index.core", "Core library"),
        ("llama_index.core.query_engine", "Query engine module"),
        ("llama_index.llms.groq", "Groq LLM integration"),
    ]

    # Required classes
    required_classes = [
        ("llama_index.core.query_engine", "NLSQLTableQueryEngine"),
        ("llama_index.core", "SQLDatabase"),
        ("llama_index.core", "Settings"),
        ("llama_index.llms.groq", "Groq"),
    ]

    # Get venv path
    in_venv = hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    venv_path = ""
    if in_venv:
        from pathlib import Path
        interpreter_path = Path(sys.executable)
        venv_path = str(interpreter_path.parent.parent)

    # Check imports
    for import_path, description in required_imports:
        try:
            module = importlib.import_module(import_path)
            module_file = getattr(module, "__file__", "namespace")

            # Check if from venv
            in_correct_venv = True
            if module_file and module_file != "namespace" and venv_path:
                in_correct_venv = venv_path in module_file

            result["imports"][import_path] = {
                "status": "ok",
                "description": description,
                "path": module_file,
                "in_venv": in_correct_venv,
            }

            if not in_correct_venv:
                result["warnings"].append(f"{import_path} loaded from outside venv")

        except ImportError as e:
            result["imports"][import_path] = {
                "status": "fail",
                "description": description,
                "error": str(e),
            }
            result["errors"].append(f"Cannot import {import_path}: {e}")

    # Check classes
    for module_path, class_name in required_classes:
        key = f"{module_path}.{class_name}"
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls and callable(cls):
                result["imports"][key] = {"status": "ok", "type": "class"}
            else:
                result["imports"][key] = {"status": "fail", "error": "Not found or not callable"}
                result["errors"].append(f"{class_name} not found in {module_path}")
        except ImportError as e:
            result["imports"][key] = {"status": "fail", "error": str(e)}
            result["errors"].append(f"Cannot import {module_path}: {e}")

    # Check versions
    packages = ["llama-index", "llama-index-core", "llama-index-llms-groq"]
    expected_prefixes = {"llama-index": "0.12", "llama-index-core": "0.12", "llama-index-llms-groq": "0.3"}

    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
            expected = expected_prefixes.get(pkg, "")
            is_expected = version.startswith(expected) if expected else True

            result["versions"][pkg] = {
                "version": version,
                "expected": f"{expected}.x" if expected else "any",
                "status": "ok" if is_expected else "warn",
            }

            if not is_expected:
                result["warnings"].append(f"{pkg} version {version} (expected {expected}.x)")

        except importlib.metadata.PackageNotFoundError:
            result["versions"][pkg] = {"status": "fail", "error": "Not installed"}
            result["errors"].append(f"{pkg} not installed")

    # Environment info
    result["environment"] = {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "in_venv": in_venv,
        "venv_path": venv_path if venv_path else None,
    }

    # Determine overall status
    if result["errors"]:
        result["status"] = "fail"
    elif result["warnings"]:
        result["status"] = "warn"
    else:
        result["status"] = "pass"

    return result


@app.get("/database/schema")
async def get_database_schema():
    """Get current database schema information"""
    try:
        if db_manager is None:
            raise HTTPException(status_code=503, detail="Database not initialized")

        schema_summary = db_manager.get_schema_summary()
        return {
            "success": True,
            "schema": schema_summary,
            "schema_text": db_manager.get_schema_for_llm()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantic-mediation/info")
async def get_semantic_mediation_info():
    """
    Get information about the Semantic Mediation Layer (v5.0).

    WHAT THIS ENDPOINT SHOWS:
    - How semantic mediation works
    - Supported concept patterns
    - Ambiguity detection rules

    WHY SEMANTIC MEDIATION (FOR REVIEWERS):
    - Operates BEFORE NL-SQL execution (not on schema)
    - Detects conceptual terms in user queries
    - Asks clarifying questions when ambiguous
    - Does NOT inject schema into prompts
    - Does NOT generate SQL
    - Preserves database agnosticism
    """
    try:
        # Import semantic mediation patterns (lazy import)
        try:
            from semantic_mediation import CONCEPT_PATTERNS, AMBIGUOUS_METRICS
            patterns = list(CONCEPT_PATTERNS.keys())
            ambiguous = list(AMBIGUOUS_METRICS.keys())
        except ImportError:
            patterns = ["route", "traffic", "loyalty", "frequency", "entity"]
            ambiguous = ["busiest", "most", "best", "top", "frequent"]

        return {
            "success": True,
            "layer": "Semantic Mediation",
            "position": "BEFORE NL-SQL (operates on user query)",
            "purpose": "Detect conceptual terms and resolve ambiguity",
            # What concepts can be detected
            "detectable_concepts": patterns,
            # What terms trigger clarification
            "ambiguous_triggers": ambiguous,
            # Architecture note
            "architecture": {
                "flow": "User Query -> Semantic Mediation -> NL-SQL -> Database",
                "does_not": [
                    "Inject schema into prompts",
                    "Generate SQL",
                    "Guess joins",
                    "Hardcode table names",
                ],
                "does": [
                    "Detect high-level concepts in natural language",
                    "Ask clarifying questions when ambiguous",
                    "Pass clear queries to NL-SQL unchanged",
                ],
            },
            "note": (
                "Semantic mediation operates on natural language BEFORE NL-SQL. "
                "It helps users use conceptual terms without knowing database structure."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/relational-correction/info")
async def get_relational_correction_info():
    """
    Get information about the Relational Correctness Layer (v6.1).

    WHAT THIS ENDPOINT SHOWS:
    - How relational correction works
    - Foreign key relationships loaded
    - Example correction scenarios

    WHY RELATIONAL CORRECTION (FOR REVIEWERS):
    - Operates AFTER NL-SQL generation, BEFORE validation
    - Uses ONLY declared FK metadata (no LLM, no guessing)
    - Fixes structurally invalid SQL when EXACTLY ONE FK path exists
    - REJECTS ambiguous cases (multiple FK paths)
    - Preserves database agnosticism
    """
    try:
        if not RELATIONAL_CORRECTION_AVAILABLE:
            return {
                "success": True,
                "enabled": False,
                "reason": "Module not available"
            }

        if not relational_corrector:
            return {
                "success": True,
                "enabled": False,
                "reason": "No schema loaded"
            }

        # Get FK statistics
        fks = relational_corrector.schema.foreign_keys
        tables_with_fks = set()
        for fk in fks:
            tables_with_fks.add(fk.source_table)

        # Sample FK relationships for display
        sample_fks = []
        for fk in fks[:10]:  # Show first 10
            sample_fks.append({
                "from": f"{fk.source_table}.{fk.source_column}",
                "to": f"{fk.target_table}.{fk.target_column}"
            })

        return {
            "success": True,
            "enabled": True,
            "layer": "Relational Correctness Layer",
            "version": "6.1",
            "position": "AFTER NL-SQL, BEFORE Column Validation",
            "purpose": "Correct FK-resolvable column errors deterministically",
            "statistics": {
                "total_foreign_keys": len(fks),
                "tables_with_fks": len(tables_with_fks),
            },
            "sample_relationships": sample_fks,
            "behavior": {
                "single_fk_path": "Rewrite SQL with JOIN (deterministic)",
                "multiple_fk_paths": "Return structured error (ambiguous)",
                "no_fk_path": "Pass through (let column validator handle)"
            },
            "guarantees": [
                "Uses ONLY declared FK metadata",
                "No LLM, no semantic guessing",
                "Rewrites are DETERMINISTIC",
                "Ambiguous cases ALWAYS fail",
            ],
            "example": {
                "invalid_sql": "SELECT airport_code, COUNT(*) FROM flight GROUP BY airport_code",
                "error": "Column 'airport_code' does not exist in table 'flight'",
                "fix_if_single_fk": (
                    "SELECT a1.airport_code, COUNT(*) FROM flight\n"
                    "JOIN airport a1 ON flight.departure_airport = a1.airport_code\n"
                    "GROUP BY a1.airport_code"
                ),
                "fail_if_ambiguous": (
                    "Ambiguous column reference \"airport_code\".\n"
                    "Multiple join paths found:\n"
                    "  - flight.departure_airport -> airport\n"
                    "  - flight.arrival_airport -> airport"
                )
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/semantic-role-resolver/info")
async def get_semantic_role_resolver_info():
    """
    Get information about the Semantic Role Resolver (v6.1).

    WHAT THIS ENDPOINT SHOWS:
    - How semantic role resolution works
    - Detected role patterns
    - FK preference mappings

    WHY SEMANTIC ROLE RESOLVER (FOR REVIEWERS):
    - Operates BEFORE NL-SQL generation (provides hints)
    - Detects semantic roles (arrival vs departure, etc.)
    - Nudges NL-SQL toward correct FK column
    - DOES NOT bypass RCL validation
    - Conservative: no hint if ambiguous
    """
    try:
        if not SEMANTIC_ROLE_RESOLVER_AVAILABLE:
            return {
                "success": True,
                "enabled": False,
                "reason": "Module not available"
            }

        # Import role definitions for display
        from semantic_role_resolver import DEFAULT_ROLES, CONFLICTING_ROLE_PAIRS

        roles_info = []
        for role in DEFAULT_ROLES:
            roles_info.append({
                "name": role.name,
                "patterns": role.patterns[:3],  # Show first 3 patterns
                "fk_preferences": role.fk_preferences
            })

        return {
            "success": True,
            "enabled": True,
            "layer": "Semantic Role Resolver",
            "version": "6.1",
            "position": "BEFORE NL-SQL (provides hints)",
            "purpose": "Detect semantic roles and provide FK hints to guide NL-SQL",
            "defined_roles": roles_info,
            "conflicting_pairs": CONFLICTING_ROLE_PAIRS,
            "behavior": {
                "single_role_detected": "Provide FK hint to NL-SQL",
                "conflicting_roles_detected": "No hint (conservative)",
                "no_role_detected": "No hint (passthrough)"
            },
            "guarantees": [
                "CONSERVATIVE: Only hints when unambiguous",
                "OPTIONAL: No hint = no change to behavior",
                "DOES NOT BYPASS RCL: RCL still validates",
                "NO SQL GENERATION: Only advisory hints",
            ],
            "examples": [
                {
                    "query": "which airport has the most arrivals?",
                    "detected_role": "arrival",
                    "hint": {"flight": "arrival_airport"}
                },
                {
                    "query": "top airports by departures",
                    "detected_role": "departure",
                    "hint": {"flight": "departure_airport"}
                },
                {
                    "query": "airport with most flights",
                    "detected_role": None,
                    "hint": "No hint (ambiguous)"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/semantic-role-resolver/test")
async def test_semantic_role_resolver(request: dict):
    """
    Test the semantic role resolver with a query.

    Request body:
        {"query": "which airport has the most arrivals?"}

    Returns the detected roles and FK preferences.
    """
    try:
        if not SEMANTIC_ROLE_RESOLVER_AVAILABLE:
            return {
                "success": False,
                "error": "Semantic role resolver not available"
            }

        query = request.get("query", "")
        if not query:
            return {
                "success": False,
                "error": "No query provided"
            }

        result = resolve_semantic_roles(query)
        hint_dict = format_fk_hint_as_dict(result)
        hint_text = get_fk_hint_for_query(query)

        return {
            "success": True,
            "query": query,
            "result": hint_dict,
            "hint_text": hint_text if hint_text else "(no hint)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DatabaseConnectionRequest(BaseModel):
    database_url: str


class SystemPromptRequest(BaseModel):
    prompt: str
    use_dynamic_schema: bool = True  # Whether to inject schema into prompt


@app.post("/database/test-connection")
async def test_database_connection(request: DatabaseConnectionRequest):
    """Test a database connection without changing the current one"""
    try:
        from sqlalchemy import create_engine, text

        # Try to connect and query
        test_engine = create_engine(request.database_url)
        with test_engine.connect() as conn:
            # Test query
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # If successful, get table count
        from sqlalchemy import inspect
        inspector = inspect(test_engine)

        # Try to detect schema
        schemas = inspector.get_schema_names()
        table_count = 0

        for schema in schemas:
            if schema not in ['information_schema', 'pg_catalog']:
                tables = inspector.get_table_names(schema=schema)
                table_count += len(tables)

        test_engine.dispose()

        return {
            "success": True,
            "message": "Connection successful",
            "table_count": table_count,
            "schemas": [s for s in schemas if s not in ['information_schema', 'pg_catalog']]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/database/connect")
async def connect_to_database(request: DatabaseConnectionRequest):
    """
    Connect to a new database and reload NL-SQL engine.

    v5.0 CHANGES:
    - Reinitializes NL-SQL engine (primary SQL execution path)
    - Reinitializes DJPI schema graph (advisory mode)
    - No schema injection into prompts
    """
    global agent, db_manager, llm, sessions, custom_prompt_config, DATABASE_URL
    global sql_database, nl_sql_engine, schema_graph  # v5.0: NL-SQL components
    global relational_corrector  # v6.1: FK-based SQL correction

    try:
        logger.info(f"Attempting to connect to new database...")

        # First, test the connection
        from sqlalchemy import text, inspect

        test_engine = create_engine(request.database_url)
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # Get schema info
        inspector = inspect(test_engine)
        schemas = inspector.get_schema_names()
        table_count = 0
        schema_names = []

        for schema in schemas:
            if schema not in ['information_schema', 'pg_catalog']:
                tables = inspector.get_table_names(schema=schema)
                table_count += len(tables)
                schema_names.append(schema)

        test_engine.dispose()

        # Connection successful - now switch to it
        DATABASE_URL = request.database_url

        # Reinitialize tools and database manager
        tools, new_db_manager = initialize_tools(DATABASE_URL)
        db_manager = new_db_manager
        logger.info(f"[OK] Database manager reinitialized: {table_count} tables")

        # ====================================================================
        # NL-SQL ENGINE REINITIALIZATION (v5.0 - Dynamic Table Scoping)
        # ====================================================================
        # Same logic as startup - see lifespan() for detailed comments
        # CRITICAL: Schema and table names must be separated for SQLDatabase
        # ====================================================================
        engine = create_engine(DATABASE_URL)

        # Get separated schema and table names
        nl_sql_config = db_manager.get_tables_for_nl_sql()
        raw_table_names = nl_sql_config["tables"]
        detected_schema = nl_sql_config["schema"]

        logger.info(f"[OK] Discovered {len(raw_table_names)} tables" +
                   (f" in schema '{detected_schema}'" if detected_schema else ""))

        # ====================================================================
        # SQLDatabase REINITIALIZATION (v5.0)
        # ====================================================================
        # Same as startup - only supported parameters used.
        # Semantic understanding handled by SEMANTIC MEDIATION LAYER.
        # ====================================================================
        sql_database = SQLDatabase(
            engine,
            schema=detected_schema,         # Schema passed separately
            include_tables=raw_table_names  # Raw table names only
        )
        nl_sql_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=raw_table_names,
            llm=llm,
            synthesize_response=False,
            verbose=True,
        )
        logger.info(f"[OK] NL-SQL Engine reinitialized with {len(raw_table_names)} table constraints")

        # ====================================================================
        # DJPI SCHEMA GRAPH REINITIALIZATION (v5.0 - Advisory mode)
        # ====================================================================
        schema_graph = SchemaGraph()
        schema_dict = {}
        for table_name, table_info in db_manager.schema["tables"].items():
            schema_dict[table_name] = [
                {"name": col["name"], "type": str(col["type"])}
                for col in table_info["columns"]
            ]
        schema_graph.build_from_schema(schema_dict)
        logger.info(f"[OK] DJPI schema graph reinitialized (advisory mode)")

        # ====================================================================
        # RELATIONAL CORRECTOR REINITIALIZATION (v6.1)
        # ====================================================================
        if RELATIONAL_CORRECTION_AVAILABLE:
            relational_corrector = create_relational_corrector(db_manager.schema)
            if relational_corrector:
                fk_count = len(relational_corrector.schema.foreign_keys)
                logger.info(f"[OK] Relational Corrector reinitialized ({fk_count} FK relationships)")
            else:
                logger.info("[OK] Relational Corrector skipped (no schema)")
        else:
            relational_corrector = None

        # v5.0: No schema injection into prompts
        final_prompt = SYSTEM_PROMPT_TEMPLATE
        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            final_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", "")

        # Reinitialize agent (for non-SQL queries)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            context=final_prompt,
        )

        # Clear all existing sessions
        cleared_sessions = len(sessions)
        sessions.clear()

        logger.info("=" * 60)
        logger.info(f"[OK] Connected to new database (v5.0 NL-SQL Architecture)!")
        logger.info(f"Tables: {len(raw_table_names)} (constrained)")
        logger.info(f"Active schema: {detected_schema or 'None'}")
        logger.info(f"SQL Engine: NLSQLTableQueryEngine (table-scoped)")
        logger.info(f"Sessions cleared: {cleared_sessions}")
        logger.info("=" * 60)

        return {
            "success": True,
            "message": "Successfully connected to database",
            "table_count": len(raw_table_names),
            "active_schema": detected_schema,
            "schemas": schema_names,
            "sql_engine": "NLSQLTableQueryEngine",
            "table_scoping": "dynamic",
            "discovered_tables": raw_table_names[:10],  # Raw names for display
            "sessions_cleared": cleared_sessions,
            # v5.0: Semantic mediation operates on user queries, not schema
            "semantic_mediation": "enabled",
        }

    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/system-prompt/active")
async def get_active_system_prompt():
    """
    Get the currently active system prompt.

    v5.0 NOTE: Schema injection is DISABLED.
    The NL-SQL engine reads schema directly via SQLAlchemy.
    Prompts now focus only on behavioral rules.
    """
    try:
        global custom_prompt_config

        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            # Remove any legacy schema placeholders
            active_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", "")
            return {
                "type": "custom",
                "prompt": active_prompt,
                "raw_prompt": custom_prompt_config["prompt"],
                "schema_injection": "disabled (v5.0)",
                "note": "Schema is read by NL-SQL engine via SQLAlchemy, not injected into prompts"
            }
        else:
            return {
                "type": "default",
                "prompt": SYSTEM_PROMPT_TEMPLATE,
                "raw_prompt": SYSTEM_PROMPT_TEMPLATE,
                "schema_injection": "disabled (v5.0)",
                "note": "Schema is read by NL-SQL engine via SQLAlchemy, not injected into prompts"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/save")
async def save_system_prompt(request: SystemPromptRequest):
    """Save a custom system prompt permanently"""
    try:
        global custom_prompt_config
        import json

        config = {
            "enabled": True,
            "prompt": request.prompt,
            "use_dynamic_schema": request.use_dynamic_schema,
            "saved_at": datetime.now().isoformat()
        }

        # Save to file
        with open(CUSTOM_PROMPT_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Update in-memory config
        custom_prompt_config["enabled"] = True
        custom_prompt_config["prompt"] = request.prompt
        custom_prompt_config["use_dynamic_schema"] = request.use_dynamic_schema

        logger.info("[OK] Custom system prompt saved")

        return {
            "success": True,
            "message": "Custom prompt saved. Restart backend to apply changes.",
            "config": config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/reset")
async def reset_system_prompt():
    """Reset to default dynamic system prompt"""
    try:
        global custom_prompt_config

        # Remove custom prompt file
        if os.path.exists(CUSTOM_PROMPT_FILE):
            os.remove(CUSTOM_PROMPT_FILE)

        # Update in-memory config
        custom_prompt_config["enabled"] = False
        custom_prompt_config["prompt"] = None
        custom_prompt_config["use_dynamic_schema"] = True

        logger.info("[OK] Reset to default dynamic prompt")

        return {
            "success": True,
            "message": "Reset to default prompt. Restart backend to apply changes."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system-prompt/apply")
async def apply_system_prompt():
    """
    Apply the current saved system prompt without restart (hot reload).

    v5.0 NOTE: Schema injection is DISABLED.
    This only affects behavioral rules in the prompt.
    """
    try:
        global agent, custom_prompt_config, sessions, llm, db_manager

        # Load custom prompt if exists
        if os.path.exists(CUSTOM_PROMPT_FILE):
            try:
                with open(CUSTOM_PROMPT_FILE, 'r') as f:
                    import json
                    config = json.load(f)
                    custom_prompt_config["enabled"] = config.get("enabled", False)
                    custom_prompt_config["prompt"] = config.get("prompt")
                    custom_prompt_config["use_dynamic_schema"] = False  # v5.0: Always disabled
            except Exception as e:
                logger.error(f"Failed to load custom prompt during apply: {e}")
                custom_prompt_config["enabled"] = False

        # v5.0: No schema injection - prompt is for behavioral rules only
        if custom_prompt_config["enabled"] and custom_prompt_config["prompt"]:
            # Remove any legacy schema placeholders
            final_prompt = custom_prompt_config["prompt"].replace("{SCHEMA_SECTION}", "")
            prompt_type = "CUSTOM (behavioral rules only)"
        else:
            final_prompt = SYSTEM_PROMPT_TEMPLATE
            prompt_type = "DEFAULT (behavioral rules only)"

        # Reinitialize agent (for non-SQL queries)
        tools, _ = initialize_tools(DATABASE_URL)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            context=final_prompt,
        )

        # Clear all sessions
        cleared_sessions = len(sessions)
        sessions.clear()

        logger.info(f"[OK] Applied system prompt: {prompt_type}")
        logger.info(f"[OK] Cleared {cleared_sessions} sessions")

        return {
            "success": True,
            "message": f"System prompt applied successfully! Using {prompt_type}",
            "prompt_type": prompt_type,
            "sessions_cleared": cleared_sessions,
            "prompt_preview": final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt
        }

    except Exception as e:
        logger.error(f"Failed to apply system prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to apply prompt: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint - thin HTTP wrapper.

    v5.3.2: All orchestration logic has been moved to QueryPipeline.
    This endpoint only:
    1. Validates initialization
    2. Gets/creates session
    3. Calls pipeline.handle()
    4. Converts PipelineResult to ChatResponse
    """
    # === DEBUG HEARTBEAT: Proves logging works during request handling ===
    print(f"\n{'='*60}", flush=True)
    print(f"[HEARTBEAT] /chat endpoint invoked: {message.message[:50]}...", flush=True)
    print(f"{'='*60}", flush=True)
    logger.info(f"[HEARTBEAT] /chat endpoint invoked: {message.message[:50]}...")
    # === END DEBUG HEARTBEAT ===

    if agent is None or query_pipeline is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Get or create session
    session_id = message.session_id or str(uuid.uuid4())
    session = get_or_create_session(session_id, message.system_prompt)
    session["message_count"] += 1

    prompt_status = "custom prompt" if session.get("custom_prompt") else "default prompt"
    logger.info(
        f"Processing query [session={session_id}, {prompt_status}]: {message.message[:100]}..."
    )

    # Delegate to pipeline
    result = await query_pipeline.handle(
        query=message.message,
        session_id=session_id,
        session=session,
        row_limit=message.row_limit,
        include_tasks=message.include_tasks,
        system_prompt=message.system_prompt,
    )

    # Convert PipelineResult to ChatResponse
    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
        sql_query=result.sql_query,
        query_results=result.query_results,
        data=result.data,
        execution_time=result.execution_time,
        tasks=None,
        clarification_needed=result.clarification_needed,
        agent_reasoning=None,
        chart_suggestion=result.chart_suggestion,
        error=result.error,
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    session_list = []
    for sid, session in sessions.items():
        session_list.append(
            {
                "session_id": sid,
                "created_at": session["created_at"].isoformat(),
                "last_active": session["last_active"].isoformat(),
                "message_count": session["message_count"],
            }
        )

    return {
        "total_sessions": len(session_list),
        "sessions": session_list,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/models")
async def get_models():
    """Get model information"""
    return {
        "model": GROQ_MODEL,
        "provider": "Groq",
        "architecture": "NL-SQL Engine + DJPI (advisory) + Query Governance",
        "version": "5.0",
        "sql_engine": "NLSQLTableQueryEngine",
    }


@app.get("/table-info")
async def get_table_info():
    """Get database table information"""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        schema = db_manager.get_schema_text()
        return {
            "schema": schema,
            "table": "us_accidents",
            "total_records": "7.7M+",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
async def get_performance():
    """Get performance metrics (for frontend compatibility)"""
    total_sessions = len(sessions)
    total_messages = sum(s["message_count"] for s in sessions.values())

    return {
        "total_requests": total_messages,
        "active_sessions": total_sessions,
        "total_sessions_created": total_sessions,
        "avg_tasks_per_query": 1.0,
        "clarification_requests": 0,
        "clarification_rate": 0.0,
        "multi_step_queries": 0,
    }


@app.get("/agent/info")
async def get_agent_info():
    """Get agent/engine information (for frontend compatibility)"""
    if nl_sql_engine is None:
        raise HTTPException(status_code=503, detail="NL-SQL engine not initialized")

    # Get table count from db_manager
    table_count = len(db_manager.get_all_table_names()) if db_manager else 0

    return {
        "agent_type": "LlamaIndex NLSQLTableQueryEngine",
        "agent_model": GROQ_MODEL,
        "sql_engine": "NLSQLTableQueryEngine (v5.0 - Table-Scoped)",
        "visualization": "One-shot LLM (separate from SQL engine)",
        "features": {
            "nl_sql_engine": True,
            "dynamic_table_scoping": True,  # v5.0: Tables discovered, not hardcoded
            "hard_gated_execution": True,   # v5.0: No LLM prose on errors
            "djpi_advisory": True,
            "query_governance": True,
            "sql_validation": True,
            "session_management": True,
        },
        "version": "5.0",
        "table_count": table_count,
        "architecture_notes": {
            "sql_generation": "NLSQLTableQueryEngine constrained to discovered tables",
            "table_scoping": "Dynamic (discovered from DB, not hardcoded)",
            "djpi_mode": "Advisory only (explainability, not execution)",
            "error_handling": "Hard-gated (no LLM repair/explanation)",
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Use log_config=None to prevent uvicorn from overriding our logging config
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None,  # Don't let uvicorn override our logging
    )
