#!/usr/bin/env python3
"""
LlamaIndex NL-SQL Verification Script

PURPOSE:
    Validate that LlamaIndex is correctly installed, importable, and
    functionally working for NL-SQL operations.

CHECKS:
    1. Import Verification - All required modules importable with correct paths
    2. Version Consistency - No version mismatches between packages
    3. Runtime NL-SQL Smoke Test - Actual SQL generation works
    4. Environment Integrity - All modules resolve inside .venv
    5. Summary - PASS/WARN/FAIL with actionable fixes

USAGE:
    python verify_llamaindex.py [--smoke-test] [--verbose]

    --smoke-test  Run the full NL-SQL runtime test (requires DB connection)
    --verbose     Show detailed module paths and diagnostics
"""

import sys
import os
import importlib
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    details: List[str] = field(default_factory=list)
    fix: Optional[str] = None

# Required LlamaIndex imports for NL-SQL
REQUIRED_IMPORTS = [
    # (import_path, description, is_critical)
    ("llama_index.core", "Core library", True),
    ("llama_index.core.query_engine", "Query engine module", True),
    ("llama_index.core.query_engine.sql_query_engine", "SQL query engine", True),
    ("llama_index.core.indices.struct_store.sql_query", "SQL query internals", True),
    ("llama_index.core.utilities.sql_wrapper", "SQL wrapper utilities", True),
    ("llama_index.llms.groq", "Groq LLM integration", True),
]

# Specific classes/functions that must exist
REQUIRED_CLASSES = [
    ("llama_index.core.query_engine", "NLSQLTableQueryEngine"),
    ("llama_index.core", "SQLDatabase"),
    ("llama_index.core", "Settings"),
    ("llama_index.llms.groq", "Groq"),
]

# Expected package versions (Groq-only — no meta-package)
EXPECTED_VERSIONS = {
    "llama-index-core": "0.14",
    "llama-index-llms-groq": "0.4",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_venv_path() -> Tuple[bool, str]:
    """Get the virtual environment path."""
    in_venv = hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    if in_venv:
        interpreter_path = Path(sys.executable)
        if sys.platform == "win32":
            venv_path = str(interpreter_path.parent.parent)
        else:
            venv_path = str(interpreter_path.parent.parent)
        return True, venv_path
    return False, ""


def get_site_packages() -> List[str]:
    """Get all site-packages directories."""
    import site
    return site.getsitepackages() + [site.getusersitepackages()]


def is_path_in_venv(path: str, venv_path: str) -> bool:
    """Check if a path is inside the virtual environment."""
    if not venv_path:
        return False
    try:
        path_resolved = Path(path).resolve()
        venv_resolved = Path(venv_path).resolve()
        return str(path_resolved).startswith(str(venv_resolved))
    except Exception:
        return False


def get_package_version(package_name: str) -> Optional[str]:
    """Get installed package version."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def print_status(status: Status, message: str, indent: int = 0):
    """Print a status message with color indicators."""
    prefix = "  " * indent
    if status == Status.PASS:
        indicator = "[PASS]"
    elif status == Status.WARN:
        indicator = "[WARN]"
    else:
        indicator = "[FAIL]"
    print(f"{prefix}{indicator} {message}")


# ============================================================================
# CHECK 1: IMPORT VERIFICATION
# ============================================================================

def check_imports(verbose: bool = False) -> CheckResult:
    """
    Verify all required LlamaIndex modules are importable.
    """
    print("\n" + "=" * 70)
    print("CHECK 1: IMPORT VERIFICATION")
    print("=" * 70)

    errors = []
    warnings = []
    details = []

    in_venv, venv_path = get_venv_path()

    for import_path, description, is_critical in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(import_path)
            module_file = getattr(module, "__file__", None)

            if module_file:
                # Check if module is from venv
                if venv_path and not is_path_in_venv(module_file, venv_path):
                    msg = f"{import_path}: Loaded from OUTSIDE venv"
                    warnings.append(msg)
                    print_status(Status.WARN, f"{import_path}")
                    if verbose:
                        print(f"      Path: {module_file}")
                        print(f"      Expected venv: {venv_path}")
                else:
                    print_status(Status.PASS, f"{import_path}")
                    if verbose:
                        print(f"      Path: {module_file}")

                details.append(f"{import_path}: {module_file}")
            else:
                print_status(Status.PASS, f"{import_path} (namespace package)")
                details.append(f"{import_path}: namespace package")

        except ImportError as e:
            msg = f"{import_path}: {e}"
            if is_critical:
                errors.append(msg)
                print_status(Status.FAIL, f"{import_path}")
            else:
                warnings.append(msg)
                print_status(Status.WARN, f"{import_path}")
            print(f"      Error: {e}")

    # Check required classes
    print("\n  Verifying specific classes:")
    for module_path, class_name in REQUIRED_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is None:
                errors.append(f"{module_path}.{class_name}: Not found")
                print_status(Status.FAIL, f"{class_name}", indent=1)
            elif not callable(cls):
                errors.append(f"{module_path}.{class_name}: Not callable")
                print_status(Status.FAIL, f"{class_name} (not callable)", indent=1)
            else:
                print_status(Status.PASS, f"{class_name}", indent=1)
        except ImportError as e:
            errors.append(f"{module_path}.{class_name}: Import failed - {e}")
            print_status(Status.FAIL, f"{class_name}", indent=1)

    # Determine overall status
    if errors:
        return CheckResult(
            name="Import Verification",
            status=Status.FAIL,
            message=f"{len(errors)} critical import(s) failed",
            details=details,
            fix="pip install llama-index llama-index-core llama-index-llms-groq"
        )
    elif warnings:
        return CheckResult(
            name="Import Verification",
            status=Status.WARN,
            message=f"All imports successful, {len(warnings)} warning(s)",
            details=details,
            fix="Reinstall packages in venv: pip install --force-reinstall llama-index"
        )
    else:
        return CheckResult(
            name="Import Verification",
            status=Status.PASS,
            message="All imports successful",
            details=details
        )


# ============================================================================
# CHECK 2: VERSION CONSISTENCY
# ============================================================================

def check_versions(verbose: bool = False) -> CheckResult:
    """
    Check version consistency between LlamaIndex packages.
    """
    print("\n" + "=" * 70)
    print("CHECK 2: VERSION CONSISTENCY")
    print("=" * 70)

    errors = []
    warnings = []
    details = []
    versions = {}

    # Get all llama-index related packages
    all_packages = []
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name", "").lower()
            if "llama" in name or "llama-index" in name:
                all_packages.append((name, dist.version))
    except Exception as e:
        warnings.append(f"Could not enumerate packages: {e}")

    # Check expected versions
    for package_name, expected_prefix in EXPECTED_VERSIONS.items():
        version = get_package_version(package_name)
        versions[package_name] = version

        if version is None:
            errors.append(f"{package_name}: NOT INSTALLED")
            print_status(Status.FAIL, f"{package_name}: NOT INSTALLED")
        elif not version.startswith(expected_prefix):
            warnings.append(
                f"{package_name}: {version} (expected {expected_prefix}.x)"
            )
            print_status(Status.WARN, f"{package_name}: {version} (expected {expected_prefix}.x)")
        else:
            print_status(Status.PASS, f"{package_name}: {version}")

        if version:
            details.append(f"{package_name}=={version}")

    # Check for unwanted OpenAI provider
    print("\n  Checking LLM provider isolation:")
    core_version = versions.get("llama-index-core")
    openai_detected = False
    for name, version in all_packages:
        if "openai" in name.lower():
            openai_detected = True
            warnings.append(
                f"{name}=={version} is installed but should NOT be (causes OpenAI fallback). "
                f"Remove with: pip uninstall {name}"
            )
            print_status(
                Status.WARN,
                f"{name}=={version} detected — causes OpenAI fallback!",
                indent=1
            )
    if not openai_detected:
        print_status(Status.PASS, "No OpenAI provider packages detected (Groq-only)", indent=1)

    # List all LlamaIndex packages found
    if verbose and all_packages:
        print("\n  All LlamaIndex packages installed:")
        for name, version in sorted(all_packages):
            print(f"      {name}=={version}")
            details.append(f"{name}=={version}")

    # Determine overall status
    if errors:
        return CheckResult(
            name="Version Consistency",
            status=Status.FAIL,
            message=f"{len(errors)} package(s) missing",
            details=details,
            fix="pip install llama-index-core==0.14.5 llama-index-llms-groq==0.4.1"
        )
    elif warnings:
        return CheckResult(
            name="Version Consistency",
            status=Status.WARN,
            message=f"{len(warnings)} version warning(s)",
            details=details,
            fix="pip install --upgrade llama-index llama-index-core llama-index-llms-groq"
        )
    else:
        return CheckResult(
            name="Version Consistency",
            status=Status.PASS,
            message="All versions consistent",
            details=details
        )


# ============================================================================
# CHECK 3: RUNTIME NL-SQL SMOKE TEST
# ============================================================================

def check_nlsql_runtime(verbose: bool = False) -> CheckResult:
    """
    Run a minimal NL-SQL smoke test to verify functionality.
    """
    print("\n" + "=" * 70)
    print("CHECK 3: RUNTIME NL-SQL SMOKE TEST")
    print("=" * 70)

    errors = []
    details = []

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        errors.append("python-dotenv not installed")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="Cannot load environment",
            fix="pip install python-dotenv"
        )

    database_url = os.getenv("DATABASE_URL")
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not database_url:
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.WARN,
            message="DATABASE_URL not set - skipping runtime test",
            fix="Set DATABASE_URL in .env file"
        )

    if not groq_api_key:
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.WARN,
            message="GROQ_API_KEY not set - skipping runtime test",
            fix="Set GROQ_API_KEY in .env file"
        )

    # Step 1: Create SQLAlchemy engine
    print("  Step 1: Creating SQLAlchemy engine...")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row and row[0] == 1:
                print_status(Status.PASS, "Database connection successful")
                details.append("SQLAlchemy engine: OK")
            else:
                errors.append("Database returned unexpected result")
                print_status(Status.FAIL, "Database returned unexpected result")

    except Exception as e:
        errors.append(f"Database connection failed: {e}")
        print_status(Status.FAIL, f"Database connection failed: {e}")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="Database connection failed",
            details=details,
            fix=f"Check DATABASE_URL and database server"
        )

    # Step 2: Initialize SQLDatabase
    print("  Step 2: Initializing SQLDatabase...")
    try:
        from llama_index.core import SQLDatabase

        # Get table names to pass to SQLDatabase
        from sqlalchemy import inspect
        inspector = inspect(engine)

        # Find tables (try common schemas)
        tables = []
        for schema in [None, "public", "postgres_air"]:
            try:
                schema_tables = inspector.get_table_names(schema=schema)
                if schema_tables:
                    tables = schema_tables[:5]  # Limit for smoke test
                    active_schema = schema
                    break
            except Exception:
                continue

        if not tables:
            errors.append("No tables found in database")
            print_status(Status.FAIL, "No tables found")
            return CheckResult(
                name="NL-SQL Runtime",
                status=Status.FAIL,
                message="No tables found in database",
                details=details
            )

        sql_database = SQLDatabase(
            engine,
            schema=active_schema,
            include_tables=tables
        )
        print_status(Status.PASS, f"SQLDatabase initialized ({len(tables)} tables)")
        details.append(f"SQLDatabase: {len(tables)} tables loaded")
        if verbose:
            print(f"      Tables: {tables}")

    except Exception as e:
        errors.append(f"SQLDatabase initialization failed: {e}")
        print_status(Status.FAIL, f"SQLDatabase failed: {e}")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="SQLDatabase initialization failed",
            details=details,
            fix="Check llama-index-core installation"
        )

    # Step 3: Initialize LLM
    print("  Step 3: Initializing Groq LLM...")
    try:
        from llama_index.llms.groq import Groq
        from llama_index.core import Settings

        llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=groq_api_key,
            temperature=0.1,
        )
        Settings.llm = llm
        Settings.embed_model = None  # Disable embeddings for NL-SQL
        print_status(Status.PASS, "Groq LLM initialized")
        details.append("Groq LLM: OK")

    except Exception as e:
        errors.append(f"LLM initialization failed: {e}")
        print_status(Status.FAIL, f"LLM failed: {e}")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="LLM initialization failed",
            details=details,
            fix="Check GROQ_API_KEY and llama-index-llms-groq installation"
        )

    # Step 4: Create NLSQLTableQueryEngine
    print("  Step 4: Creating NLSQLTableQueryEngine...")
    try:
        from llama_index.core.query_engine import NLSQLTableQueryEngine

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=tables,
            llm=llm,
            synthesize_response=False,
            verbose=False,
        )
        print_status(Status.PASS, "NLSQLTableQueryEngine created")
        details.append("NLSQLTableQueryEngine: OK")

    except Exception as e:
        errors.append(f"Query engine creation failed: {e}")
        print_status(Status.FAIL, f"Query engine failed: {e}")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="Query engine creation failed",
            details=details,
            fix="Check llama-index-core installation"
        )

    # Step 5: Execute minimal query
    print("  Step 5: Executing NL-SQL query...")
    try:
        # Use the first table for a simple query
        test_table = tables[0]
        test_query = f"Show 1 row from {test_table}"

        response = query_engine.query(test_query)

        # Check response has SQL
        sql_query = None
        if hasattr(response, 'metadata') and response.metadata:
            sql_query = response.metadata.get('sql_query')

        if sql_query:
            print_status(Status.PASS, "SQL generated successfully")
            details.append(f"Generated SQL: {sql_query[:100]}...")
            if verbose:
                print(f"      Query: {test_query}")
                print(f"      SQL: {sql_query}")

            # Verify it's a SELECT statement
            if sql_query.strip().upper().startswith("SELECT"):
                print_status(Status.PASS, "Valid SELECT statement generated")
            else:
                print_status(Status.WARN, "Generated SQL is not a SELECT")

        else:
            errors.append("No SQL generated from query")
            print_status(Status.FAIL, "No SQL in response metadata")

    except Exception as e:
        errors.append(f"Query execution failed: {e}")
        print_status(Status.FAIL, f"Query failed: {e}")
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message="Query execution failed",
            details=details,
            fix="Check database schema and LLM API access"
        )

    # Summary
    if errors:
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.FAIL,
            message=f"{len(errors)} error(s) during smoke test",
            details=details
        )
    else:
        return CheckResult(
            name="NL-SQL Runtime",
            status=Status.PASS,
            message="NL-SQL pipeline fully functional",
            details=details
        )


# ============================================================================
# CHECK 4: ENVIRONMENT INTEGRITY
# ============================================================================

def check_environment_integrity(verbose: bool = False) -> CheckResult:
    """
    Verify all LlamaIndex modules resolve inside .venv.
    """
    print("\n" + "=" * 70)
    print("CHECK 4: ENVIRONMENT INTEGRITY")
    print("=" * 70)

    warnings = []
    details = []

    # Print basic environment info
    print(f"  sys.executable: {sys.executable}")
    print(f"  sys.prefix: {sys.prefix}")
    print(f"  sys.base_prefix: {getattr(sys, 'base_prefix', 'N/A')}")
    details.append(f"Interpreter: {sys.executable}")

    in_venv, venv_path = get_venv_path()
    print(f"  In virtualenv: {in_venv}")
    if venv_path:
        print(f"  Venv path: {venv_path}")
        details.append(f"Venv: {venv_path}")

    if not in_venv:
        return CheckResult(
            name="Environment Integrity",
            status=Status.FAIL,
            message="Not running in a virtual environment",
            details=details,
            fix="Activate the virtual environment: .venv\\Scripts\\activate"
        )

    # Get site-packages
    site_packages = get_site_packages()
    print(f"\n  Site-packages directories:")
    for sp in site_packages:
        in_venv_sp = is_path_in_venv(sp, venv_path)
        status = Status.PASS if in_venv_sp else Status.WARN
        print_status(status, sp, indent=1)
        if not in_venv_sp and os.path.exists(sp):
            warnings.append(f"Site-packages outside venv: {sp}")

    # Check LlamaIndex module locations
    print(f"\n  LlamaIndex module locations:")
    modules_to_check = [
        "llama_index",
        "llama_index.core",
        "llama_index.llms.groq",
    ]

    outside_venv = []
    for mod_name in modules_to_check:
        try:
            module = importlib.import_module(mod_name)
            mod_file = getattr(module, "__file__", None)
            if mod_file:
                in_venv_mod = is_path_in_venv(mod_file, venv_path)
                if in_venv_mod:
                    print_status(Status.PASS, f"{mod_name}", indent=1)
                else:
                    print_status(Status.WARN, f"{mod_name} (outside venv)", indent=1)
                    outside_venv.append(mod_name)
                    warnings.append(f"{mod_name} loaded from outside venv: {mod_file}")
                if verbose:
                    print(f"          {mod_file}")
            else:
                print_status(Status.PASS, f"{mod_name} (namespace)", indent=1)
        except ImportError as e:
            print_status(Status.FAIL, f"{mod_name}: {e}", indent=1)

    # Check for global Python paths in sys.path
    print(f"\n  Checking sys.path for global Python:")
    global_paths = []
    for p in sys.path:
        if p and not is_path_in_venv(p, venv_path):
            # Ignore empty string (cwd) and stdlib paths
            if p and "site-packages" in p.lower():
                global_paths.append(p)
                print_status(Status.WARN, f"Global path in sys.path: {p}", indent=1)

    if global_paths:
        warnings.append(f"Found {len(global_paths)} global site-packages in sys.path")

    # Summary
    if warnings:
        return CheckResult(
            name="Environment Integrity",
            status=Status.WARN,
            message=f"{len(warnings)} warning(s) about environment paths",
            details=details,
            fix="Reinstall packages: pip install --force-reinstall -r requirements.txt"
        )
    else:
        return CheckResult(
            name="Environment Integrity",
            status=Status.PASS,
            message="All modules resolve inside .venv",
            details=details
        )


# ============================================================================
# MAIN VERIFICATION
# ============================================================================

def run_verification(smoke_test: bool = False, verbose: bool = False) -> bool:
    """
    Run all verification checks.
    """
    print("=" * 70)
    print("LLAMAINDEX NL-SQL VERIFICATION")
    print("=" * 70)
    print(f"Timestamp: {__import__('datetime').datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")

    results: List[CheckResult] = []

    # Run checks
    results.append(check_imports(verbose))
    results.append(check_versions(verbose))

    if smoke_test:
        results.append(check_nlsql_runtime(verbose))
    else:
        print("\n" + "=" * 70)
        print("CHECK 3: RUNTIME NL-SQL SMOKE TEST")
        print("=" * 70)
        print("  [SKIP] Use --smoke-test to run runtime verification")
        results.append(CheckResult(
            name="NL-SQL Runtime",
            status=Status.WARN,
            message="Skipped (use --smoke-test flag)",
            details=[]
        ))

    results.append(check_environment_integrity(verbose))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    has_fail = False
    has_warn = False

    for result in results:
        status_str = result.status.value
        print(f"\n  {result.name}: {status_str}")
        print(f"    {result.message}")

        if result.status == Status.FAIL:
            has_fail = True
            if result.fix:
                print(f"    FIX: {result.fix}")
        elif result.status == Status.WARN:
            has_warn = True
            if result.fix:
                print(f"    FIX: {result.fix}")

    # Final verdict
    print("\n" + "=" * 70)
    if has_fail:
        print("FINAL RESULT: FAIL")
        print("=" * 70)
        print("\nLlamaIndex NL-SQL is NOT properly installed.")
        print("Fix the errors above before running OptimaX.")
        return False
    elif has_warn:
        print("FINAL RESULT: WARN")
        print("=" * 70)
        print("\nLlamaIndex NL-SQL is installed but has warnings.")
        print("Application may work, but consider fixing warnings.")
        return True
    else:
        print("FINAL RESULT: PASS")
        print("=" * 70)
        print("\nLlamaIndex NL-SQL is correctly installed and verified.")
        print("Ready for production use.")
        return True


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify LlamaIndex NL-SQL installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_llamaindex.py                    # Basic verification
  python verify_llamaindex.py --smoke-test       # Include runtime test
  python verify_llamaindex.py --smoke-test -v    # Verbose output
        """
    )
    parser.add_argument(
        "--smoke-test", "-s",
        action="store_true",
        help="Run full NL-SQL runtime test (requires DB connection)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed module paths and diagnostics"
    )

    args = parser.parse_args()

    try:
        success = run_verification(
            smoke_test=args.smoke_test,
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
