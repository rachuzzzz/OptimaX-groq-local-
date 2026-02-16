#!/usr/bin/env python3
"""
OptimaX Environment Guard - Fail-Fast Startup Validation

This module MUST be imported at the TOP of main.py (before any other imports)
to ensure the environment is correct before proceeding.

GUARANTEES:
1. Python interpreter is from the expected .venv
2. All critical dependencies are importable
3. LlamaIndex NL-SQL components are verified
4. Database connection is available (via env vars)

USAGE:
    from env_guard import validate_environment
    validate_environment()  # Raises EnvironmentError if invalid
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================
EXPECTED_VENV_NAME = ".venv"  # Expected virtual environment directory name
REQUIRED_PACKAGES = [
    # (module_name, package_name, verification_attribute)
    # NOTE: We do NOT require the llama-index meta-package (it pulls in OpenAI).
    # Only llama-index-core and llama-index-llms-groq are needed.
    ("llama_index.core", "llama-index-core", None),
    ("llama_index.llms.groq", "llama-index-llms-groq", None),
    ("llama_index.core.query_engine", "llama-index-core", "NLSQLTableQueryEngine"),
    ("fastapi", "fastapi", "__version__"),
    ("uvicorn", "uvicorn", "__version__"),
    ("sqlalchemy", "sqlalchemy", "__version__"),
    ("pydantic", "pydantic", "__version__"),
    ("dotenv", "python-dotenv", None),
    ("pandas", "pandas", "__version__"),
]

REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "DATABASE_URL",
]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def get_venv_info() -> Tuple[bool, str, str]:
    """
    Check if running inside a virtual environment.

    Returns:
        (is_venv, venv_path, interpreter_path)
    """
    interpreter = sys.executable

    # Check for virtual environment indicators
    in_venv = (
        hasattr(sys, 'real_prefix') or  # virtualenv
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)  # venv
    )

    venv_path = ""
    if in_venv:
        # Derive venv path from interpreter location
        # Windows: .venv\Scripts\python.exe -> .venv
        # Unix: .venv/bin/python -> .venv
        interpreter_path = Path(interpreter)
        if sys.platform == "win32":
            # .venv\Scripts\python.exe
            venv_path = str(interpreter_path.parent.parent)
        else:
            # .venv/bin/python
            venv_path = str(interpreter_path.parent.parent)

    return in_venv, venv_path, interpreter


def validate_interpreter() -> List[str]:
    """
    Validate the Python interpreter is from the expected virtual environment.

    Returns:
        List of warning/error messages (empty if valid)
    """
    errors = []

    in_venv, venv_path, interpreter = get_venv_info()

    if not in_venv:
        errors.append(
            f"NOT RUNNING IN VIRTUAL ENVIRONMENT!\n"
            f"  Current interpreter: {interpreter}\n"
            f"  Expected: .venv/Scripts/python.exe (Windows) or .venv/bin/python (Unix)\n"
            f"  Solution: Activate the virtual environment before running:\n"
            f"    Windows: .venv\\Scripts\\activate\n"
            f"    Unix:    source .venv/bin/activate"
        )
    else:
        venv_name = Path(venv_path).name
        if venv_name not in [EXPECTED_VENV_NAME, "venv"]:  # Accept both .venv and venv
            errors.append(
                f"Virtual environment name mismatch:\n"
                f"  Found: {venv_name}\n"
                f"  Expected: {EXPECTED_VENV_NAME}\n"
                f"  This may cause VS Code/debugger misalignment."
            )

    return errors


def validate_packages() -> Tuple[List[str], List[str]]:
    """
    Validate all required packages are importable.

    Returns:
        (errors, package_info)
    """
    errors = []
    info = []

    for module_name, package_name, verify_attr in REQUIRED_PACKAGES:
        try:
            module = __import__(module_name, fromlist=[''])

            # Get version or verification attribute
            if verify_attr:
                if hasattr(module, verify_attr):
                    attr_value = getattr(module, verify_attr)
                    if verify_attr == "__version__":
                        info.append(f"  {package_name}: {attr_value}")
                    else:
                        info.append(f"  {package_name}: {verify_attr} found")
                else:
                    info.append(f"  {package_name}: imported (no {verify_attr})")
            else:
                info.append(f"  {package_name}: imported")

            # Get file location for debugging
            if hasattr(module, "__file__") and module.__file__:
                module_path = Path(module.__file__)
                # Check if module is from our venv
                _, venv_path, _ = get_venv_info()
                if venv_path and venv_path not in str(module_path):
                    errors.append(
                        f"Package {package_name} is NOT from the active venv!\n"
                        f"  Package location: {module_path}\n"
                        f"  Expected venv: {venv_path}"
                    )

        except ImportError as e:
            errors.append(
                f"MISSING PACKAGE: {package_name}\n"
                f"  Import error: {e}\n"
                f"  Solution: pip install {package_name}"
            )

    return errors, info


def validate_llama_index_nlsql() -> List[str]:
    """
    Specifically validate LlamaIndex NL-SQL engine components.

    Returns:
        List of errors (empty if valid)
    """
    errors = []

    try:
        from llama_index.core.query_engine import NLSQLTableQueryEngine
        from llama_index.core import SQLDatabase
        from llama_index.llms.groq import Groq

        # Verify these are the actual classes, not stubs
        if not callable(NLSQLTableQueryEngine):
            errors.append("NLSQLTableQueryEngine is not callable - broken import")
        if not callable(SQLDatabase):
            errors.append("SQLDatabase is not callable - broken import")
        if not callable(Groq):
            errors.append("Groq LLM is not callable - broken import")

    except ImportError as e:
        errors.append(
            f"LLAMA_INDEX NL-SQL COMPONENTS NOT AVAILABLE!\n"
            f"  Import error: {e}\n"
            f"  This is a CRITICAL ERROR - the application cannot function.\n"
            f"  Solution: pip install llama-index-core llama-index-llms-groq"
        )

    return errors


def validate_env_vars() -> List[str]:
    """
    Validate required environment variables are set.

    Returns:
        List of errors (empty if valid)
    """
    errors = []

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # Will be caught by package validation

    for var_name in REQUIRED_ENV_VARS:
        value = os.getenv(var_name)
        if not value:
            errors.append(
                f"MISSING ENVIRONMENT VARIABLE: {var_name}\n"
                f"  Solution: Add {var_name}=your_value to .env file"
            )
        elif var_name == "GROQ_API_KEY" and not value.startswith("gsk_"):
            errors.append(
                f"INVALID {var_name} FORMAT\n"
                f"  Expected format: gsk_... (Groq API keys start with 'gsk_')\n"
                f"  Get a valid key at: https://console.groq.com/keys"
            )

    return errors


def check_multiple_interpreters() -> Optional[str]:
    """
    Warn if multiple Python interpreters are detected in PATH.

    Returns:
        Warning message if multiple interpreters found, None otherwise
    """
    import shutil

    python_paths = []

    # Find all python executables in PATH
    for name in ["python", "python3", "python3.11", "python3.12"]:
        path = shutil.which(name)
        if path and path not in python_paths:
            python_paths.append(path)

    if len(python_paths) > 1:
        return (
            f"MULTIPLE PYTHON INTERPRETERS IN PATH:\n" +
            "\n".join(f"  - {p}" for p in python_paths) +
            f"\n  Active interpreter: {sys.executable}"
        )

    return None


def validate_environment(strict: bool = True) -> bool:
    """
    Run all environment validations.

    Args:
        strict: If True, raise EnvironmentError on any failure.
                If False, print warnings and return success status.

    Returns:
        True if environment is valid, False otherwise.

    Raises:
        EnvironmentError: If strict=True and validation fails.
    """
    print("=" * 70)
    print("OPTIMAX ENVIRONMENT GUARD - Startup Validation")
    print("=" * 70)

    all_errors = []

    # 1. Interpreter validation
    print("\n[1/5] Checking Python interpreter...")
    interpreter_errors = validate_interpreter()
    all_errors.extend(interpreter_errors)

    in_venv, venv_path, interpreter = get_venv_info()
    print(f"  Interpreter: {interpreter}")
    print(f"  In venv: {in_venv}")
    if venv_path:
        print(f"  Venv path: {venv_path}")

    # 2. Package validation
    print("\n[2/5] Checking required packages...")
    package_errors, package_info = validate_packages()
    all_errors.extend(package_errors)
    for info in package_info:
        print(info)

    # 3. LlamaIndex NL-SQL validation
    print("\n[3/5] Checking LlamaIndex NL-SQL components...")
    nlsql_errors = validate_llama_index_nlsql()
    all_errors.extend(nlsql_errors)
    if not nlsql_errors:
        print("  NLSQLTableQueryEngine: OK")
        print("  SQLDatabase: OK")
        print("  Groq LLM: OK")

    # 4. Environment variables
    print("\n[4/5] Checking environment variables...")
    env_errors = validate_env_vars()
    all_errors.extend(env_errors)
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            masked = value[:4] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  {var}: {masked}")
        else:
            print(f"  {var}: NOT SET")

    # 5. Multiple interpreter warning
    print("\n[5/5] Checking for interpreter ambiguity...")
    multi_warning = check_multiple_interpreters()
    if multi_warning:
        print(f"  WARNING: {multi_warning}")
    else:
        print("  No interpreter ambiguity detected")

    # Summary
    print("\n" + "=" * 70)

    if all_errors:
        print("ENVIRONMENT VALIDATION FAILED!")
        print("=" * 70)
        for i, error in enumerate(all_errors, 1):
            print(f"\nError {i}:")
            print(error)

        if strict:
            raise EnvironmentError(
                f"Environment validation failed with {len(all_errors)} error(s). "
                f"See above for details."
            )
        return False
    else:
        print("ENVIRONMENT VALIDATION PASSED!")
        print("=" * 70)
        print(f"\nAll checks passed. Ready to start OptimaX.")
        return True


def check_llama_index_versions() -> Tuple[List[str], Dict[str, str]]:
    """
    Check LlamaIndex package version consistency.

    Returns:
        (warnings, versions_dict)
    """
    warnings = []
    versions = {}

    try:
        import importlib.metadata

        # Expected version prefixes (Groq-only — no meta-package)
        expected = {
            "llama-index-core": "0.14",
            "llama-index-llms-groq": "0.4",
        }

        for pkg, expected_prefix in expected.items():
            try:
                version = importlib.metadata.version(pkg)
                versions[pkg] = version
                if not version.startswith(expected_prefix):
                    warnings.append(
                        f"{pkg}: {version} (expected {expected_prefix}.x)"
                    )
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "NOT INSTALLED"
                warnings.append(f"{pkg}: NOT INSTALLED")

        # Check for drift packages (register OpenAI as default LLM)
        # NOTE: llama-index-llms-openai and llama-index-llms-openai-like are
        # REQUIRED transitive deps of llama-index-llms-groq (Groq inherits
        # from OpenAILike). Only the EXTRA adapters below cause drift.
        drift_packages = [
            "llama-index-agent-openai",
            "llama-index-embeddings-openai",
            "llama-index-multi-modal-llms-openai",
            "llama-index-program-openai",
            "llama-index-question-gen-openai",
        ]
        for drift_pkg in drift_packages:
            try:
                drift_ver = importlib.metadata.version(drift_pkg)
                versions[drift_pkg] = drift_ver
                warnings.append(
                    f"{drift_pkg}={drift_ver} causes LLM drift. "
                    f"Remove with: pip uninstall {drift_pkg}"
                )
            except importlib.metadata.PackageNotFoundError:
                pass  # Good — not installed

    except ImportError:
        warnings.append("importlib.metadata not available")

    return warnings, versions


def print_environment_report():
    """
    Print a detailed environment report for debugging.
    """
    print("\n" + "=" * 70)
    print("DETAILED ENVIRONMENT REPORT")
    print("=" * 70)

    print(f"\nPython Version: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Prefix: {sys.prefix}")
    print(f"Base Prefix: {getattr(sys, 'base_prefix', 'N/A')}")
    print(f"Platform: {sys.platform}")

    print(f"\nWorking Directory: {os.getcwd()}")

    print("\nPYTHONPATH:")
    for p in sys.path[:10]:
        print(f"  {p}")
    if len(sys.path) > 10:
        print(f"  ... and {len(sys.path) - 10} more")

    # LlamaIndex locations
    print("\nLlamaIndex Installation:")
    try:
        import llama_index
        print(f"  llama_index: {llama_index.__file__}")
        print(f"  Version: {getattr(llama_index, '__version__', 'unknown')}")
    except ImportError:
        print("  NOT INSTALLED")

    try:
        import llama_index.core as lic
        print(f"  llama_index.core: {lic.__file__}")
    except ImportError:
        print("  llama_index.core: NOT INSTALLED")

    # Version consistency
    print("\nVersion Consistency Check:")
    warnings, versions = check_llama_index_versions()
    for pkg, ver in versions.items():
        status = "OK" if pkg not in [w.split(":")[0] for w in warnings] else "WARN"
        print(f"  [{status}] {pkg}: {ver}")
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")


# ============================================================================
# QUICK LLAMAINDEX VERIFICATION (for startup)
# ============================================================================

def quick_verify_llamaindex() -> Tuple[bool, List[str]]:
    """
    Quick verification that LlamaIndex NL-SQL is properly installed.

    This is faster than the full verify_llamaindex.py script and suitable
    for startup checks.

    Returns:
        (success, messages)
    """
    messages = []
    success = True

    # 1. Check imports
    required_imports = [
        ("llama_index.core.query_engine", "NLSQLTableQueryEngine"),
        ("llama_index.core", "SQLDatabase"),
        ("llama_index.llms.groq", "Groq"),
    ]

    for module_path, class_name in required_imports:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is None:
                messages.append(f"[FAIL] {class_name} not found in {module_path}")
                success = False
            elif not callable(cls):
                messages.append(f"[WARN] {class_name} not callable")
        except ImportError as e:
            messages.append(f"[FAIL] Cannot import {module_path}: {e}")
            success = False

    # 2. Check versions
    version_warnings, versions = check_llama_index_versions()
    for w in version_warnings:
        messages.append(f"[WARN] {w}")

    # 3. Check module paths
    in_venv, venv_path, _ = get_venv_info()
    if in_venv and venv_path:
        try:
            import llama_index.core
            core_path = getattr(llama_index.core, "__file__", "")
            if core_path and venv_path not in core_path:
                messages.append(f"[WARN] llama_index.core loaded from outside venv")
        except ImportError:
            pass

    if success and not version_warnings:
        messages.insert(0, "[OK] LlamaIndex NL-SQL verified")
    elif success:
        messages.insert(0, "[OK] LlamaIndex NL-SQL importable (with warnings)")
    else:
        messages.insert(0, "[FAIL] LlamaIndex NL-SQL verification failed")

    return success, messages


# ============================================================================
# MAIN (for standalone testing)
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptimaX Environment Guard")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code on failure"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed environment report"
    )
    args = parser.parse_args()

    if args.report:
        print_environment_report()

    try:
        success = validate_environment(strict=args.strict)
        sys.exit(0 if success else 1)
    except EnvironmentError as e:
        print(f"\n\nFATAL: {e}")
        sys.exit(1)
