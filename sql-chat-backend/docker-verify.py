#!/usr/bin/env python3
"""
Docker Build Verification Script

This script runs DURING Docker build to verify that:
1. All LlamaIndex components are importable
2. NLSQLTableQueryEngine is functional
3. Groq LLM integration is available
4. No missing dependencies

If ANY check fails, the Docker build FAILS.
This ensures broken images cannot be built.

Usage (in Dockerfile):
    RUN python docker-verify.py
"""

import sys
import importlib
import importlib.metadata

# ============================================================================
# CONFIGURATION
# ============================================================================

REQUIRED_IMPORTS = [
    ("llama_index", "Meta-package"),
    ("llama_index.core", "Core library"),
    ("llama_index.core.query_engine", "Query engine module"),
    ("llama_index.core.query_engine.sql_query_engine", "SQL query engine"),
    ("llama_index.llms.groq", "Groq LLM integration"),
    ("fastapi", "FastAPI framework"),
    ("uvicorn", "ASGI server"),
    ("sqlalchemy", "Database ORM"),
    ("pydantic", "Data validation"),
    ("pandas", "Data processing"),
]

REQUIRED_CLASSES = [
    ("llama_index.core.query_engine", "NLSQLTableQueryEngine"),
    ("llama_index.core", "SQLDatabase"),
    ("llama_index.core", "Settings"),
    ("llama_index.llms.groq", "Groq"),
]

REQUIRED_PACKAGES = [
    "llama-index",
    "llama-index-core",
    "llama-index-llms-groq",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "psycopg2-binary",
    "pydantic",
    "pandas",
    "python-dotenv",
]

# ============================================================================
# VERIFICATION
# ============================================================================

def main():
    print("=" * 70)
    print("DOCKER BUILD VERIFICATION")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

    errors = []

    # Check 1: Imports
    print("[1/3] Verifying imports...")
    for import_path, description in REQUIRED_IMPORTS:
        try:
            importlib.import_module(import_path)
            print(f"  [OK] {import_path}")
        except ImportError as e:
            print(f"  [FAIL] {import_path}: {e}")
            errors.append(f"Cannot import {import_path}: {e}")

    print()

    # Check 2: Classes
    print("[2/3] Verifying NL-SQL classes...")
    for module_path, class_name in REQUIRED_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is None:
                print(f"  [FAIL] {class_name}: not found in {module_path}")
                errors.append(f"{class_name} not found in {module_path}")
            elif not callable(cls):
                print(f"  [FAIL] {class_name}: not callable")
                errors.append(f"{class_name} is not callable")
            else:
                print(f"  [OK] {class_name}")
        except ImportError as e:
            print(f"  [FAIL] {class_name}: {e}")
            errors.append(f"Cannot import {module_path}: {e}")

    print()

    # Check 3: Package versions
    print("[3/3] Verifying package versions...")
    for pkg in REQUIRED_PACKAGES:
        try:
            version = importlib.metadata.version(pkg)
            print(f"  [OK] {pkg}=={version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  [FAIL] {pkg}: NOT INSTALLED")
            errors.append(f"Package {pkg} is not installed")

    print()
    print("=" * 70)

    if errors:
        print("VERIFICATION FAILED!")
        print("=" * 70)
        for error in errors:
            print(f"  - {error}")
        print()
        print("The Docker image cannot be built with missing dependencies.")
        sys.exit(1)
    else:
        print("VERIFICATION PASSED!")
        print("=" * 70)
        print("All LlamaIndex NL-SQL components verified.")
        print("Docker image is ready for production.")
        sys.exit(0)


if __name__ == "__main__":
    main()
