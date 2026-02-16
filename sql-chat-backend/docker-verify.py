#!/usr/bin/env python3
"""
Docker Build Verification Script (LlamaIndex 0.14.x + Groq-only)

Runs DURING Docker build to verify:
1. Core LlamaIndex imports
2. NL→SQL engine availability
3. Groq LLM integration
4. Required packages installed

FAILS build if verification fails.
"""

import sys
import importlib
import importlib.metadata

# ============================================================================
# CONFIGURATION (UPDATED FOR LLAMAINDEX 0.14)
# ============================================================================

REQUIRED_IMPORTS = [
    ("llama_index.core", "Core library"),
    ("llama_index.core.query_engine", "Query engine module"),
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
    print("DOCKER BUILD VERIFICATION (LlamaIndex 0.12+)")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print()

    errors = []

    # ------------------------------------------------------------------------
    # 1. Verify imports
    # ------------------------------------------------------------------------
    print("[1/3] Verifying imports...")
    for import_path, description in REQUIRED_IMPORTS:
        try:
            importlib.import_module(import_path)
            print(f"  [OK] {import_path}")
        except ImportError as e:
            print(f"  [FAIL] {import_path}: {e}")
            errors.append(f"Cannot import {import_path}: {e}")

    print()

    # ------------------------------------------------------------------------
    # 2. Verify required classes / objects
    # ------------------------------------------------------------------------
    print("[2/3] Verifying NL-SQL components...")
    for module_path, name in REQUIRED_CLASSES:
        try:
            module = importlib.import_module(module_path)
            attr = getattr(module, name, None)

            if attr is None:
                print(f"  [FAIL] {name}: not found in {module_path}")
                errors.append(f"{name} not found in {module_path}")
            else:
                # Settings is NOT callable in 0.12+
                if name == "Settings":
                    print(f"  [OK] {name} (singleton object)")
                else:
                    print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            errors.append(f"Cannot import {module_path}: {e}")

    print()

    # ------------------------------------------------------------------------
    # 3. Verify package versions
    # ------------------------------------------------------------------------
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
        sys.exit(1)
    else:
        print("VERIFICATION PASSED!")
        print("=" * 70)
        print("LlamaIndex NL-SQL stack verified successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()