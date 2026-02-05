#!/bin/bash
# ============================================================================
# OptimaX - Virtual Environment Setup (macOS/Linux)
# ============================================================================
# This script creates a deterministic Python virtual environment.
# Run: chmod +x setup-venv.sh && ./setup-venv.sh
# ============================================================================

set -e  # Exit on error

VENV_NAME=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  OptimaX Virtual Environment Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Parse arguments
FORCE=false
NO_INSTALL=false
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        --no-install) NO_INSTALL=true ;;
    esac
done

# Step 1: Check Python version
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"

# Try python3 first, then python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}  ERROR: Python not found in PATH${NC}"
    echo -e "${RED}  Please install Python 3.9+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}  Found: $PYTHON_VERSION${NC}"

# Verify Python 3.9+
VERSION_NUM=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo $VERSION_NUM | cut -d. -f1)
MINOR=$(echo $VERSION_NUM | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo -e "${RED}  ERROR: Python 3.9+ required, found $VERSION_NUM${NC}"
    exit 1
fi

# Step 2: Check for existing venv
echo ""
echo -e "${YELLOW}[2/5] Checking for existing virtual environment...${NC}"

if [ -d "$VENV_PATH" ]; then
    if [ "$FORCE" = true ]; then
        echo -e "${YELLOW}  Removing existing venv (--force specified)...${NC}"
        rm -rf "$VENV_PATH"
    else
        echo -e "${YELLOW}  Found existing venv at: $VENV_PATH${NC}"
        read -p "  Continue with existing venv? (y/n) " response
        if [ "$response" != "y" ]; then
            echo -e "${YELLOW}  Removing existing venv...${NC}"
            rm -rf "$VENV_PATH"
        fi
    fi
fi

# Step 3: Create virtual environment
echo ""
echo -e "${YELLOW}[3/5] Creating virtual environment...${NC}"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${CYAN}  Creating $VENV_NAME at $VENV_PATH${NC}"
    $PYTHON_CMD -m venv "$VENV_PATH"

    if [ ! -d "$VENV_PATH" ]; then
        echo -e "${RED}  ERROR: Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}  Virtual environment created successfully${NC}"
else
    echo -e "${GREEN}  Using existing virtual environment${NC}"
fi

# Step 4: Activate and upgrade pip
echo ""
echo -e "${YELLOW}[4/5] Activating environment and upgrading pip...${NC}"

source "$VENV_PATH/bin/activate"
echo -e "${GREEN}  Activated: $VENV_NAME${NC}"

# Upgrade pip
echo -e "${CYAN}  Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Step 5: Install dependencies
echo ""
echo -e "${YELLOW}[5/5] Installing dependencies...${NC}"

if [ "$NO_INSTALL" = false ]; then
    if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
        echo -e "${CYAN}  Installing from requirements.txt...${NC}"
        pip install -r "$SCRIPT_DIR/requirements.txt"

        if [ $? -ne 0 ]; then
            echo -e "${RED}  ERROR: Package installation failed${NC}"
            exit 1
        fi
        echo -e "${GREEN}  All packages installed successfully${NC}"
    else
        echo -e "${YELLOW}  WARNING: requirements.txt not found${NC}"
    fi
else
    echo -e "${YELLOW}  Skipping package installation (--no-install)${NC}"
fi

# Verify LlamaIndex installation
echo ""
echo -e "${YELLOW}Verifying LlamaIndex installation...${NC}"
python -c "from llama_index.core.query_engine import NLSQLTableQueryEngine; print('  NLSQLTableQueryEngine: OK')" 2>&1 || \
    echo -e "${YELLOW}  WARNING: LlamaIndex NL-SQL not properly installed${NC}"

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Virtual environment: $VENV_PATH${NC}"
echo -e "${CYAN}Python interpreter:  $VENV_PATH/bin/python${NC}"
echo ""
echo -e "${YELLOW}To activate:${NC}"
echo -e "  source $VENV_NAME/bin/activate"
echo ""
echo -e "${YELLOW}To run the backend:${NC}"
echo -e "  python -m uvicorn main:app --reload"
echo ""
