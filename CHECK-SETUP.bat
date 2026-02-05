@echo off
title OptimaX v6.1 - Environment Verification
color 0B

echo.
echo ========================================
echo   OptimaX v6.1 - Environment Verification
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

set ERRORS=0
set WARNINGS=0

REM Check Python
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version') do echo   [OK] %%i
) else (
    echo   [ERROR] Python not found! Please install Python 3.9+
    set /a ERRORS+=1
)
echo.

REM Check Node.js
echo [2/8] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('node --version') do echo   [OK] Node.js %%i
) else (
    echo   [ERROR] Node.js not found! Please install Node.js 18+
    set /a ERRORS+=1
)
echo.

REM Check npm
echo [3/8] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('npm --version') do echo   [OK] npm %%i
) else (
    echo   [ERROR] npm not found!
    set /a ERRORS+=1
)
echo.

REM Check virtual environment
echo [4/8] Checking Python virtual environment...
if exist "sql-chat-backend\.venv\Scripts\python.exe" (
    echo   [OK] .venv found (preferred location)
    set "PYTHON_EXE=sql-chat-backend\.venv\Scripts\python.exe"
) else if exist "sql-chat-backend\venv\Scripts\python.exe" (
    echo   [WARN] Legacy 'venv' found - consider migrating to '.venv'
    echo   [ACTION] Run: sql-chat-backend\migrate-venv.bat
    set "PYTHON_EXE=sql-chat-backend\venv\Scripts\python.exe"
    set /a WARNINGS+=1
) else (
    echo   [ERROR] Virtual environment not found!
    echo   [ACTION] Run: cd sql-chat-backend ^&^& python -m venv .venv
    set /a ERRORS+=1
    set "PYTHON_EXE="
)
echo.

REM Check .env file
echo [5/8] Checking backend configuration...
if exist "sql-chat-backend\.env" (
    echo   [OK] .env file exists
    findstr /C:"GROQ_API_KEY" sql-chat-backend\.env >nul 2>&1
    if %errorlevel% equ 0 (
        echo   [OK] GROQ_API_KEY found in .env
    ) else (
        echo   [ERROR] GROQ_API_KEY not found in .env
        set /a ERRORS+=1
    )
    findstr /C:"DATABASE_URL" sql-chat-backend\.env >nul 2>&1
    if %errorlevel% equ 0 (
        echo   [OK] DATABASE_URL found in .env
    ) else (
        echo   [ERROR] DATABASE_URL not found in .env
        set /a ERRORS+=1
    )
) else (
    echo   [ERROR] .env file not found!
    echo   [ACTION] Copy .env.example to .env and configure
    set /a ERRORS+=1
)
echo.

REM Check LlamaIndex installation (if venv exists)
echo [6/8] Checking LlamaIndex installation...
if defined PYTHON_EXE (
    "%PYTHON_EXE%" -c "from llama_index.core.query_engine import NLSQLTableQueryEngine; print('  [OK] NLSQLTableQueryEngine available')" 2>&1
    if %errorlevel% neq 0 (
        echo   [ERROR] LlamaIndex NL-SQL not installed!
        echo   [ACTION] Activate venv and run: pip install -r requirements.txt
        set /a ERRORS+=1
    )
) else (
    echo   [SKIP] Cannot check - no virtual environment
)
echo.

REM Check requirements.txt
echo [7/8] Checking backend dependencies file...
if exist "sql-chat-backend\requirements.txt" (
    echo   [OK] requirements.txt found
) else (
    echo   [ERROR] requirements.txt not found!
    set /a ERRORS+=1
)
echo.

REM Check frontend dependencies
echo [8/8] Checking frontend dependencies...
if exist "sql-chat-app\package.json" (
    echo   [OK] package.json found
    if exist "sql-chat-app\node_modules" (
        echo   [OK] node_modules folder exists
    ) else (
        echo   [WARN] node_modules not found
        echo   [ACTION] Run: cd sql-chat-app ^&^& npm install
        set /a WARNINGS+=1
    )
) else (
    echo   [ERROR] package.json not found!
    set /a ERRORS+=1
)
echo.

REM VS Code configuration check
echo ----------------------------------------
echo VS Code Configuration:
echo ----------------------------------------
if exist ".vscode\settings.json" (
    echo   [OK] .vscode\settings.json exists
) else (
    echo   [WARN] .vscode\settings.json not found
    set /a WARNINGS+=1
)
if exist ".vscode\launch.json" (
    echo   [OK] .vscode\launch.json exists
) else (
    echo   [WARN] .vscode\launch.json not found
    set /a WARNINGS+=1
)
echo.

REM Summary
echo ========================================
if %ERRORS% gtr 0 (
    color 0C
    echo   SETUP CHECK FAILED
    echo   Errors: %ERRORS%, Warnings: %WARNINGS%
) else if %WARNINGS% gtr 0 (
    color 0E
    echo   SETUP CHECK PASSED WITH WARNINGS
    echo   Warnings: %WARNINGS%
) else (
    color 0A
    echo   SETUP CHECK PASSED
    echo   All checks passed!
)
echo ========================================
echo.

if %ERRORS% equ 0 (
    echo Ready to run: START-OPTIMAX.bat
) else (
    echo Fix the errors above before running OptimaX.
)
echo.

pause
