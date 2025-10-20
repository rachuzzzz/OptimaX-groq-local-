@echo off
title OptimaX v4.0 - Setup Checker
color 0B

echo.
echo ========================================
echo   OptimaX v4.0 - Setup Checker
echo ========================================
echo.

REM Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Python is installed
    python --version
) else (
    echo   [ERROR] Python not found! Please install Python 3.9+
)
echo.

REM Check Node.js
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Node.js is installed
    node --version
) else (
    echo   [ERROR] Node.js not found! Please install Node.js 18+
)
echo.

REM Check npm
echo [3/6] Checking npm installation...
npm --version >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] npm is installed
    npm --version
) else (
    echo   [ERROR] npm not found!
)
echo.

REM Check .env file
echo [4/6] Checking backend configuration...
if exist "sql-chat-backend\.env" (
    echo   [OK] .env file exists
    echo.
    echo   Checking environment variables:
    findstr /C:"GROQ_API_KEY" sql-chat-backend\.env >nul 2>&1
    if %errorlevel% equ 0 (
        echo   [OK] GROQ_API_KEY found in .env
    ) else (
        echo   [WARNING] GROQ_API_KEY not found in .env
    )
    findstr /C:"DATABASE_URL" sql-chat-backend\.env >nul 2>&1
    if %errorlevel% equ 0 (
        echo   [OK] DATABASE_URL found in .env
    ) else (
        echo   [WARNING] DATABASE_URL not found in .env
    )
) else (
    echo   [ERROR] .env file not found!
    echo.
    echo   Please create sql-chat-backend\.env with:
    echo     GROQ_API_KEY=your_groq_api_key_here
    echo     DATABASE_URL=postgresql://user:pass@localhost:5432/traffic_db
)
echo.

REM Check backend dependencies
echo [5/6] Checking backend dependencies...
if exist "sql-chat-backend\requirements.txt" (
    echo   [OK] requirements.txt found
) else (
    echo   [ERROR] requirements.txt not found!
)
echo.

REM Check frontend dependencies
echo [6/6] Checking frontend dependencies...
if exist "sql-chat-app\package.json" (
    echo   [OK] package.json found
    if exist "sql-chat-app\node_modules" (
        echo   [OK] node_modules folder exists (dependencies installed)
    ) else (
        echo   [WARNING] node_modules not found
        echo   [ACTION] Run: cd sql-chat-app && npm install
    )
) else (
    echo   [ERROR] package.json not found!
)
echo.

echo ========================================
echo   Setup Check Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Ensure .env file is configured
echo   2. Install backend dependencies: cd sql-chat-backend ^&^& pip install -r requirements.txt
echo   3. Install frontend dependencies: cd sql-chat-app ^&^& npm install
echo   4. Run START-OPTIMAX-V4.bat to launch the application
echo.
pause
