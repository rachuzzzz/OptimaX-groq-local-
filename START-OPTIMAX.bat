@echo off
title OptimaX v6.1 - System Launcher
color 0A

echo.
echo ========================================
echo   OptimaX v6.1 - System Launcher
echo ========================================
echo.
echo Starting OptimaX SQL Chat Application...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if .env exists
if not exist "sql-chat-backend\.env" (
    echo [ERROR] .env file not found in sql-chat-backend folder!
    echo.
    echo Please create sql-chat-backend\.env with:
    echo   GROQ_API_KEY=your_groq_api_key_here
    echo   DATABASE_URL=postgresql://user:pass@localhost:5432/traffic_db
    echo.
    pause
    exit /b 1
)

REM Check for .venv or venv
set "VENV_NAME="
if exist "sql-chat-backend\.venv\Scripts\python.exe" (
    set "VENV_NAME=.venv"
)
if exist "sql-chat-backend\venv\Scripts\python.exe" (
    if "%VENV_NAME%"=="" set "VENV_NAME=venv"
)

if "%VENV_NAME%"=="" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Expected locations:
    echo   - sql-chat-backend\.venv\Scripts\python.exe
    echo   - sql-chat-backend\venv\Scripts\python.exe
    echo.
    echo To create the virtual environment:
    echo   cd sql-chat-backend
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [OK] Using venv: sql-chat-backend\%VENV_NAME%
echo.

REM Start Backend with uvicorn
echo [1/2] Starting Backend Server - Port 8000
echo.
start "OptimaX Backend v6.1" cmd /k "cd /d %~dp0sql-chat-backend && set PYTHONUNBUFFERED=1 && %VENV_NAME%\Scripts\python.exe -u -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info"

REM Wait for backend to initialize
echo Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

REM Start Frontend
echo.
echo [2/2] Starting Frontend Server - Port 4200
echo.
start "OptimaX Frontend" cmd /k "cd /d %~dp0sql-chat-app && npm start"

REM Wait a bit more
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   OptimaX v6.1 Started Successfully!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:4200
echo.
echo Two terminal windows have been opened:
echo   - Backend - Python/FastAPI/Uvicorn
echo   - Frontend - Angular
echo.
echo To stop: Close both terminal windows or Ctrl+C
echo.
echo ========================================
echo.

REM Wait for frontend to start and open browser
timeout /t 15 /nobreak >nul
start http://localhost:4200

echo Application opened in browser!
echo.
echo You can now close this window.
echo The backend and frontend will continue running.
echo.
pause
