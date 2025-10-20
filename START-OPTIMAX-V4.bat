@echo off
title OptimaX v4.0 - System Launcher
color 0A

echo.
echo ========================================
echo   OptimaX v4.0 - System Launcher
echo ========================================
echo.
echo Starting OptimaX SQL Chat Application...
echo.

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

REM Start Backend
echo [1/2] Starting Backend Server (Port 8000)...
echo.
start "OptimaX Backend v4.0" cmd /k "cd sql-chat-backend && py main.py"

REM Wait for backend to initialize
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start Frontend
echo.
echo [2/2] Starting Frontend Server (Port 4200)...
echo.
start "OptimaX Frontend" cmd /k "cd sql-chat-app && npm start"

REM Wait a bit more
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo   OptimaX v4.0 Started Successfully!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:4200
echo.
echo Two terminal windows have been opened:
echo   - Backend (Python/FastAPI)
echo   - Frontend (Angular)
echo.
echo The application will open in your browser shortly...
echo.
echo To stop the application:
echo   - Close both terminal windows
echo   - Or press Ctrl+C in each window
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
