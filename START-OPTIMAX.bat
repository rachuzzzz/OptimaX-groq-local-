@echo off
title OptimaX Startup
color 0A

echo ========================================
echo   OptimaX - SQL Chat Application
echo   DUAL-MODEL Architecture
echo ========================================
echo.
echo [1/5] Killing old processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM py.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/5] Clearing Python cache...
cd sql-chat-backend
if exist __pycache__ rd /s /q __pycache__ >nul 2>&1
del /s /q *.pyc >nul 2>&1
cd ..

echo [3/5] Starting Backend (Port 8003)...
start "OptimaX Backend" cmd /k "cd sql-chat-backend && venv\Scripts\python.exe main_agentic.py"
timeout /t 5 /nobreak >nul

echo [4/5] Starting Frontend (Port 4200)...
start "OptimaX Frontend" cmd /k "cd sql-chat-app && npm start"
timeout /t 3 /nobreak >nul

echo [5/5] Opening browser...
timeout /t 8 /nobreak >nul
start http://localhost:4200

echo.
echo ========================================
echo   OptimaX is starting!
echo ========================================
echo.
echo Backend:  http://localhost:8003
echo Frontend: http://localhost:4200
echo.
echo Architecture:
echo   - Agent: Groq Llama-3.3-70B (cloud)
echo   - SQL Tool: Ollama Qwen 2.5-Coder 3B (local)
echo.
echo Check the other windows for logs.
echo.
pause
