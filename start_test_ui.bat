@echo off
setlocal
cd /d "%~dp0"

echo.
echo   =========================================================
echo    Quant SQL Test Interface  ^|  http://localhost:8001
echo   =========================================================
echo.

:: ── 1. Validate venv ──────────────────────────────────────────────────────────
if not exist "sql-chat-backend\venv\Scripts\activate.bat" (
    echo   [ERROR] venv not found at sql-chat-backend\venv\Scripts\activate.bat
    echo.
    echo   Set it up with:
    echo     cd sql-chat-backend
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── 2. Validate server script ─────────────────────────────────────────────────
if not exist "experiments\quant_sql\test_server.py" (
    echo   [ERROR] experiments\quant_sql\test_server.py not found
    pause
    exit /b 1
)

:: ── 3. Quick model check (informational only) ─────────────────────────────────
set "MODELS=%USERPROFILE%\Desktop\Quant-SQL-Experiment\models"
if exist "%MODELS%\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" (
    echo   [OK]  Q4_K_M model found
) else (
    echo   [--]  Q4_K_M not found  (local Q4 inference unavailable)
)
if exist "%MODELS%\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf" (
    echo   [OK]  Q8_0 model found
) else (
    echo   [--]  Q8_0 not found  (local Q8 inference unavailable)
)

:: ── 4. Port conflict check ────────────────────────────────────────────────────
netstat -an 2>nul | find ":8001 " | find "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [WARN] Port 8001 is already in use.
    echo          Close the existing server window first.
    echo.
    choice /c YN /m "   Launch anyway? [Y/N]"
    if errorlevel 2 exit /b 0
)

:: ── 5. Write a self-contained helper script for the server window ─────────────
::   This avoids all quote-nesting / line-continuation issues with start+cmd /k.
set "HELPER=%TEMP%\quant_sql_server.bat"
> "%HELPER%"  echo @echo off
>> "%HELPER%" echo title Quant SQL Test Server [port 8001]
>> "%HELPER%" echo cd /d "%CD%"
>> "%HELPER%" echo call "sql-chat-backend\venv\Scripts\activate.bat"
>> "%HELPER%" echo echo.
>> "%HELPER%" echo echo   Starting uvicorn on http://localhost:8001 ...
>> "%HELPER%" echo echo.
>> "%HELPER%" echo python "experiments\quant_sql\test_server.py"
>> "%HELPER%" echo echo.
>> "%HELPER%" echo echo   Server stopped.  Press any key to close.
>> "%HELPER%" echo pause ^>nul

echo.
echo   Launching server window...
start "Quant SQL Test Server [port 8001]" cmd /k "%HELPER%"

:: ── 6. Wait for uvicorn to bind, then open the browser ───────────────────────
echo   Waiting for server to start...
timeout /t 4 /nobreak >nul
start "" http://localhost:8001

echo   [OK]  Opened http://localhost:8001
echo.
echo   Close the "Quant SQL Test Server" window to stop.
echo.
timeout /t 5 /nobreak >nul
