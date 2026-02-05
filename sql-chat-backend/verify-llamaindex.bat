@echo off
title LlamaIndex NL-SQL Verification
color 0B

cd /d "%~dp0"

echo.
echo ========================================
echo   LlamaIndex NL-SQL Verification
echo ========================================
echo.

REM Find Python interpreter
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    echo [ERROR] Virtual environment not found!
    pause
    exit /b 1
)

echo Using: %PYTHON_EXE%
echo.

REM Parse arguments
set "ARGS="
if "%1"=="--smoke-test" set "ARGS=--smoke-test"
if "%1"=="-s" set "ARGS=--smoke-test"
if "%2"=="--verbose" set "ARGS=%ARGS% --verbose"
if "%2"=="-v" set "ARGS=%ARGS% --verbose"
if "%1"=="--verbose" set "ARGS=--verbose"
if "%1"=="-v" set "ARGS=--verbose"

"%PYTHON_EXE%" verify_llamaindex.py %ARGS%

echo.
pause
