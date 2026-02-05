@echo off
title OptimaX - Migrate venv to .venv
color 0E

echo.
echo ========================================
echo   OptimaX Virtual Environment Migration
echo ========================================
echo.
echo This script migrates from 'venv' to '.venv' for VS Code alignment.
echo.

REM Change to backend directory
cd /d "%~dp0"

REM Check if old venv exists
if not exist "venv" (
    echo [INFO] No legacy 'venv' directory found.
    echo.
    if exist ".venv" (
        echo [OK] Already using '.venv' - no migration needed.
    ) else (
        echo [ACTION] Run setup-venv.ps1 to create a new .venv
    )
    pause
    exit /b 0
)

REM Check if new .venv already exists
if exist ".venv" (
    echo [ERROR] Both 'venv' and '.venv' exist!
    echo.
    echo Please manually decide which to keep:
    echo   - Delete venv\: rd /s /q venv
    echo   - Delete .venv\: rd /s /q .venv
    echo.
    pause
    exit /b 1
)

echo [INFO] Found legacy 'venv' directory
echo [INFO] Will rename to '.venv'
echo.

REM Confirm with user
set /p confirm="Proceed with migration? (y/n): "
if /i not "%confirm%"=="y" (
    echo Migration cancelled.
    pause
    exit /b 0
)

echo.
echo [1/3] Deactivating current environment...
call venv\Scripts\deactivate.bat 2>nul

echo [2/3] Renaming venv to .venv...
ren venv .venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to rename directory!
    echo [ERROR] Close any terminals using venv and try again.
    pause
    exit /b 1
)

echo [3/3] Updating activation scripts...
REM The paths in pyvenv.cfg are absolute, so no changes needed

echo.
echo ========================================
echo   Migration Complete!
echo ========================================
echo.
echo Old location: venv\
echo New location: .venv\
echo.
echo VS Code should now automatically detect the correct interpreter.
echo If not, reload VS Code (Ctrl+Shift+P -^> "Developer: Reload Window")
echo.
echo To activate in CMD:
echo   .venv\Scripts\activate
echo.
pause
