@echo off
title OptimaX v4.0 - Stop Services
color 0C

echo.
echo ========================================
echo   OptimaX v4.0 - Stop Services
echo ========================================
echo.

echo Stopping OptimaX services...
echo.

REM Kill Python processes (backend)
echo Stopping Backend (Python)...
taskkill /FI "WindowTitle eq OptimaX Backend v4.0*" /T /F >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Backend stopped
) else (
    echo   [INFO] Backend was not running
)

REM Kill Node processes (frontend)
echo Stopping Frontend (Node)...
taskkill /FI "WindowTitle eq OptimaX Frontend*" /T /F >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Frontend stopped
) else (
    echo   [INFO] Frontend was not running
)

echo.
echo ========================================
echo   All OptimaX services stopped!
echo ========================================
echo.
pause
