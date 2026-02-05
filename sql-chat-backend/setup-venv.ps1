# ============================================================================
# OptimaX - Virtual Environment Setup (Windows PowerShell)
# ============================================================================
# This script creates a deterministic Python virtual environment.
# Run from PowerShell: .\setup-venv.ps1
# ============================================================================

param(
    [switch]$Force,  # Force recreation of existing venv
    [switch]$NoInstall  # Skip package installation
)

$ErrorActionPreference = "Stop"
$VenvName = ".venv"
$VenvPath = Join-Path $PSScriptRoot $VenvName

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OptimaX Virtual Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green

    # Verify Python 3.9+
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
            Write-Host "  ERROR: Python 3.9+ required, found $major.$minor" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "  ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.9+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Step 2: Check for existing venv
Write-Host ""
Write-Host "[2/5] Checking for existing virtual environment..." -ForegroundColor Yellow

if (Test-Path $VenvPath) {
    if ($Force) {
        Write-Host "  Removing existing venv (--Force specified)..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $VenvPath
    } else {
        Write-Host "  Found existing venv at: $VenvPath" -ForegroundColor Yellow
        Write-Host "  Use -Force to recreate, or skip to step 4" -ForegroundColor Yellow

        $response = Read-Host "  Continue with existing venv? (y/n)"
        if ($response -ne "y") {
            Write-Host "  Removing existing venv..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $VenvPath
        }
    }
}

# Step 3: Create virtual environment
Write-Host ""
Write-Host "[3/5] Creating virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path $VenvPath)) {
    Write-Host "  Creating $VenvName at $VenvPath" -ForegroundColor Cyan
    python -m venv $VenvPath

    if (-not (Test-Path $VenvPath)) {
        Write-Host "  ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created successfully" -ForegroundColor Green
} else {
    Write-Host "  Using existing virtual environment" -ForegroundColor Green
}

# Step 4: Activate and upgrade pip
Write-Host ""
Write-Host "[4/5] Activating environment and upgrading pip..." -ForegroundColor Yellow

$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Host "  ERROR: Activation script not found at $ActivateScript" -ForegroundColor Red
    exit 1
}

# Source the activation script
. $ActivateScript
Write-Host "  Activated: $VenvName" -ForegroundColor Green

# Upgrade pip
Write-Host "  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Step 5: Install dependencies
Write-Host ""
Write-Host "[5/5] Installing dependencies..." -ForegroundColor Yellow

if (-not $NoInstall) {
    $RequirementsPath = Join-Path $PSScriptRoot "requirements.txt"
    if (Test-Path $RequirementsPath) {
        Write-Host "  Installing from requirements.txt..." -ForegroundColor Cyan
        pip install -r $RequirementsPath

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: Package installation failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "  All packages installed successfully" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: requirements.txt not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Skipping package installation (--NoInstall)" -ForegroundColor Yellow
}

# Verify LlamaIndex installation
Write-Host ""
Write-Host "Verifying LlamaIndex installation..." -ForegroundColor Yellow
python -c "from llama_index.core.query_engine import NLSQLTableQueryEngine; print('  NLSQLTableQueryEngine: OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: LlamaIndex NL-SQL not properly installed" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Virtual environment: $VenvPath" -ForegroundColor Cyan
Write-Host "Python interpreter:  $VenvPath\Scripts\python.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate in PowerShell:" -ForegroundColor Yellow
Write-Host "  . .\$VenvName\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To activate in CMD:" -ForegroundColor Yellow
Write-Host "  $VenvName\Scripts\activate.bat" -ForegroundColor White
Write-Host ""
Write-Host "To run the backend:" -ForegroundColor Yellow
Write-Host "  python -m uvicorn main:app --reload" -ForegroundColor White
Write-Host ""
