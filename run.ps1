# AuraShop Bot Startup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting AuraShop Chatbot & Defect Detection Portal" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Python is in Path
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not found in your system PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9+ and add it to your PATH." -ForegroundColor Yellow
    Exit 1
}

# 2. Setup Virtual Environment
$venvPath = Join-Path $PSScriptRoot "venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Python Virtual Environment in $venvPath..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 3. Activate venv
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& $activateScript

# 4. Install Dependencies
Write-Host "Checking and installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host "Dependencies successfully verified." -ForegroundColor Green

# 5. Launch FastAPI via Uvicorn
Write-Host "Starting development server at http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to terminate the server." -ForegroundColor Yellow
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
