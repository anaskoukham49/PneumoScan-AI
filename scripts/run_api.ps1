$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$python = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found. Run: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1"
}

Write-Host "Starting API at http://127.0.0.1:8000 (Ctrl+C to stop)" -ForegroundColor Green
& $python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
