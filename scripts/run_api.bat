@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run scripts\setup.ps1 first.
    exit /b 1
)
echo Starting API at http://127.0.0.1:8000 (Ctrl+C to stop)
".venv\Scripts\python.exe" -m uvicorn api.app:app --host 127.0.0.1 --port 8000
