@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -m scripts.complete_improvement
pause
