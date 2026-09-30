@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m src.evaluation.external_kneemri
pause
