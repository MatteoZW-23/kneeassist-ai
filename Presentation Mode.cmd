@echo off
cd /d "%~dp0"

echo ============================================================
echo KneeAssist AI - Presentation Mode
echo Starts the dashboard locally for live demos and presentations.
echo ============================================================

a) 
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: The Python environment was not found.
  echo Run Install.cmd first, then try again.
  pause
  exit /b 1
)

echo Opening the app in the default browser once it is ready...
".venv\Scripts\python.exe" launch.py
if errorlevel 1 pause
