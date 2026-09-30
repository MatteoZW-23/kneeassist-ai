@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -m ipykernel install --prefix .venv --name kneeassist-ai --display-name "KneeAssist AI (project .venv)"
if errorlevel 1 goto failure
echo Installation complete. Double-click Launch KneeAssist AI.cmd.
pause
exit /b 0
:failure
echo Installation failed. Check the error above. Python 3.14 and internet access are required.
pause
exit /b 1
