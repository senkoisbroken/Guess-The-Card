@echo off
setlocal
if exist "%~dp0.venv\Scripts\pythonw.exe" goto :update
echo Run start.bat to finish setup before updating.
pause
exit /b 1
:update
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0updater.py" %*
