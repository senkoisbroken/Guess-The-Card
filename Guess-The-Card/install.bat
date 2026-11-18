@echo off
setlocal
set "CARD_SETUP_ONLY=1"
call "%~dp0start.bat"
exit /b %errorlevel%
