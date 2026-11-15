@echo off
REM sMeetSec - one-time setup (needs internet ONCE). sMeetSec.exe runs the same setup automatically on first start.
REM Installs a private Python inside runtime\ (nothing system-wide), packages and models.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup.ps1"
if errorlevel 1 ( echo. & echo Setup failed - see the messages above. & pause & exit /b 1 )
echo.
echo ============================================================
echo  Setup done.
echo  For translation and reports, install Ollama (https://ollama.com)
echo  then run ONE of these in a terminal:
echo     ollama pull qwen3.5:9b      (16 GB RAM or more)
echo     ollama pull qwen3.5:4b      (8 GB RAM)
echo  Start sMeetSec with sMeetSec.exe (or run.bat for the browser)
echo ============================================================
pause
