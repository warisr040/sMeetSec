@echo off
cd /d "%~dp0"
if exist runtime\python\python.exe (
  runtime\python\python.exe run.py
  exit /b
)
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe run.py
  exit /b
)
echo Please start sMeetSec.exe once (it sets everything up), or run setup_windows.bat.
pause
