@echo off
REM Optional: package the Python backend as backend\sMeetSec-server.exe (no Python needed on other PCs).
REM sMeetSec.exe uses it automatically when it exists. Models are NOT bundled (models\ folder stays next to sMeetSec.exe).
cd /d "%~dp0"
if exist runtime\python\python.exe (set PY=runtime\python\python.exe) else (call .venv\Scripts\activate.bat & set PY=python)
%PY% -m pip install pyinstaller
%PY% -m PyInstaller --noconfirm --name sMeetSec-server --noconsole --icon logo\icon.ico ^
  --add-data "smeetsec\static;smeetsec\static" --add-data "legal;legal" ^
  --collect-all faster_whisper --collect-all ctranslate2 --collect-all av ^
  --collect-all sherpa_onnx --collect-all onnxruntime ^
  --collect-submodules uvicorn --hidden-import docx ^
  run.py
if errorlevel 1 ( echo Build failed. & pause & exit /b 1 )
xcopy /E /I /Y dist\sMeetSec-server backend >nul
echo Built: backend\sMeetSec-server.exe  (start the app with sMeetSec.exe)
pause
