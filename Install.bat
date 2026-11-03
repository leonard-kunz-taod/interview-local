@echo off
cd /d "%~dp0"
echo Interview Local: Installation requires internet access and Python 3.
py -3 -m venv .installer
if errorlevel 1 goto error
.installer\Scripts\python.exe -m pip install "uv>=0.7.8,<1"
if errorlevel 1 goto error
.installer\Scripts\uv.exe sync --frozen --python 3.12
if errorlevel 1 goto error
echo For readable drafts, install and open Ollama: https://ollama.com/download
echo Then run Start.bat --prepare-llm once. No Hugging Face login is needed for Ministral.
echo Done. Open Start.bat next.
pause
exit /b 0
:error
echo Installation failed. Check the message above and README.md.
pause
exit /b 1
