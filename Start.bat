@echo off
cd /d "%~dp0"
if exist .cache\models set "HF_HUB_CACHE=%CD%\.cache\models"
if exist .cache\nltk_data set "NLTK_DATA=%CD%\.cache\nltk_data"
if not exist .venv\Scripts\python.exe (
  echo Run Install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m interview_local %*
pause
