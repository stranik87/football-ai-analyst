@echo off
chcp 65001 > nul
cd /d "D:\analyses\football-ai-analyst"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

"D:\analyses\venv\Scripts\python.exe" -m scripts.update_pipeline >> "data\reports\pipeline.log" 2>&1

exit /b %ERRORLEVEL%
