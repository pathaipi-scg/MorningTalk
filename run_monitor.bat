@echo off
setlocal
cd /d "D:\AI\MorningTalk"
".venv\Scripts\python.exe" -m app.main monitor --interval 300 --sync-db
endlocal
