@echo off
REM ============================================================
REM  Audiobook beta server - STOP
REM  Kills ONLY the process listening on port 8000 (nothing else).
REM ============================================================
echo Stopping the audiobook server on port 8000 ...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /c:":8000" ^| findstr /c:"LISTENING"') do (
  echo   stopping PID %%p
  taskkill /PID %%p /F >nul 2>&1
)
echo Done. If a server window is still open, close it too.
