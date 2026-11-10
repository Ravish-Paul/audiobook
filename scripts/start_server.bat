@echo off
REM ============================================================
REM  Audiobook beta server - START (localhost only)
REM
REM  Binds to 127.0.0.1:8000. Share it through your Cloudflare
REM  tunnel pointing at http://localhost:8000 - nothing on the
REM  network can reach it directly (safer).
REM
REM  Keep this window open while the server should run.
REM  Stop = close this window (or run scripts\stop_server.bat).
REM ============================================================
cd /d "%~dp0.."

echo.
echo ============================================================
echo   Audiobook beta server starting on http://localhost:8000
echo ============================================================
echo.
echo   Point your Cloudflare tunnel at:  http://localhost:8000
echo   Clients use the tunnel URL and log in with their beta token
echo   (tokens: run scripts\show_tokens.bat - share each client ONLY their own).
echo.
echo   Local test in this PC's browser:  http://127.0.0.1:8000
echo.
echo   STOP the server: close this window (or run scripts\stop_server.bat).
echo ============================================================
echo.

".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
