@echo off
REM ============================================================
REM  Beta tester tokens - shows name:token for each tester.
REM  KEEP PRIVATE. Share each client ONLY their own token -
REM  usage limits are tracked per token.
REM ============================================================
cd /d "%~dp0.."
for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b "BETA_ACCESS_TOKENS=" .env`) do set "line=%%b"
for %%t in (%line:,= %) do echo   %%t
echo.
echo (Copy the line for each client - never post these publicly.)
