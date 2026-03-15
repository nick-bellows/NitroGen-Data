@echo off
echo ============================================================
echo NitroGen - Celeste (Original Script)
echo ============================================================
echo.
echo Using the original play.py script (slower but more compatible).
echo.
echo Make sure:
echo   1. Server is running
echo   2. Celeste is running
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\play.py --process "Celeste.exe"
