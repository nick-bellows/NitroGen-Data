@echo off
echo ============================================================
echo   NitroGen AI Player - Hades
echo ============================================================
echo.
echo   Make sure:
echo     1. Server is running: start_server.bat (from project root)
echo     2. Hades is running
echo.
echo   Controls:
echo     F1 = Pause/Resume AI
echo     F2 = Exit
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\play_fast.py --process "Hades.exe" --actions-per-chunk 4
