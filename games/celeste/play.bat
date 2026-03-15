@echo off
echo ============================================================
echo NitroGen - Celeste
echo ============================================================
echo.
echo Make sure:
echo   1. Server is running: start_server.bat (from project root)
echo   2. Celeste is running
echo   3. You're in gameplay (not menus)
echo.
echo Controls:
echo   F1: Pause/Resume AI
echo   F2: Exit
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\play_fast.py --process "Celeste.exe" --actions-per-chunk 4
