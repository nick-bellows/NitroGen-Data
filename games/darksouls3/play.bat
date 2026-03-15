@echo off
echo ============================================================
echo NitroGen - Dark Souls III
echo ============================================================
echo.
echo Make sure:
echo   1. Server is running: start_server.bat (from project root)
echo   2. Dark Souls III is running (Borderless Windowed recommended)
echo   3. You're loaded into the game (past menus)
echo   4. Unplug your real controller!
echo.
echo Controls:
echo   F1: Pause/Resume AI (you can play when paused)
echo   F2: Exit
echo.
echo Deadzone compensation: ENABLED (amplify=2.5x, min=0.35)
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\play_fast.py --process "DarkSoulsIII.exe" --actions-per-chunk 4
