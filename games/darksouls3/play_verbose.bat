@echo off
echo ============================================================
echo NitroGen - Dark Souls III (Verbose Mode)
echo ============================================================
echo.
echo This mode shows detailed joystick values for debugging.
echo.
echo Make sure:
echo   1. Server is running: start_server.bat (from project root)
echo   2. Dark Souls III is running
echo   3. You're in-game (not in menus)
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\play_fast.py --process "DarkSoulsIII.exe" --actions-per-chunk 4 --verbose
