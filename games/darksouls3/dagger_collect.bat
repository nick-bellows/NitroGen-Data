@echo off
echo ============================================================
echo DAgger Data Collection - Dark Souls III
echo ============================================================
echo.
echo This lets AI play while recording your controller corrections.
echo.
echo Controls:
echo   - Your Xbox controller overrides AI when you use it
echo   - F1: Toggle full manual mode (you play entirely)
echo   - F2: Save and exit
echo.
echo Make sure:
echo   1. Server is running: start_server.bat (from project root)
echo   2. Dark Souls III is running
echo   3. Your real Xbox controller is connected
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\dagger_collect.py --process "DarkSoulsIII.exe" --output-dir "games\darksouls3\dagger_data"
