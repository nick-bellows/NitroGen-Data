@echo off
echo ============================================================
echo   NitroGen Gameplay Recorder - Hades
echo ============================================================
echo.
echo   This records YOUR gameplay for training (no AI involved).
echo   Pure Behavior Cloning data collection at 30 FPS.
echo.
echo   Make sure Hades is running!
echo.
echo   Controls:
echo     F5 = Start/Stop Recording
echo     F6 = Quit
echo.
echo   Tips:
echo     - Enable God Mode in Hades settings for easier gameplay
echo     - Play naturally, don't try to be perfect
echo     - Record multiple runs for variety
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\record_gameplay.py --process "Hades.exe" --fps 30 --output-dir "games\hades\recordings"
pause
