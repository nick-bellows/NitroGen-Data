@echo off
echo ============================================================
echo   NitroGen Gameplay Recorder
echo ============================================================
echo.
echo   This records YOUR gameplay for training (no AI involved).
echo   Pure Behavior Cloning data collection.
echo.
echo   Common games:
echo     - Hades.exe
echo     - DarkSoulsIII.exe
echo     - Celeste.exe
echo     - HollowKnight.exe
echo.
echo   For game-specific recorders, check games/ folder.
echo.
set /p GAME="Enter game executable name: "
echo.
set /p FPS="Capture FPS (default 30): "
if "%FPS%"=="" set FPS=30
echo.
echo   Controls:
echo     F5 = Start/Stop Recording
echo     F6 = Quit
echo.
pause
venv\Scripts\python.exe scripts\record_gameplay.py --process "%GAME%" --fps %FPS%
pause
