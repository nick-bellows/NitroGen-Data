@echo off
echo ============================================================
echo NitroGen - Play Any Game
echo ============================================================
echo.
echo Make sure:
echo   1. The inference server is running (start_server.bat)
echo   2. Your game is running
echo.
echo Common game executables:
echo   - DarkSoulsIII.exe
echo   - eldenring.exe
echo   - Celeste.exe
echo   - sekiro.exe
echo   - HollowKnight.exe
echo.
echo For game-specific configs, check the games/ folder:
echo   - games\darksouls3\
echo   - games\celeste\
echo.
set /p GAME="Enter game executable name: "
echo.
echo Starting NitroGen on %GAME%...
echo.
echo Controls:
echo   F1: Pause/Resume AI
echo   F2: Exit
echo.
venv\Scripts\python.exe scripts\play_fast.py --process "%GAME%" --actions-per-chunk 4
