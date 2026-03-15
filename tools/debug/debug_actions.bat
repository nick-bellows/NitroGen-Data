@echo off
echo ============================================================
echo Debug NitroGen Actions
echo ============================================================
echo.
echo This captures frames and shows what the model is predicting.
echo Useful for understanding model behavior.
echo.
echo Make sure:
echo   1. Server is running
echo   2. Your game is running
echo.
set /p GAME="Enter game executable (e.g., DarkSoulsIII.exe): "
set /p FRAMES="Number of frames to capture (default 10): "
if "%FRAMES%"=="" set FRAMES=10
echo.
cd /d "%~dp0..\.."
venv\Scripts\python.exe tools\debug\debug_actions.py --process "%GAME%" --frames %FRAMES%
pause
