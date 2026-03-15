@echo off
cd /d "%~dp0..\.."
echo ============================================================
echo   DAgger Collection - Hades
echo ============================================================
echo.
echo This lets the AI play while you correct its mistakes.
echo.
echo   Grab controller = You control (H indicator)
echo   Release for 0.5s = AI resumes (. indicator)
echo   Ctrl+C = End session and save
echo.
echo Requirements:
echo   - AI server must be running (start_server_multiframe.bat)
echo   - Hades should be running
echo   - ViGEmBus driver installed
echo.
echo ============================================================
echo.
echo Checking dependencies...
venv\Scripts\pip show vgamepad >nul 2>&1
if errorlevel 1 (
    echo Installing vgamepad...
    venv\Scripts\pip install vgamepad
)
venv\Scripts\pip show XInput-Python >nul 2>&1
if errorlevel 1 (
    echo Installing XInput-Python...
    venv\Scripts\pip install XInput-Python
)
venv\Scripts\pip show mss >nul 2>&1
if errorlevel 1 (
    echo Installing mss...
    venv\Scripts\pip install mss
)
echo.
echo Make sure:
echo   1. AI server is running (start_server_multiframe.bat)
echo   2. Hades is running and in focus
echo.
pause
echo.
venv\Scripts\python scripts\dagger_collect.py --process "Hades.exe"
echo.
pause
