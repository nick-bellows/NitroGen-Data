@echo off
echo ============================================================
echo Virtual Controller Test
echo ============================================================
echo.
echo This tests if the virtual Xbox controller works.
echo.
echo IMPORTANT: Unplug your REAL controller first!
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe tools\debug\test_controller.py
