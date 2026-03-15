@echo off
echo ============================================================
echo NitroGen Performance Benchmark
echo ============================================================
echo.
echo This measures inference speed without running a game.
echo.
echo Make sure the server is running first!
echo.
pause
cd /d "%~dp0..\.."
venv\Scripts\python.exe tools\debug\benchmark.py --iterations 100
pause
