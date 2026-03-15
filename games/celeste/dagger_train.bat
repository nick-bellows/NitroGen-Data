@echo off
echo ============================================================
echo DAgger Training - Celeste
echo ============================================================
echo.
echo Available data directories:
echo.
dir /b "games\celeste\dagger_data" 2>nul || echo   (No data collected yet)
echo.
set /p DATA_DIR="Enter data directory name: "
echo.
echo Training with data from: games\celeste\dagger_data\%DATA_DIR%
echo.
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\dagger_train.py --data-dir "games\celeste\dagger_data\%DATA_DIR%" --output-dir "games\celeste\checkpoints" --epochs 10 --batch-size 4 --lr 1e-5
pause
