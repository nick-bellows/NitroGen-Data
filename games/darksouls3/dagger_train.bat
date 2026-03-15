@echo off
echo ============================================================
echo DAgger Training - Dark Souls III
echo ============================================================
echo.
echo Available data directories:
echo.
dir /b "games\darksouls3\dagger_data" 2>nul || echo   (No data collected yet)
echo.
set /p DATA_DIR="Enter data directory name: "
echo.
echo Training with data from: games\darksouls3\dagger_data\%DATA_DIR%
echo.
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\dagger_train.py --data-dir "games\darksouls3\dagger_data\%DATA_DIR%" --output-dir "games\darksouls3\checkpoints" --epochs 10 --batch-size 4 --lr 1e-5
pause
