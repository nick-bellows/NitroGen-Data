@echo off
echo ============================================================
echo   NitroGen Training - Hades
echo ============================================================
echo.
echo   Available recordings:
echo.
dir /b "games\hades\recordings" 2>nul || echo     (No recordings yet - run record.bat first)
echo.
set /p DATA_DIR="Enter recording folder name: "
echo.
echo   Training on: games\hades\recordings\%DATA_DIR%
echo.
cd /d "%~dp0..\.."
venv\Scripts\python.exe scripts\dagger_train.py --data-dir "games\hades\recordings\%DATA_DIR%" --output-dir "games\hades\checkpoints" --epochs 10 --batch-size 4 --lr 1e-5
pause
