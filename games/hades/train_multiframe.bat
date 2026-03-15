@echo off
cd /d "%~dp0..\.."
echo ============================================================
echo   NitroGen Multi-Frame Training - Hades
echo ============================================================
echo.
echo This trains the model with 4-frame temporal context.
echo The model will learn to understand motion and direction.
echo.
echo   Data: games\hades\recordings\YOUR_SESSION
echo   Frames: 4 (temporal context)
echo   Epochs: 10
echo.
echo ============================================================
echo.

REM Find the most recent recording
set "DATA_DIR=games\hades\recordings\YOUR_SESSION"

echo Training with data from: %DATA_DIR%
echo.

venv\Scripts\python scripts\dagger_train_multiframe.py ^
    --data-dir "%DATA_DIR%" ^
    --epochs 10 ^
    --batch-size 4 ^
    --lr 1e-5 ^
    --num-frames 4 ^
    --frame-skip 1

echo.
echo ============================================================
echo   Training Complete!
echo ============================================================
echo.
echo Model saved to: checkpoints\nitrogen_hades_multiframe_best.pt
echo.
echo To test the model:
echo   1. Run start_server_multiframe.bat
echo   2. Run games\hades\play.bat
echo.
pause
