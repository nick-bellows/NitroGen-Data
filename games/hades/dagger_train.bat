@echo off
cd /d "%~dp0..\.."
echo ============================================================
echo   DAgger Training - Hades
echo ============================================================
echo.
echo This trains the model on your corrections with higher weight.
echo.
echo   Base data: games\hades\recordings\YOUR_SESSION
echo   Corrections: games\hades\dagger_sessions\
echo   Correction weight: 2.0x
echo   Epochs: 5
echo.
echo ============================================================
echo.

REM Check if dagger sessions exist
if not exist "games\hades\dagger_sessions" (
    echo No DAgger sessions found!
    echo.
    echo Run dagger_collect.bat first to collect corrections.
    echo.
    pause
    exit /b 1
)

REM Count sessions
set count=0
for /d %%i in (games\hades\dagger_sessions\*) do set /a count+=1
echo Found %count% DAgger session(s)
echo.

pause
echo.
echo Training with corrections...
echo.

venv\Scripts\python scripts\dagger_train_weighted.py ^
    --base-data games\hades\recordings\YOUR_SESSION ^
    --corrections games\hades\dagger_sessions ^
    --correction-weight 2.0 ^
    --epochs 5 ^
    --num-frames 4

echo.
echo ============================================================
echo   DAgger Training Complete!
echo ============================================================
echo.
echo Model saved to: checkpoints\nitrogen_hades_dagger_best.pt
echo.
echo To test the improved model:
echo   1. Run start_server_dagger.bat
echo   2. Run games\hades\play.bat
echo.
pause
