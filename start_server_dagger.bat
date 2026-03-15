@echo off
cd /d "%~dp0"
echo ============================================================
echo   NitroGen DAgger-Trained Model Server
echo ============================================================
echo.
echo This serves the model trained with human corrections.
echo.
echo   Checkpoint: nitrogen_hades_dagger_best.pt
echo   Context: 4 frames
echo   Timesteps: 4 (fast)
echo   FP16: Enabled
echo.
echo ============================================================
echo.

REM Check if DAgger checkpoint exists
if exist "checkpoints\nitrogen_hades_dagger_best.pt" (
    echo Using DAgger-trained checkpoint...
    venv\Scripts\python scripts\serve_multiframe.py ^
        --ckpt checkpoints\nitrogen_hades_dagger_best.pt ^
        --ctx 4 ^
        --timesteps 4 ^
        --fp16
) else (
    echo DAgger checkpoint not found!
    echo.
    echo Please run dagger_train.bat first.
    echo.
    echo Falling back to multiframe checkpoint...
    if exist "checkpoints\nitrogen_hades_multiframe_best.pt" (
        venv\Scripts\python scripts\serve_multiframe.py ^
            --ckpt checkpoints\nitrogen_hades_multiframe_best.pt ^
            --ctx 4 ^
            --timesteps 4 ^
            --fp16
    ) else (
        echo No suitable checkpoint found!
        pause
        exit /b 1
    )
)

pause
