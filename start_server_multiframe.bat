@echo off
cd /d "%~dp0"
echo ============================================================
echo   NitroGen Multi-Frame Inference Server
echo ============================================================
echo.
echo This server uses 4-frame temporal context for better
echo motion understanding and objective awareness.
echo.
echo   Context: 4 frames
echo   Timesteps: 4 (fast inference)
echo   FP16: Enabled
echo.
echo ============================================================
echo.

REM Check if multi-frame checkpoint exists
if exist "checkpoints\nitrogen_hades_multiframe_best.pt" (
    echo Using multi-frame checkpoint...
    venv\Scripts\python scripts\serve_multiframe.py ^
        --ckpt checkpoints\nitrogen_hades_multiframe_best.pt ^
        --ctx 4 ^
        --timesteps 4 ^
        --fp16
) else (
    echo No multi-frame checkpoint found.
    echo.
    echo Using standard checkpoint with 4-frame context...
    echo Note: For best results, train with train_multiframe.bat first.
    echo.
    if exist "checkpoints\nitrogen_hades_best.pt" (
        venv\Scripts\python scripts\serve_multiframe.py ^
            --ckpt checkpoints\nitrogen_hades_best.pt ^
            --ctx 4 ^
            --timesteps 4 ^
            --fp16
    ) else (
        echo Error: No checkpoint found!
        echo Please train a model first.
        pause
        exit /b 1
    )
)

pause
