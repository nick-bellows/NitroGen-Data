@echo off
echo ============================================================
echo NitroGen Inference Server
echo ============================================================
echo.
echo Select server mode:
echo   1. Standard (16 timesteps, best quality)
echo   2. Optimized (4 timesteps, FP16 - RECOMMENDED)
echo   3. Ultra (2 timesteps, compiled - fastest)
echo.
set /p MODE="Enter choice (1-3, default=2): "
if "%MODE%"=="" set MODE=2

REM Set checkpoint path - update this to your local ng.pt location
REM Download with: huggingface-cli download nvidia/NitroGen ng.pt
set CKPT="%USERPROFILE%\.cache\huggingface\hub\models--nvidia--NitroGen\snapshots\aac2ba563ea94ba612d2d464ef05dff6069e2c13\ng.pt"

if "%MODE%"=="1" (
    echo Starting Standard server...
    venv\Scripts\python.exe scripts\serve.py %CKPT%
) else if "%MODE%"=="3" (
    echo Starting Ultra server...
    venv\Scripts\python.exe scripts\serve_optimized.py %CKPT% --timesteps 2 --fp16 --compile
) else (
    echo Starting Optimized server...
    venv\Scripts\python.exe scripts\serve_optimized.py %CKPT% --timesteps 4 --fp16
)
