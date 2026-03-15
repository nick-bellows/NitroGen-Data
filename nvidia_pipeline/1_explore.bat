@echo off
cd /d "%~dp0.."
echo ============================================================
echo   Step 1: Explore NVIDIA Dataset for Hades
echo ============================================================
echo.
echo WARNING: This downloads ~1.6GB per shard from HuggingFace.
echo Default: 5 shards = ~8GB download (cached for future use)
echo.
if not exist "nvidia_data\dataset_info" mkdir nvidia_data\dataset_info
echo Installing dependencies...
venv\Scripts\pip install huggingface_hub yt-dlp --quiet
echo.
set /p MAXSHARDS="How many shards to scan? (default: 5, max: 100): "
if "%MAXSHARDS%"=="" set MAXSHARDS=5
echo.
echo Starting exploration (downloading %MAXSHARDS% shards)...
echo This may take a while depending on your internet speed...
echo.
venv\Scripts\python scripts\nvidia_dataset_explorer.py --game hades --max-shards %MAXSHARDS% --check-availability --check-limit 10 --output nvidia_data/dataset_info/hades_info.json
echo.
pause
