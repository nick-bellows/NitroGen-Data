@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NitroGen Index Builder - Step 1: Download Shards
echo ============================================================
echo.
echo This will download ALL 100 shards from HuggingFace.
echo.
echo   Total size: ~160 GB
echo   Location: nvidia_index\downloads\
echo.
echo Downloads are RESUMABLE - if interrupted, just run again.
echo Progress is saved after each shard.
echo.
pause
echo.
echo Installing dependencies...
venv\Scripts\pip install huggingface_hub --quiet
echo.
echo Starting download...
echo.
venv\Scripts\python scripts/nvidia_index/download_shards.py --start-shard 0 --end-shard 99
echo.
echo ============================================================
echo   Download step complete!
echo   Next: Run 2_extract_metadata.bat
echo ============================================================
pause
