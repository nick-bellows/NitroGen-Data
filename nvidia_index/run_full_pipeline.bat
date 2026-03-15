@echo off
cd /d "%~dp0.."
echo ############################################################
echo   NitroGen Master Index Builder - Full Pipeline
echo ############################################################
echo.
echo This will run the complete pipeline:
echo.
echo   Step 1: Download all 100 shards (~160 GB, resumable)
echo   Step 2: Extract metadata.json files from each shard
echo   Step 3: Generate master index files
echo.
echo Estimated time: 4-8 hours (depends on internet speed)
echo   - Download: ~3-6 hours at 10 MB/s
echo   - Extraction: ~1 hour
echo   - Index generation: ~10 minutes
echo.
echo Output will be in: nvidia_index\index\
echo.
echo The pipeline is RESUMABLE - if interrupted, run again
echo and it will continue where it left off.
echo.
echo ############################################################
echo.
pause
echo.
echo Installing dependencies...
venv\Scripts\pip install huggingface_hub --quiet
echo.
venv\Scripts\python scripts/nvidia_index/build_master_index.py --start-shard 0 --end-shard 99
echo.
echo ############################################################
echo   Pipeline Complete!
echo ############################################################
echo.
echo Index files are in: nvidia_index\index\
echo.
echo To find which shards contain a specific game:
echo   1. Open nvidia_index\index\game_shards.json
echo   2. Search for your game name
echo   3. Download only those shards from HuggingFace
echo.
pause
