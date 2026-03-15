@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NitroGen Index Builder - Test Run (5 shards only)
echo ============================================================
echo.
echo This is a TEST run that only downloads 5 shards (~8 GB).
echo Use this to verify everything works before running the full pipeline.
echo.
echo   Shards: 0-4 (out of 100)
echo   Download size: ~8 GB
echo   Time: ~30 minutes
echo.
pause
echo.
echo Installing dependencies...
venv\Scripts\pip install huggingface_hub --quiet
echo.
venv\Scripts\python scripts/nvidia_index/build_master_index.py --start-shard 0 --end-shard 4
echo.
echo ============================================================
echo   Test Complete!
echo ============================================================
echo.
echo Check nvidia_index\index\ to see sample output.
echo.
echo If everything looks good, run run_full_pipeline.bat
echo to process all 100 shards.
echo.
pause
