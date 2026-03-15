@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NitroGen Index Builder - Step 2: Extract Metadata
echo ============================================================
echo.
echo This extracts only metadata.json files from each shard.
echo Large parquet files (~160GB) are SKIPPED.
echo.
echo   Input: nvidia_index\downloads\actions\SHARD_*.tar.gz
echo   Output: nvidia_index\metadata\SHARD_*\*.json
echo   Output size: ~100 MB (vs 160 GB for full data)
echo.
echo This step is also RESUMABLE.
echo.
pause
echo.
venv\Scripts\python scripts/nvidia_index/extract_metadata.py
echo.
echo ============================================================
echo   Extraction complete!
echo   Next: Run 3_generate_index.bat
echo ============================================================
pause
