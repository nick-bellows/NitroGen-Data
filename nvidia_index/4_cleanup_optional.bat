@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NitroGen Index Builder - Cleanup (OPTIONAL)
echo ============================================================
echo.
echo This deletes the downloaded shard files to free disk space.
echo.
echo   Will delete: nvidia_index\downloads\ (~160 GB)
echo   Will keep:   nvidia_index\metadata\  (~100 MB)
echo   Will keep:   nvidia_index\index\     (~60 MB)
echo.
echo Only run this AFTER extraction and index generation are complete!
echo.
echo ============================================================
echo.
venv\Scripts\python scripts/nvidia_index/cleanup_downloads.py
echo.
pause
