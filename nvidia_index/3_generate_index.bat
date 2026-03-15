@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NitroGen Index Builder - Step 3: Generate Index
echo ============================================================
echo.
echo This creates the master index files from extracted metadata.
echo.
echo   Input: nvidia_index\metadata\
echo   Output: nvidia_index\index\
echo.
echo Output files:
echo   - master_index.json   (complete details)
echo   - games_summary.csv   (spreadsheet-friendly)
echo   - videos_list.csv     (all video URLs)
echo   - game_shards.json    (find shards by game - MOST USEFUL!)
echo   - shard_map.json      (find games by shard)
echo   - games_by_hours.md   (human-readable ranking)
echo   - README.md           (documentation)
echo.
pause
echo.
venv\Scripts\python scripts/nvidia_index/generate_index.py
echo.
echo ============================================================
echo   Index generation complete!
echo.
echo   Index files are in: nvidia_index\index\
echo.
echo   Optional: Run 4_cleanup_optional.bat to delete the
echo   downloaded shards and free up ~160 GB.
echo ============================================================
pause
