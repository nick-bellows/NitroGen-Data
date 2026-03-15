@echo off
cd /d "%~dp0.."
echo ============================================================
echo   Step 4: Apply NVIDIA Labels to Extracted Frames
echo ============================================================
echo.
echo This step matches your extracted frames to NVIDIA's action labels.
echo It requires scanning the dataset again (may take 10-20 minutes).
echo.
venv\Scripts\python scripts\nvidia_apply_labels.py --game hades --frames-dir nvidia_data/extracted
echo.
pause
