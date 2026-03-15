@echo off
cd /d "%~dp0.."
echo ============================================================
echo   Step 5: Merge NVIDIA Data with Your Recordings
echo ============================================================
echo.
echo This will combine NVIDIA data with your Hades recordings.
echo.
set /p USERDIR="Path to your recording (e.g., games/hades/recordings/your_session): "
if "%USERDIR%"=="" (
    echo Please specify a recording path.
    pause
    exit /b 1
)
echo.
set /p WEIGHT="NVIDIA data weight (0.0-1.0, default 1.0): "
if "%WEIGHT%"=="" set WEIGHT=1.0
echo.
venv\Scripts\python scripts\nvidia_merge_datasets.py --user-dir "%USERDIR%" --nvidia-dir nvidia_data/extracted --output-dir combined_training_data --nvidia-weight %WEIGHT% --validate
echo.
pause
