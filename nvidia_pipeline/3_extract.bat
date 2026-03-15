@echo off
cd /d "%~dp0.."
echo ============================================================
echo   Step 3: Extract Frames from Videos
echo ============================================================
echo.
if not exist "nvidia_data\extracted" mkdir nvidia_data\extracted
echo.
set /p MAXCHUNKS="How many chunks to extract? (default: 100): "
if "%MAXCHUNKS%"=="" set MAXCHUNKS=100
echo.
echo Extracting up to %MAXCHUNKS% chunks...
venv\Scripts\python scripts\nvidia_frame_extractor.py --dataset-info nvidia_data/dataset_info/hades_info.json --videos-dir nvidia_data/videos --output-dir nvidia_data/extracted --max-chunks %MAXCHUNKS%
echo.
pause
