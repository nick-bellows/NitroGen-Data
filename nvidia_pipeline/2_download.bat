@echo off
cd /d "%~dp0.."
echo ============================================================
echo   Step 2: Download Available Videos
echo ============================================================
echo.
echo   WARNING: This may download several GB of video data!
echo   Make sure you have enough disk space.
echo.
if not exist "nvidia_data\videos" mkdir nvidia_data\videos
echo.
set /p MAXVIDS="How many videos to download? (default: 30): "
if "%MAXVIDS%"=="" set MAXVIDS=30
echo.
echo Downloading up to %MAXVIDS% videos...
venv\Scripts\python scripts\nvidia_video_downloader.py --input nvidia_data/dataset_info/hades_info.json --output-dir nvidia_data/videos --max-videos %MAXVIDS% --resolution 720
echo.
pause
