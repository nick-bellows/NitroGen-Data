@echo off
cd /d "%~dp0.."
echo ============================================================
echo   NVIDIA Data Pipeline - Full Run
echo ============================================================
echo.
echo This will run all 5 steps of the pipeline.
echo Estimated time: 2-6 hours depending on downloads.
echo.
echo Press Ctrl+C to cancel, or
pause

call nvidia_pipeline\1_explore.bat
call nvidia_pipeline\2_download.bat
call nvidia_pipeline\3_extract.bat
call nvidia_pipeline\4_apply_labels.bat
call nvidia_pipeline\5_merge.bat

echo.
echo ============================================================
echo   Pipeline Complete!
echo ============================================================
echo.
echo To train on the combined dataset:
echo python scripts/dagger_train.py --data-dir combined_training_data/user_0000 --epochs 10
echo.
pause
