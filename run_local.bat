@echo off
cd /d "%~dp0"

if not exist "notebooks\models\XGBoostModel.pkl" (
    echo The saved model file is missing: notebooks\models\XGBoostModel.pkl
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project's Python environment...
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 (
        echo Could not create the Python environment. Install Python 3 and try again.
        pause
        exit /b 1
    )
)

echo Installing the small set of packages needed for local use...
".venv\Scripts\python.exe" -m pip install -r requirements-local.txt
if errorlevel 1 (
    echo Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Starting the local detector. Keep this window open.
echo When it starts, visit http://127.0.0.1:8000/docs
echo Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" -m uvicorn app.local_app:app --host 127.0.0.1 --port 8000
pause
