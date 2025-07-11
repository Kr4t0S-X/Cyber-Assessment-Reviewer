@echo off
REM Cyber Assessment Reviewer - Windows Run Script

setlocal enabledelayedexpansion

echo ============================================================
echo 🛡️  Cyber Assessment Reviewer - Starting Application
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo ❌ Virtual environment not found!
    echo 📋 Please run setup first:
    echo    • python setup.py
    echo    • setup.bat
    pause
    exit /b 1
)

echo ✅ Virtual environment found

REM Check if activation script exists
if not exist .venv\Scripts\activate.bat (
    echo ❌ Virtual environment activation script not found
    echo 📋 Please run setup again to fix the environment
    pause
    exit /b 1
)

echo 🚀 Starting Cyber Assessment Reviewer...
echo 💡 Access the application at: http://localhost:5000
echo 🔧 Press Ctrl+C to stop the server
echo.

REM Activate virtual environment and run the application
call .venv\Scripts\activate.bat

REM Check if main application file exists
if not exist app.py (
    echo ❌ Application file (app.py) not found
    echo 📋 Please ensure you're in the correct directory
    pause
    exit /b 1
)

REM Run the application
python app.py

echo.
echo 👋 Application stopped
echo Thank you for using Cyber Assessment Reviewer!
pause