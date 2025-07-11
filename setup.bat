@echo off
REM Cyber Assessment Reviewer - Windows Setup Script

setlocal enabledelayedexpansion

echo ============================================================
echo 🛡️  Cyber Assessment Reviewer - Setup
echo ============================================================
echo.

REM Step 1: Check Python version
echo 📋 Step 1/6: Checking Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo    ❌ Python is not installed or not in PATH
    echo    📋 Please install Python 3.10 or higher from https://python.org
    echo    💡 Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo    ✅ Python %PYTHON_VERSION% found

REM Step 2: Install uv
echo 📋 Step 2/6: Installing uv (ultra-fast Python package manager)
uv --version >nul 2>&1
if errorlevel 1 (
    echo    🔧 Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo    ❌ Failed to install uv
        echo    📋 Please install uv manually from https://github.com/astral-sh/uv
        pause
        exit /b 1
    )
    echo    ✅ uv installed successfully
) else (
    echo    ✅ uv is already installed
)

REM Step 3: Create virtual environment
echo 📋 Step 3/6: Creating virtual environment
if exist .venv (
    echo    ✅ Virtual environment already exists
) else (
    echo    🔧 Creating virtual environment...
    uv venv
    if errorlevel 1 (
        echo    ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo    ✅ Virtual environment created successfully
)

REM Step 4: Install dependencies
echo 📋 Step 4/6: Installing project dependencies
echo    🔧 Installing dependencies with uv...
uv pip install -e .
if errorlevel 1 (
    echo    ⚠️  uv installation failed, trying pip fallback...
    call .venv\Scripts\activate.bat
    pip install -e .
    if errorlevel 1 (
        echo    ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo    ✅ Dependencies installed with pip (fallback)
) else (
    echo    ✅ Dependencies installed successfully
)

REM Step 5: Verify installation
echo 📋 Step 5/6: Verifying installation
call .venv\Scripts\activate.bat
python -c "import flask, pandas, transformers; print('✅ Core packages imported successfully')"
if errorlevel 1 (
    echo    ❌ Installation verification failed
    pause
    exit /b 1
)
echo    ✅ Installation verified successfully

REM Step 6: Show completion message
echo 📋 Step 6/6: Setup completed successfully!
echo.
echo 🎉 Installation Complete!
echo ============================================================
echo.
echo 📋 Next Steps:
echo    1. Run the application:
echo       • run.bat  (or python run.py)
echo.
echo    2. Access the web interface:
echo       • Open your browser to: http://localhost:5000
echo.
echo 💡 Tips:
echo    • For better performance, install Ollama: https://ollama.com
echo    • First run may take longer as models download
echo    • Check README.md for detailed usage instructions
echo.
echo 🛡️  Ready to analyze cybersecurity assessments!
echo ============================================================
echo.
pause