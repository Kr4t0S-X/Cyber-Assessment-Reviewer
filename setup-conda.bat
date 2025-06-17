@echo off
REM Conda Environment Setup Script for Windows
REM Cyber Assessment Reviewer

echo ========================================
echo Cyber Assessment Reviewer - Conda Setup
echo ========================================
echo.

REM Check if conda is available
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda not found in PATH
    echo.
    echo Please install Anaconda or Miniconda:
    echo - Anaconda: https://www.anaconda.com/products/distribution
    echo - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo After installation, restart your command prompt and run this script again.
    pause
    exit /b 1
)

echo ✅ Conda detected
conda --version
echo.

REM Check if environment already exists
conda env list | findstr "cyber-assessment-env" >nul 2>&1
if not errorlevel 1 (
    echo ⚠️  Environment 'cyber-assessment-env' already exists
    echo.
    set /p choice="Do you want to remove and recreate it? (y/N): "
    if /i "%choice%"=="y" (
        echo Removing existing environment...
        conda env remove -n cyber-assessment-env -y
        echo ✅ Environment removed
    ) else (
        echo Using existing environment
        goto :activate_env
    )
)

echo Creating conda environment from environment.yml...
if exist environment.yml (
    conda env create -f environment.yml
    if errorlevel 1 (
        echo ❌ Failed to create environment from environment.yml
        echo Trying alternative method...
        goto :create_manual
    )
    echo ✅ Environment created from environment.yml
    goto :install_pip_deps
) else (
    echo ⚠️  environment.yml not found, creating environment manually...
    goto :create_manual
)

:create_manual
echo Creating environment manually...
conda create -n cyber-assessment-env python=3.10 -y
if errorlevel 1 (
    echo ❌ Failed to create conda environment
    pause
    exit /b 1
)
echo ✅ Environment created

echo Installing conda packages...
conda install -n cyber-assessment-env -c conda-forge flask pandas numpy requests python-docx openpyxl pypdf2 python-pptx transformers torch scikit-learn matplotlib seaborn jupyter ipykernel -y
if errorlevel 1 (
    echo ⚠️  Some conda packages failed to install
    echo Will try pip fallback...
)

:install_pip_deps
echo Installing pip dependencies...
conda run -n cyber-assessment-env pip install ollama
if errorlevel 1 (
    echo ⚠️  Some pip packages failed to install
)

:activate_env
echo.
echo ========================================
echo 🎉 Setup Complete!
echo ========================================
echo.
echo To use the Cyber Assessment Reviewer:
echo.
echo 1. Activate the environment:
echo    conda activate cyber-assessment-env
echo.
echo 2. Run the application:
echo    python app.py
echo.
echo 3. Or run tests:
echo    python test_ai_accuracy.py
echo.
echo To deactivate the environment:
echo    conda deactivate
echo.
echo Environment management:
echo - List environments: conda env list
echo - Export environment: conda env export -n cyber-assessment-env -f environment.yml
echo - Remove environment: conda env remove -n cyber-assessment-env
echo.

set /p choice="Do you want to activate the environment now? (Y/n): "
if /i not "%choice%"=="n" (
    echo.
    echo Activating environment...
    echo Run: conda activate cyber-assessment-env
    echo.
    cmd /k "conda activate cyber-assessment-env"
) else (
    echo.
    echo Remember to activate the environment before using the application:
    echo conda activate cyber-assessment-env
)

pause
