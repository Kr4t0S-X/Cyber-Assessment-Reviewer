@echo off
REM Debug version of Docker Build Script with enhanced error handling
REM Use this to troubleshoot conda environment creation issues

setlocal enabledelayedexpansion

echo ========================================
echo Docker + Conda Debug Build Script
echo ========================================

REM Check prerequisites
echo Checking prerequisites...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not available
    pause
    exit /b 1
)

echo ✅ Docker available

REM Check required files
if not exist "Dockerfile.conda" (
    echo ❌ Dockerfile.conda not found
    pause
    exit /b 1
)

if not exist "environment.yml" (
    echo ❌ environment.yml not found
    pause
    exit /b 1
)

echo ✅ Required files found

REM Show environment.yml contents
echo.
echo === Environment.yml Contents ===
type environment.yml
echo === End Environment.yml ===
echo.

REM Method 1: Try debug dockerfile
echo Method 1: Trying debug dockerfile with verbose output...
docker build -f Dockerfile.conda-debug --target env-builder-debug --progress=plain --no-cache -t debug-test .
if not errorlevel 1 (
    echo ✅ Debug build succeeded!
    goto :success
)

echo ⚠️  Debug build failed, trying Method 2...

REM Method 2: Try fallback dockerfile
echo Method 2: Trying step-by-step fallback build...
docker build -f Dockerfile.conda-debug --target env-builder-fallback --progress=plain --no-cache -t fallback-test .
if not errorlevel 1 (
    echo ✅ Fallback build succeeded!
    goto :success
)

echo ⚠️  Fallback build failed, trying Method 3...

REM Method 3: Try minimal environment
echo Method 3: Trying minimal environment...
if exist "environment-minimal.yml" (
    docker build -f Dockerfile.conda --target env-builder --progress=plain --no-cache --build-arg ENV_FILE=environment-minimal.yml -t minimal-test .
    if not errorlevel 1 (
        echo ✅ Minimal build succeeded!
        echo ℹ️  You can gradually add packages to environment-minimal.yml
        goto :success
    )
)

echo ⚠️  Minimal build failed, trying Method 4...

REM Method 4: Try pip-only fallback
echo Method 4: Trying pip-only fallback...
docker build -f Dockerfile.conda-debug --target pip-fallback --progress=plain --no-cache -t pip-test .
if not errorlevel 1 (
    echo ✅ Pip-only build succeeded!
    echo ℹ️  Using pip instead of conda
    goto :success
)

echo ❌ All build methods failed!
goto :failure

:success
echo.
echo ========================================
echo 🎉 Build Successful!
echo ========================================
echo.
echo Next steps:
echo 1. Test the image: docker run --rm -it [image-name] python -c "import flask; print('OK')"
echo 2. If minimal build worked, gradually add packages to environment-minimal.yml
echo 3. Use the successful method for your production build
echo.
goto :end

:failure
echo.
echo ========================================
echo ❌ Build Failed - Troubleshooting Guide
echo ========================================
echo.
echo Possible issues and solutions:
echo.
echo 1. Network connectivity:
echo    - Check internet connection
echo    - Try: docker run --rm continuumio/miniconda3 conda install -c conda-forge flask
echo.
echo 2. Disk space:
echo    - Check: dir C:\
echo    - Clean: docker system prune -a -f
echo.
echo 3. Memory issues:
echo    - Increase Docker Desktop memory (8GB+)
echo    - Docker Desktop → Settings → Resources → Advanced
echo.
echo 4. Package conflicts:
echo    - Try building with environment-minimal.yml
echo    - Add packages one by one to identify conflicts
echo.
echo 5. Platform issues:
echo    - Try: docker build --platform linux/amd64 ...
echo.
echo 6. WSL2 issues:
echo    - Restart: wsl --shutdown
echo    - Update: wsl --update
echo.
echo For detailed logs, run:
echo docker build -f Dockerfile.conda-debug --target env-builder-debug --progress=plain --no-cache -t debug-test . > build-log.txt 2>&1
echo.

:end
pause
