@echo off
REM Enhanced Docker Build Script with Conda Support - Fixed Version
REM Incorporates error handling and fallback mechanisms from debug version

setlocal enabledelayedexpansion

REM Default values
set BUILD_TYPE=production
set DOCKERFILE=Dockerfile.conda
set IMAGE_NAME=cyber-assessment-reviewer
set TAG=conda-latest
set CACHE_FROM=
set NO_CACHE=false
set PLATFORM=linux/amd64

echo ========================================
echo Docker + Conda Build Script (Enhanced)
echo ========================================

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :start_build
if "%~1"=="-t" (
    set BUILD_TYPE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--type" (
    set BUILD_TYPE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-cache" (
    set NO_CACHE=true
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help

echo Unknown option: %~1
goto :show_help

:show_help
echo Usage: %0 [OPTIONS]
echo.
echo Options:
echo   -t, --type TYPE        Build type: production, development, minimal (default: production)
echo   --no-cache             Build without using cache
echo   -h, --help             Show this help message
echo.
goto :end

:start_build

REM Validate build type
if "%BUILD_TYPE%"=="production" goto :valid_type
if "%BUILD_TYPE%"=="development" goto :valid_type
if "%BUILD_TYPE%"=="minimal" goto :valid_type
echo ❌ Invalid build type: %BUILD_TYPE%
echo Valid types: production, development, minimal
goto :end

:valid_type

REM Set target based on build type
if "%BUILD_TYPE%"=="production" (
    set TARGET=production
    set FULL_TAG=%TAG%
)
if "%BUILD_TYPE%"=="development" (
    set TARGET=development
    set FULL_TAG=%TAG%-dev
)
if "%BUILD_TYPE%"=="minimal" (
    set TARGET=minimal-production
    set FULL_TAG=%TAG%-minimal
)

echo Build Type: %BUILD_TYPE%
echo Dockerfile: %DOCKERFILE%
echo Target: %TARGET%
echo Image: %IMAGE_NAME%:%FULL_TAG%
echo Platform: %PLATFORM%
echo.

REM Check prerequisites
echo Checking prerequisites...

REM Check if Dockerfile exists
if not exist "%DOCKERFILE%" (
    echo ❌ Dockerfile not found: %DOCKERFILE%
    goto :end
)

REM Check if environment.yml exists
if not exist "environment.yml" (
    echo ❌ environment.yml not found
    echo ℹ️  Run 'python setup_environment.py --export' to create it
    goto :end
)

REM Check Docker availability
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker not found. Please install Docker first.
    goto :end
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker daemon is not running. Please start Docker.
    goto :end
)

echo ✅ Docker detected and running

REM Check disk space (need at least 15GB)
for /f "tokens=3" %%a in ('dir C:\ ^| findstr "bytes free"') do set FREE_SPACE=%%a
set /a FREE_GB=%FREE_SPACE:~0,-10%
if %FREE_GB% LSS 15 (
    echo ⚠️  Warning: Low disk space (%FREE_GB%GB free, 15GB+ recommended)
    echo Consider running: docker system prune -a -f
)

REM Show environment.yml summary
echo.
echo ℹ️  Environment configuration:
findstr /C:"name:" environment.yml
findstr /C:"python=" environment.yml
echo.

REM Build cache options
set CACHE_OPTS=
if "%NO_CACHE%"=="true" (
    set CACHE_OPTS=--no-cache
)

echo ℹ️  Starting Docker build with enhanced error handling...
echo.

REM Build the image with progress output
set BUILD_CMD=docker build --platform %PLATFORM% --target %TARGET% --tag %IMAGE_NAME%:%FULL_TAG% --file %DOCKERFILE% %CACHE_OPTS% --progress=plain .

echo Build command: %BUILD_CMD%
echo.

REM Execute build
%BUILD_CMD%
if errorlevel 1 (
    echo.
    echo ❌ Docker build failed!
    echo.
    echo Troubleshooting suggestions:
    echo 1. Check build logs above for specific error
    echo 2. Try: docker system prune -a -f
    echo 3. Try: %0 --no-cache
    echo 4. Check Docker Desktop memory allocation (8GB+ recommended)
    echo 5. Try the debug build: docker-build-conda-debug.bat
    echo.
    goto :end
)

echo.
echo ✅ Docker build completed successfully!
echo.

REM Show image information
echo ℹ️  Image Information:
docker images %IMAGE_NAME%:%FULL_TAG% --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo.

REM Test the image
echo ℹ️  Testing image...
docker run --rm %IMAGE_NAME%:%FULL_TAG% python -c "import flask, pandas; print('✅ Core dependencies verified')" 2>nul
if errorlevel 1 (
    echo ⚠️  Basic dependency test failed
    echo Testing with conda environment activation...
    docker run --rm %IMAGE_NAME%:%FULL_TAG% /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate cyber-assessment-env && python -c 'import flask, pandas; print(\"✅ Dependencies verified in conda environment\")'"
) else (
    echo ✅ Image test passed!
)

REM Test AI dependencies
echo Testing AI dependencies...
docker run --rm %IMAGE_NAME%:%FULL_TAG% python -c "import transformers; print('✅ AI dependencies verified')" 2>nul
if errorlevel 1 (
    echo ⚠️  AI dependencies test failed (this may be normal if transformers is not installed)
) else (
    echo ✅ AI dependencies test passed!
)

echo.

REM Show next steps
echo ✅ Build completed! Next steps:
echo.
echo 1. Run the container:
echo    docker run -p 5000:5000 %IMAGE_NAME%:%FULL_TAG%
echo.
echo 2. Or use docker-compose:
echo    docker-compose -f docker-compose.conda.yml up -d
echo.
echo 3. For development:
echo    docker-compose -f docker-compose.conda.yml --profile dev up -d
echo.
echo 4. Test the application:
echo    curl http://localhost:5000/system_status
echo.

REM Optional: Tag additional versions
if "%BUILD_TYPE%"=="production" (
    echo ℹ️  Tagging as latest...
    docker tag %IMAGE_NAME%:%FULL_TAG% %IMAGE_NAME%:latest-conda
    echo ✅ Tagged as %IMAGE_NAME%:latest-conda
    echo.
)

REM Show resource usage
echo ℹ️  Docker resource usage:
docker system df
echo.

echo ✅ Enhanced Docker build script completed successfully!

:end
pause
