@echo off
REM Enhanced Docker Build Script with Conda Support for Windows
REM Builds optimized Docker images using conda for dependency management

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
echo Docker + Conda Build Script for Windows
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
if "%~1"=="-f" (
    set DOCKERFILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--dockerfile" (
    set DOCKERFILE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-n" (
    set IMAGE_NAME=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--name" (
    set IMAGE_NAME=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--tag" (
    set TAG=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-cache" (
    set NO_CACHE=true
    shift
    goto :parse_args
)
if "%~1"=="--platform" (
    set PLATFORM=%~2
    shift
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
echo   -f, --dockerfile FILE  Dockerfile to use (default: Dockerfile.conda)
echo   -n, --name NAME        Image name (default: cyber-assessment-reviewer)
echo   --tag TAG              Image tag (default: conda-latest)
echo   --no-cache             Build without using cache
echo   --platform PLATFORM   Target platform (default: linux/amd64)
echo   -h, --help             Show this help message
echo.
echo Examples:
echo   %0                                    # Build production image
echo   %0 --type development                 # Build development image
echo   %0 --type minimal --no-cache          # Build minimal image without cache
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
echo.

REM Build cache options
set CACHE_OPTS=
if "%NO_CACHE%"=="true" (
    set CACHE_OPTS=--no-cache
)

echo ℹ️  Starting Docker build...
echo.

REM Build the image
set BUILD_CMD=docker build --platform %PLATFORM% --target %TARGET% --tag %IMAGE_NAME%:%FULL_TAG% --file %DOCKERFILE% %CACHE_OPTS% .

echo Build command: %BUILD_CMD%
echo.

REM Execute build
%BUILD_CMD%
if errorlevel 1 (
    echo ❌ Docker build failed!
    goto :end
)

echo ✅ Docker build completed successfully!
echo.

REM Show image information
echo ℹ️  Image Information:
docker images %IMAGE_NAME%:%FULL_TAG% --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo.

REM Test the image
echo ℹ️  Testing image...
docker run --rm %IMAGE_NAME%:%FULL_TAG% python -c "import flask, pandas, transformers; print('✅ Dependencies verified')"
if errorlevel 1 (
    echo ⚠️  Image test failed - dependencies may be missing
) else (
    echo ✅ Image test passed!
)
echo.

REM Show next steps
echo ✅ Build completed! Next steps:
echo 1. Run the container:
echo    docker run -p 5000:5000 %IMAGE_NAME%:%FULL_TAG%
echo.
echo 2. Or use docker-compose:
echo    docker-compose -f docker-compose.conda.yml up
echo.
echo 3. For development:
echo    docker-compose -f docker-compose.conda.yml --profile dev up
echo.

REM Optional: Tag additional versions
if "%BUILD_TYPE%"=="production" (
    echo ℹ️  Tagging as latest...
    docker tag %IMAGE_NAME%:%FULL_TAG% %IMAGE_NAME%:latest-conda
    echo ✅ Tagged as %IMAGE_NAME%:latest-conda
    echo.
)

REM Optional: Show build cache usage
echo ℹ️  Build cache information:
docker system df
echo.

echo ✅ Docker build script completed successfully!

:end
pause
