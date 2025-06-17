@echo off
REM Docker Health Check Script for Windows
REM Diagnoses common Docker Desktop issues

echo ========================================
echo Docker Health Check for Windows
echo ========================================
echo.

REM Check if Docker command is available
echo [1/8] Checking if Docker is installed...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker command not found. Please install Docker Desktop.
    echo Download from: https://www.docker.com/products/docker-desktop
    goto end
) else (
    echo ✅ Docker is installed
    docker --version
)
echo.

REM Check Docker daemon connection
echo [2/8] Checking Docker daemon connection...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Cannot connect to Docker daemon
    echo 💡 Try: Restart Docker Desktop
    echo 💡 Check if Docker Desktop is running in system tray
) else (
    echo ✅ Docker daemon is running
)
echo.

REM Test basic Docker functionality
echo [3/8] Testing basic Docker functionality...
docker run --rm hello-world >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker run test failed
    echo 💡 Try: docker run hello-world
) else (
    echo ✅ Docker run test passed
)
echo.

REM Check Docker Desktop status
echo [4/8] Checking Docker Desktop status...
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>nul | find /I "Docker Desktop.exe" >nul
if errorlevel 1 (
    echo ❌ Docker Desktop is not running
    echo 💡 Start Docker Desktop from Start Menu
) else (
    echo ✅ Docker Desktop is running
)
echo.

REM Check WSL2 (if applicable)
echo [5/8] Checking WSL2 status...
wsl --list --verbose >nul 2>&1
if errorlevel 1 (
    echo ⚠️  WSL2 not available or not configured
    echo 💡 Install WSL2: wsl --install
) else (
    echo ✅ WSL2 is available
    wsl --list --verbose
)
echo.

REM Check available disk space
echo [6/8] Checking disk space...
for /f "tokens=3" %%a in ('dir /-c %SystemDrive%\ ^| find "bytes free"') do set free=%%a
echo Available disk space: %free% bytes
echo.

REM Check if our project files exist
echo [7/8] Checking project files...
if exist Dockerfile (
    echo ✅ Dockerfile found
) else (
    echo ❌ Dockerfile not found
    echo 💡 Make sure you're in the project directory
)

if exist docker-compose.yml (
    echo ✅ docker-compose.yml found
) else (
    echo ❌ docker-compose.yml not found
)
echo.

REM Test Docker build capability
echo [8/8] Testing Docker build capability...
echo FROM hello-world > test.dockerfile
docker build -f test.dockerfile -t test-build . >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker build test failed
    echo 💡 This is the likely cause of your build error
) else (
    echo ✅ Docker build test passed
    docker rmi test-build >nul 2>&1
)
del test.dockerfile >nul 2>&1
echo.

echo ========================================
echo Health Check Complete
echo ========================================
echo.

REM Provide recommendations
echo 🔧 RECOMMENDATIONS:
echo.
if errorlevel 1 (
    echo 1. Restart Docker Desktop:
    echo    - Right-click Docker icon in system tray
    echo    - Select "Quit Docker Desktop"
    echo    - Wait 30 seconds
    echo    - Start Docker Desktop from Start Menu
    echo.
    echo 2. If that doesn't work, try:
    echo    - Docker Desktop Settings ^> Troubleshoot ^> Reset to factory defaults
    echo.
    echo 3. Check Windows features:
    echo    - Enable Hyper-V, WSL2, Virtual Machine Platform
    echo    - Restart computer after enabling
    echo.
    echo 4. For persistent issues:
    echo    - See DOCKER_TROUBLESHOOTING.md for detailed solutions
) else (
    echo ✅ Docker appears to be working correctly!
    echo.
    echo You can now try building the Cyber Assessment Reviewer:
    echo   docker-manage.bat build
    echo   docker-manage.bat deploy
)

:end
echo.
pause
