@echo off
REM Docker management script for Cyber Assessment Reviewer (Windows)

setlocal enabledelayedexpansion

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="build" goto build
if "%1"=="deploy" goto deploy
if "%1"=="deploy-transformers" goto deploy_transformers
if "%1"=="test" goto test
if "%1"=="logs" goto logs
if "%1"=="status" goto status
if "%1"=="restart" goto restart
if "%1"=="stop" goto stop
if "%1"=="down" goto down
if "%1"=="clean" goto clean
goto help

:help
echo Cyber Assessment Reviewer - Docker Management
echo =============================================
echo.
echo Available commands:
echo   docker-manage.bat setup              - Initial setup
echo   docker-manage.bat build              - Build Docker images
echo   docker-manage.bat deploy             - Deploy with Ollama backend
echo   docker-manage.bat deploy-transformers - Deploy with Transformers backend
echo   docker-manage.bat test               - Run deployment tests
echo   docker-manage.bat logs               - View application logs
echo   docker-manage.bat status             - Show container status
echo   docker-manage.bat restart            - Restart services
echo   docker-manage.bat stop               - Stop services
echo   docker-manage.bat down               - Stop and remove containers
echo   docker-manage.bat clean              - Clean up Docker resources
echo.
goto end

:setup
echo Setting up environment...
if not exist .env copy .env.example .env
if not exist data mkdir data
if not exist data\uploads mkdir data\uploads
if not exist data\sessions mkdir data\sessions
if not exist data\logs mkdir data\logs
if not exist data\models mkdir data\models
if not exist data\ollama mkdir data\ollama
if not exist data\transformers_cache mkdir data\transformers_cache
echo Setup complete! Please edit .env file with your configuration.
goto end

:build
echo Building Docker images...
call :setup
docker build --target production --tag cyber-assessment-reviewer:latest --tag cyber-assessment-reviewer:ollama .
if errorlevel 1 (
    echo Error building production image
    goto end
)
docker build --target transformers --tag cyber-assessment-reviewer:transformers .
if errorlevel 1 (
    echo Error building transformers image
    goto end
)
echo Build complete!
goto end

:deploy
echo Deploying with Ollama backend...
call :build
docker-compose up -d
if errorlevel 1 (
    echo Error deploying with Ollama
    goto end
)
echo Deployment complete! Application available at http://localhost:5000
goto end

:deploy_transformers
echo Deploying with Transformers backend...
call :build
docker-compose -f docker-compose.transformers.yml up -d
if errorlevel 1 (
    echo Error deploying with Transformers
    goto end
)
echo Deployment complete! Application available at http://localhost:5000
goto end

:test
echo Testing deployment...
echo Checking if containers are running...
docker ps | findstr cyber-assessment
echo Testing application endpoint...
curl -f http://localhost:5000/system_status
echo Test complete!
goto end

:logs
docker-compose logs -f
goto end

:status
echo Container Status:
echo ==================
docker-compose ps
echo.
echo Resource Usage:
echo ===============
docker stats --no-stream
goto end

:restart
echo Restarting services...
docker-compose restart
echo Services restarted!
goto end

:stop
echo Stopping services...
docker-compose stop
echo Services stopped!
goto end

:down
echo Stopping and removing containers...
docker-compose down
docker-compose -f docker-compose.transformers.yml down
echo Containers removed!
goto end

:clean
echo Cleaning up Docker resources...
call :down
docker system prune -f
docker volume prune -f
echo Cleanup complete!
goto end

:end
endlocal
