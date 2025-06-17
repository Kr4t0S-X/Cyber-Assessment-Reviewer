#!/bin/bash
# Docker Deployment Script with Conda Support
# Deploys the Cyber Assessment Reviewer using conda-optimized containers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Default values
MODE="production"
COMPOSE_FILE="docker-compose.conda.yml"
BUILD_FIRST=true
DETACHED=true
PULL_MODELS=true
CLEANUP_FIRST=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Deployment mode: production, development, test (default: production)"
    echo "  -f, --file FILE        Docker compose file (default: docker-compose.conda.yml)"
    echo "  --no-build             Skip building images (use existing)"
    echo "  --no-detach            Run in foreground (don't detach)"
    echo "  --no-models            Skip model pulling"
    echo "  --cleanup              Clean up existing containers first"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy production"
    echo "  $0 --mode development                 # Deploy development"
    echo "  $0 --cleanup --no-models              # Clean deploy without models"
    echo "  $0 --mode test --no-detach            # Test deployment in foreground"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --no-build)
            BUILD_FIRST=false
            shift
            ;;
        --no-detach)
            DETACHED=false
            shift
            ;;
        --no-models)
            PULL_MODELS=false
            shift
            ;;
        --cleanup)
            CLEANUP_FIRST=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate mode
case $MODE in
    production|development|test)
        ;;
    *)
        print_error "Invalid mode: $MODE"
        print_info "Valid modes: production, development, test"
        exit 1
        ;;
esac

echo "🐳🐍 Docker + Conda Deployment Script"
echo "====================================="
echo "Mode: $MODE"
echo "Compose File: $COMPOSE_FILE"
echo "Build First: $BUILD_FIRST"
echo "Pull Models: $PULL_MODELS"
echo ""

# Check if compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    print_error "Docker compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Check Docker and Docker Compose availability
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Use docker compose or docker-compose based on availability
COMPOSE_CMD="docker-compose"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

print_status "Docker and Docker Compose detected"

# Cleanup existing containers if requested
if [[ "$CLEANUP_FIRST" == "true" ]]; then
    print_info "Cleaning up existing containers..."
    $COMPOSE_CMD -f $COMPOSE_FILE down --volumes --remove-orphans || true
    docker system prune -f || true
    print_status "Cleanup completed"
fi

# Build images if requested
if [[ "$BUILD_FIRST" == "true" ]]; then
    print_info "Building Docker images..."
    
    case $MODE in
        production)
            ./docker-build-conda.sh --type production
            ;;
        development)
            ./docker-build-conda.sh --type development
            ;;
        test)
            ./docker-build-conda.sh --type production  # Use production for testing
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        print_status "Images built successfully"
    else
        print_error "Image build failed"
        exit 1
    fi
fi

# Set up environment variables
export SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32 2>/dev/null || echo "cyber-assessment-secret-change-in-production")}

# Create data directories
print_info "Creating data directories..."
mkdir -p data/{uploads,sessions,logs,models,ollama}
print_status "Data directories created"

# Set compose profiles based on mode
PROFILES=""
case $MODE in
    development)
        PROFILES="--profile dev"
        ;;
    test)
        PROFILES="--profile test"
        ;;
esac

# Deploy services
print_info "Deploying services..."

DEPLOY_CMD="$COMPOSE_CMD -f $COMPOSE_FILE $PROFILES up"
if [[ "$DETACHED" == "true" ]]; then
    DEPLOY_CMD="$DEPLOY_CMD -d"
fi

print_info "Deploy command: $DEPLOY_CMD"

if eval $DEPLOY_CMD; then
    print_status "Services deployed successfully!"
else
    print_error "Deployment failed!"
    exit 1
fi

# Wait for services to be ready
if [[ "$DETACHED" == "true" ]]; then
    print_info "Waiting for services to be ready..."
    
    # Wait for main application
    for i in {1..30}; do
        if curl -f http://localhost:5000/system_status &> /dev/null; then
            print_status "Main application is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Wait for Ollama if models should be pulled
    if [[ "$PULL_MODELS" == "true" ]]; then
        print_info "Waiting for Ollama to be ready..."
        for i in {1..30}; do
            if curl -f http://localhost:11434/api/tags &> /dev/null; then
                print_status "Ollama is ready!"
                break
            fi
            echo -n "."
            sleep 2
        done
    fi
fi

# Show deployment status
echo ""
print_info "Deployment Status:"
$COMPOSE_CMD -f $COMPOSE_FILE $PROFILES ps

# Show service URLs
echo ""
print_status "Service URLs:"
case $MODE in
    production)
        echo "🌐 Main Application: http://localhost:5000"
        ;;
    development)
        echo "🌐 Main Application: http://localhost:5000"
        echo "🔧 Development App: http://localhost:5001"
        echo "📓 Jupyter Notebook: http://localhost:8888"
        ;;
    test)
        echo "🧪 Test Application: http://localhost:5000"
        ;;
esac

if [[ "$PULL_MODELS" == "true" ]]; then
    echo "🤖 Ollama API: http://localhost:11434"
fi

# Show logs command
echo ""
print_info "Useful commands:"
echo "📋 View logs: $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo "🔍 Check status: $COMPOSE_CMD -f $COMPOSE_FILE ps"
echo "🛑 Stop services: $COMPOSE_CMD -f $COMPOSE_FILE down"
echo "🔄 Restart: $COMPOSE_CMD -f $COMPOSE_FILE restart"

# Show resource usage
echo ""
print_info "Resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

print_status "Deployment completed successfully!"

# Optional: Run tests if in test mode
if [[ "$MODE" == "test" ]]; then
    echo ""
    print_info "Running tests..."
    $COMPOSE_CMD -f $COMPOSE_FILE --profile test up conda-test
    print_status "Tests completed!"
fi
