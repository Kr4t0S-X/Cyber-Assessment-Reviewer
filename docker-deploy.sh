#!/bin/bash
# Docker deployment script for Cyber Assessment Reviewer

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default mode
MODE="ollama"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--mode ollama|transformers]"
            echo ""
            echo "Options:"
            echo "  --mode ollama        Deploy with Ollama backend (default)"
            echo "  --mode transformers  Deploy with Transformers backend"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Deploying Cyber Assessment Reviewer"
echo "======================================"
print_status "Deployment mode: $MODE"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if images exist
if [ "$MODE" = "ollama" ]; then
    if ! docker image inspect cyber-assessment-reviewer:latest > /dev/null 2>&1; then
        print_error "Docker image not found. Please run ./docker-build.sh first"
        exit 1
    fi
elif [ "$MODE" = "transformers" ]; then
    if ! docker image inspect cyber-assessment-reviewer:transformers > /dev/null 2>&1; then
        print_error "Transformers Docker image not found. Please run ./docker-build.sh first"
        exit 1
    fi
else
    print_error "Invalid mode: $MODE. Use 'ollama' or 'transformers'"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration"
fi

# Create data directories if they don't exist
print_status "Ensuring data directories exist..."
mkdir -p data/{uploads,sessions,logs,models,ollama,transformers_cache}
chmod 755 data data/*

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down 2>/dev/null || true
docker-compose -f docker-compose.transformers.yml down 2>/dev/null || true

# Deploy based on mode
if [ "$MODE" = "ollama" ]; then
    print_status "Deploying with Ollama backend..."
    docker-compose up -d
    
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check if Ollama is healthy
    print_status "Checking Ollama health..."
    for i in {1..30}; do
        if docker-compose exec -T ollama curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
            print_success "Ollama is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Ollama failed to start properly"
            docker-compose logs ollama
            exit 1
        fi
        sleep 2
    done
    
elif [ "$MODE" = "transformers" ]; then
    print_status "Deploying with Transformers backend..."
    docker-compose -f docker-compose.transformers.yml up -d
    
    print_status "Waiting for Transformers model to load (this may take several minutes)..."
    sleep 30
fi

# Check application health
print_status "Checking application health..."
for i in {1..60}; do
    if curl -f http://localhost:5000/system_status > /dev/null 2>&1; then
        print_success "Application is healthy and ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        print_error "Application failed to start properly"
        if [ "$MODE" = "ollama" ]; then
            docker-compose logs cyber-assessment-reviewer
        else
            docker-compose -f docker-compose.transformers.yml logs cyber-assessment-reviewer
        fi
        exit 1
    fi
    sleep 5
done

# Show running containers
print_status "Running containers:"
docker ps --filter "name=cyber-assessment"

echo ""
print_success "🎉 Deployment completed successfully!"
echo ""
echo "Application is available at: http://localhost:5000"
echo ""
echo "Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart: docker-compose restart"
echo ""
if [ "$MODE" = "ollama" ]; then
    echo "Ollama API: http://localhost:11434"
    echo "Available models: docker-compose exec ollama ollama list"
fi
