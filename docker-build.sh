#!/bin/bash
# Docker build script for Cyber Assessment Reviewer

set -e  # Exit on any error

echo "🐳 Building Cyber Assessment Reviewer Docker Images"
echo "=================================================="

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running ✓"

# Build production image (with Ollama support)
print_status "Building production image (Ollama mode)..."
docker build \
    --target production \
    --tag cyber-assessment-reviewer:latest \
    --tag cyber-assessment-reviewer:ollama \
    .

if [ $? -eq 0 ]; then
    print_success "Production image built successfully"
else
    print_error "Failed to build production image"
    exit 1
fi

# Build Transformers image (full dependencies)
print_status "Building Transformers image (standalone mode)..."
docker build \
    --target transformers \
    --tag cyber-assessment-reviewer:transformers \
    .

if [ $? -eq 0 ]; then
    print_success "Transformers image built successfully"
else
    print_error "Failed to build Transformers image"
    exit 1
fi

# Show built images
print_status "Built images:"
docker images | grep cyber-assessment-reviewer

# Create data directories
print_status "Creating data directories..."
mkdir -p data/{uploads,sessions,logs,models,ollama,transformers_cache}
chmod 755 data data/*

print_success "Data directories created"

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your configuration before running"
else
    print_status ".env file already exists"
fi

echo ""
print_success "🎉 Build completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run with Ollama: docker-compose up -d"
echo "3. Run with Transformers: docker-compose -f docker-compose.transformers.yml up -d"
echo ""
echo "Access the application at: http://localhost:5000"
