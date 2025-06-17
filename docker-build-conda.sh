#!/bin/bash
# Enhanced Docker Build Script with Conda Support
# Builds optimized Docker images using conda for dependency management

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
BUILD_TYPE="production"
DOCKERFILE="Dockerfile.conda"
IMAGE_NAME="cyber-assessment-reviewer"
TAG="conda-latest"
CACHE_FROM=""
NO_CACHE=false
PARALLEL_BUILD=true
PLATFORM="linux/amd64"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Build type: production, development, minimal (default: production)"
    echo "  -f, --dockerfile FILE  Dockerfile to use (default: Dockerfile.conda)"
    echo "  -n, --name NAME        Image name (default: cyber-assessment-reviewer)"
    echo "  --tag TAG              Image tag (default: conda-latest)"
    echo "  --no-cache             Build without using cache"
    echo "  --cache-from IMAGE     Use specific image for cache"
    echo "  --platform PLATFORM   Target platform (default: linux/amd64)"
    echo "  --sequential           Build stages sequentially (slower but more reliable)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build production image"
    echo "  $0 --type development                 # Build development image"
    echo "  $0 --type minimal --no-cache          # Build minimal image without cache"
    echo "  $0 --platform linux/arm64            # Build for ARM64"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --sequential)
            PARALLEL_BUILD=false
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

# Validate build type
case $BUILD_TYPE in
    production|development|minimal)
        ;;
    *)
        print_error "Invalid build type: $BUILD_TYPE"
        print_info "Valid types: production, development, minimal"
        exit 1
        ;;
esac

# Set target based on build type
case $BUILD_TYPE in
    production)
        TARGET="production"
        FULL_TAG="${TAG}"
        ;;
    development)
        TARGET="development"
        FULL_TAG="${TAG}-dev"
        ;;
    minimal)
        TARGET="minimal-production"
        FULL_TAG="${TAG}-minimal"
        ;;
esac

echo "🐳🐍 Docker + Conda Build Script"
echo "================================"
echo "Build Type: $BUILD_TYPE"
echo "Dockerfile: $DOCKERFILE"
echo "Target: $TARGET"
echo "Image: $IMAGE_NAME:$FULL_TAG"
echo "Platform: $PLATFORM"
echo ""

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    print_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Check if environment.yml exists
if [[ ! -f "environment.yml" ]]; then
    print_error "environment.yml not found"
    print_info "Run 'python setup_environment.py --export' to create it"
    exit 1
fi

# Check Docker availability
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

# Build cache options
CACHE_OPTS=""
if [[ "$NO_CACHE" == "true" ]]; then
    CACHE_OPTS="--no-cache"
elif [[ -n "$CACHE_FROM" ]]; then
    CACHE_OPTS="--cache-from $CACHE_FROM"
fi

# Build arguments
BUILD_ARGS=""
if [[ "$PARALLEL_BUILD" == "false" ]]; then
    BUILD_ARGS="--build-arg BUILDKIT_INLINE_CACHE=1"
fi

print_info "Starting Docker build..."

# Build the image
BUILD_CMD="docker build \
    --platform $PLATFORM \
    --target $TARGET \
    --tag $IMAGE_NAME:$FULL_TAG \
    --file $DOCKERFILE \
    $CACHE_OPTS \
    $BUILD_ARGS \
    ."

print_info "Build command: $BUILD_CMD"
echo ""

# Execute build
if eval $BUILD_CMD; then
    print_status "Docker build completed successfully!"
    
    # Show image information
    echo ""
    print_info "Image Information:"
    docker images $IMAGE_NAME:$FULL_TAG --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # Test the image
    echo ""
    print_info "Testing image..."
    if docker run --rm $IMAGE_NAME:$FULL_TAG python -c "import flask, pandas, transformers; print('✅ Dependencies verified')"; then
        print_status "Image test passed!"
    else
        print_warning "Image test failed - dependencies may be missing"
    fi
    
    # Show next steps
    echo ""
    print_status "Build completed! Next steps:"
    echo "1. Run the container:"
    echo "   docker run -p 5000:5000 $IMAGE_NAME:$FULL_TAG"
    echo ""
    echo "2. Or use docker-compose:"
    echo "   docker-compose -f docker-compose.conda.yml up"
    echo ""
    echo "3. For development:"
    echo "   docker-compose -f docker-compose.conda.yml --profile dev up"
    
else
    print_error "Docker build failed!"
    exit 1
fi

# Optional: Tag additional versions
if [[ "$BUILD_TYPE" == "production" ]]; then
    print_info "Tagging as latest..."
    docker tag $IMAGE_NAME:$FULL_TAG $IMAGE_NAME:latest-conda
    print_status "Tagged as $IMAGE_NAME:latest-conda"
fi

# Optional: Show build cache usage
echo ""
print_info "Build cache information:"
docker system df

print_status "Docker build script completed successfully!"
