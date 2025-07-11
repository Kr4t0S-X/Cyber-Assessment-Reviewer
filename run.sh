#!/bin/bash
# Cyber Assessment Reviewer - Linux/macOS Run Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo "============================================================"
    echo "🛡️  Cyber Assessment Reviewer - Starting Application"
    echo "============================================================"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}💡 $1${NC}"
}

# Main function
main() {
    print_header
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found!"
        echo "📋 Please run setup first:"
        echo "   • python setup.py"
        echo "   • ./setup.sh"
        exit 1
    fi
    
    print_success "Virtual environment found"
    
    # Check if activation script exists
    if [ ! -f ".venv/bin/activate" ]; then
        print_error "Virtual environment activation script not found"
        echo "📋 Please run setup again to fix the environment"
        exit 1
    fi
    
    echo "🚀 Starting Cyber Assessment Reviewer..."
    echo "💡 Access the application at: http://localhost:5000"
    echo "🔧 Press Ctrl+C to stop the server"
    echo
    
    # Activate virtual environment and run the application
    source .venv/bin/activate
    
    # Check if main application file exists
    if [ ! -f "app.py" ]; then
        print_error "Application file (app.py) not found"
        echo "📋 Please ensure you're in the correct directory"
        exit 1
    fi
    
    # Run the application
    python app.py
}

# Trap Ctrl+C and show goodbye message
trap 'echo -e "\n\n👋 Application stopped by user\nThank you for using Cyber Assessment Reviewer!"' INT

# Run main function
main "$@"