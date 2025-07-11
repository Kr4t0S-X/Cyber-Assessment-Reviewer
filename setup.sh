#!/bin/bash
# Cyber Assessment Reviewer - Linux/macOS Setup Script

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
    echo "🛡️  Cyber Assessment Reviewer - Setup"
    echo "============================================================"
    echo
}

print_step() {
    echo -e "${BLUE}📋 Step $1/$2: $3${NC}"
}

print_success() {
    echo -e "${GREEN}   ✅ $1${NC}"
}

print_error() {
    echo -e "${RED}   ❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}   ⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}   💡 $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    print_header
    
    # Step 1: Check Python version
    print_step 1 6 "Checking Python version"
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        echo "   📋 Please install Python 3.10 or higher"
        echo "   🐧 Ubuntu/Debian: sudo apt-get install python3 python3-pip"
        echo "   🍎 macOS: brew install python3"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    REQUIRED_VERSION="3.10"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_error "Python $PYTHON_VERSION is not supported"
        echo "   📋 This application requires Python 3.10 or higher"
        exit 1
    fi
    
    print_success "Python $PYTHON_VERSION is compatible"
    
    # Step 2: Install uv
    print_step 2 6 "Installing uv (ultra-fast Python package manager)"
    
    if command_exists uv; then
        print_success "uv is already installed"
    else
        echo "   🔧 Installing uv..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            print_success "uv installed successfully"
            # Add uv to PATH for current session
            export PATH="$HOME/.cargo/bin:$PATH"
        else
            print_error "Failed to install uv"
            echo "   📋 Please install uv manually: https://github.com/astral-sh/uv"
            exit 1
        fi
    fi
    
    # Step 3: Create virtual environment
    print_step 3 6 "Creating virtual environment"
    
    if [ -d ".venv" ]; then
        print_success "Virtual environment already exists"
    else
        echo "   🔧 Creating virtual environment..."
        if uv venv; then
            print_success "Virtual environment created successfully"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    fi
    
    # Step 4: Install dependencies
    print_step 4 6 "Installing project dependencies"
    
    echo "   🔧 Installing dependencies with uv..."
    if uv pip install -e .; then
        print_success "Dependencies installed successfully"
    else
        print_warning "uv installation failed, trying pip fallback..."
        source .venv/bin/activate
        if pip install -e .; then
            print_success "Dependencies installed with pip (fallback)"
        else
            print_error "Failed to install dependencies"
            exit 1
        fi
    fi
    
    # Step 5: Verify installation
    print_step 5 6 "Verifying installation"
    
    source .venv/bin/activate
    if python3 -c "import flask, pandas, transformers; print('✅ Core packages imported successfully')"; then
        print_success "Installation verified successfully"
    else
        print_error "Installation verification failed"
        exit 1
    fi
    
    # Step 6: Show completion message
    print_step 6 6 "Setup completed successfully!"
    echo
    echo "🎉 Installation Complete!"
    echo "============================================================"
    echo
    echo "📋 Next Steps:"
    echo "   1. Run the application:"
    echo "      • ./run.sh  (or python run.py)"
    echo
    echo "   2. Access the web interface:"
    echo "      • Open your browser to: http://localhost:5000"
    echo
    echo "💡 Tips:"
    echo "   • For better performance, install Ollama: https://ollama.com"
    echo "   • First run may take longer as models download"
    echo "   • Check README.md for detailed usage instructions"
    echo
    echo "🛡️  Ready to analyze cybersecurity assessments!"
    echo "============================================================"
}

# Run main function
main "$@"