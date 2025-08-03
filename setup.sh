#!/usr/bin/env bash
# Cyber Assessment Reviewer - Enhanced Linux/macOS Setup Script

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.10.11"  # Exact Python version required
UV_MIN_VERSION="0.4.0"    # Minimum uv version required

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

# Compare version numbers (returns 0 if $1 >= $2)
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Get uv version
get_uv_version() {
    uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0"
}

# Check if virtual environment is valid and has correct Python version
check_venv_health() {
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        # Check if Python executable exists and works
        if ! "$VENV_DIR/bin/python" --version &>/dev/null; then
            print_warning "Virtual environment appears corrupted"
            return 1
        fi
        
        # Check Python version in venv
        local venv_python_version=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        if [ "$venv_python_version" != "$PYTHON_VERSION" ]; then
            print_warning "Virtual environment has Python $venv_python_version, but $PYTHON_VERSION is required"
            return 1
        fi
        
        print_success "Virtual environment is healthy with Python $venv_python_version"
        return 0
    else
        return 1
    fi
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
    
    # Step 2: Install/Update uv
    print_step 2 6 "Installing/Updating uv (ultra-fast Python package manager)"
    
    if command_exists uv; then
        local current_version=$(get_uv_version)
        if version_ge "$current_version" "$UV_MIN_VERSION"; then
            print_success "uv $current_version is already installed and up to date"
        else
            print_warning "uv $current_version is older than required $UV_MIN_VERSION. Updating..."
            echo "   🔧 Updating uv..."
            if curl -LsSf https://astral.sh/uv/install.sh | sh; then
                print_success "uv updated successfully"
                export PATH="$HOME/.cargo/bin:$PATH"
            else
                print_error "Failed to update uv"
                exit 1
            fi
        fi
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
    
    # Step 3: Create/Validate virtual environment
    print_step 3 6 "Creating/Validating virtual environment"
    
    if check_venv_health; then
        print_success "Virtual environment is ready"
    else
        if [ -d "$VENV_DIR" ]; then
            print_warning "Removing corrupted virtual environment..."
            rm -rf "$VENV_DIR"
        fi
        
        echo "   🔧 Creating virtual environment with Python $PYTHON_VERSION..."
        if uv venv "$VENV_DIR" --python "$PYTHON_VERSION"; then
            print_success "Virtual environment created successfully"
        elif uv venv "$VENV_DIR" --python python3.10; then
            print_success "Virtual environment created with fallback Python 3.10"
        elif uv venv "$VENV_DIR"; then
            print_success "Virtual environment created with system Python"
        else
            print_error "Failed to create virtual environment"
            exit 1
        fi
    fi
    
    # Step 4: Install dependencies
    print_step 4 6 "Installing project dependencies"
    
    # Configure uv to use the virtual environment
    export VIRTUAL_ENV="$PWD/$VENV_DIR"
    export UV_PROJECT_ENVIRONMENT="$PWD/$VENV_DIR"
    
    echo "   🔧 Installing dependencies with uv..."
    if [ -f "pyproject.toml" ]; then
        print_info "Using pyproject.toml for dependencies"
        
        # Try with dev dependencies first
        echo "   🔧 Attempting installation with dev dependencies..."
        if uv pip install -e ".[dev]" 2>/dev/null; then
            print_success "Dependencies installed from pyproject.toml (with dev extras)"
        else
            print_warning "Dev dependencies installation failed, trying without dev extras..."
            # Try without dev dependencies
            if uv pip install -e . 2>/dev/null; then
                print_success "Dependencies installed from pyproject.toml"
            else
                print_error "pyproject.toml installation failed. This may be due to:"
                print_error "  • Build configuration issues"
                print_error "  • Dependency conflicts" 
                print_error "  • Missing system dependencies"
                print_warning "Trying requirements.txt as fallback..."
                
                if [ -f "requirements.txt" ] && uv pip install -r requirements.txt; then
                    print_success "Dependencies installed from requirements.txt (fallback)"
                else
                    print_error "All dependency installation methods failed"
                    print_error "Please check:"
                    print_error "  • Python version compatibility"
                    print_error "  • System dependencies (gcc, python3-dev, etc.)"
                    print_error "  • Network connectivity"
                    exit 1
                fi
            fi
        fi
    elif [ -f "requirements.txt" ]; then
        print_info "Using requirements.txt for dependencies"
        if uv pip install -r requirements.txt; then
            print_success "Dependencies installed from requirements.txt"
        else
            print_error "Failed to install dependencies"
            exit 1
        fi
    else
        print_error "No dependency file found (pyproject.toml or requirements.txt)"
        exit 1
    fi
    
    # Step 5: Verify installation
    print_step 5 6 "Verifying installation"
    
    echo "   🔧 Testing core package imports..."
    if "$VENV_DIR/bin/python" -c "import flask, pandas, transformers; print('✅ Core packages imported successfully')"; then
        print_success "Installation verified successfully"
        
        # Show versions of key packages
        echo "   📦 Installed package versions:"
        "$VENV_DIR/bin/python" -c "
import flask, pandas, transformers
print(f'   • Flask: {flask.__version__}')
print(f'   • Pandas: {pandas.__version__}')
print(f'   • Transformers: {transformers.__version__}')
"
    else
        print_error "Installation verification failed"
        print_info "Some packages may not have installed correctly"
        exit 1
    fi
    
    # Step 6: Show completion message
    print_step 6 6 "Setup completed successfully!"
    echo
    echo "🎉 Installation Complete!"
    echo "============================================================"
    echo
    echo "📋 Next Steps:"
    echo "   1. Activate the virtual environment:"
    echo "      • source $VENV_DIR/bin/activate"
    echo
    echo "   2. Run the application:"
    echo "      • ./run.sh  (or python run.py)"
    echo
    echo "   3. Access the web interface:"
    echo "      • Open your browser to: http://localhost:5000"
    echo
    echo "💡 Tips:"
    echo "   • Virtual environment with Python $PYTHON_VERSION is ready"
    echo "   • For better performance, install Ollama: https://ollama.com"
    echo "   • First run may take longer as models download"
    echo "   • Check README.md for detailed usage instructions"
    echo
    echo "🛡️  Ready to analyze cybersecurity assessments!"
    echo "============================================================"
}

# Run main function
main "$@"