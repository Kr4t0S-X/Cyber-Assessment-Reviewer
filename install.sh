#!/usr/bin/env bash
set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR=".venv"
PYTHON_VERSION="3.10.11"  # Configurable via environment variable
UV_VERSION="0.4.0"        # Minimum required version

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

print_header() {
    echo
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}🛡️  Cyber Assessment Reviewer - Advanced Installation${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Compare version numbers
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Install uv if not present
install_uv() {
    print_info "Installing uv package manager..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            print_info "Using Homebrew to install uv..."
            if brew install uv; then
                print_status "uv installed via Homebrew"
            else
                print_warning "Homebrew installation failed, trying curl..."
                curl -LsSf https://astral.sh/uv/install.sh | sh
            fi
        else
            print_info "Installing uv via curl..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        print_info "Installing uv via curl..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        print_error "Unsupported OS: $OSTYPE"
        echo "Please install uv manually from: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    # Add to PATH if necessary
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Verify installation
    if command_exists uv; then
        local installed_version=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
        print_status "uv $installed_version installed successfully"
    else
        print_error "uv installation failed"
        exit 1
    fi
}

# Check uv installation
check_uv() {
    if ! command_exists uv; then
        print_warning "uv not found. Installing..."
        install_uv
    else
        # Check uv version
        local current_version=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0")
        if ! version_ge "$current_version" "$UV_VERSION"; then
            print_warning "uv version $current_version is older than required $UV_VERSION. Updating..."
            install_uv
        else
            print_status "uv $current_version is already installed"
        fi
    fi
}

# Check if virtual environment exists and is valid
check_venv() {
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        print_info "Virtual environment found at $VENV_DIR"
        
        # Verify it's a valid Python environment
        if ! "$VENV_DIR/bin/python" --version &>/dev/null; then
            print_warning "Virtual environment appears corrupted. Recreating..."
            rm -rf "$VENV_DIR"
            return 1
        fi
        
        # Check Python version in venv
        local venv_python_version=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        local required_python_major_minor=$(echo $PYTHON_VERSION | grep -oE '[0-9]+\.[0-9]+')
        local venv_python_major_minor=$(echo $venv_python_version | grep -oE '[0-9]+\.[0-9]+')
        
        if [ "$venv_python_major_minor" != "$required_python_major_minor" ]; then
            print_warning "Virtual environment has Python $venv_python_version, but Python $required_python_major_minor.x is required. Recreating..."
            rm -rf "$VENV_DIR"
            return 1
        fi
        
        print_status "Virtual environment is valid with Python $venv_python_version"
        return 0
    else
        print_info "No virtual environment found"
        return 1
    fi
}

# Create virtual environment using uv
create_venv() {
    print_info "Creating virtual environment with Python $PYTHON_VERSION..."
    
    # Try with specific Python version first
    if uv venv "$VENV_DIR" --python "$PYTHON_VERSION"; then
        print_status "Virtual environment created with Python $PYTHON_VERSION"
        return 0
    fi
    
    # Fallback to Python 3.10
    print_warning "Specific version $PYTHON_VERSION not found, trying Python 3.10..."
    if uv venv "$VENV_DIR" --python python3.10; then
        print_status "Virtual environment created with Python 3.10"
        return 0
    fi
    
    # Final fallback to system Python
    print_warning "Python 3.10 not found, trying system Python..."
    if uv venv "$VENV_DIR"; then
        local system_python_version=$("$VENV_DIR/bin/python" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        print_status "Virtual environment created with system Python $system_python_version"
        
        # Warn if system Python is too old
        if ! version_ge "$system_python_version" "3.10.0"; then
            print_warning "System Python $system_python_version is older than recommended 3.10+"
            print_warning "Some features may not work correctly"
        fi
        return 0
    fi
    
    print_error "Failed to create virtual environment with any Python version"
    exit 1
}

# Activate virtual environment
activate_venv() {
    # This doesn't actually activate for the current shell, 
    # but ensures uv uses the venv
    export VIRTUAL_ENV="$PWD/$VENV_DIR"
    export UV_PROJECT_ENVIRONMENT="$PWD/$VENV_DIR"
    print_status "Virtual environment configured for uv"
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies in virtual environment..."
    
    # Check for various dependency files
    if [ -f "pyproject.toml" ]; then
        print_info "Found pyproject.toml"
        
        # Try to install with dev dependencies first
        print_info "Attempting installation with dev dependencies..."
        if uv pip install -e ".[dev]" 2>/dev/null; then
            print_status "Dependencies installed from pyproject.toml (with dev extras)"
        else
            print_warning "Dev dependencies installation failed, trying without dev extras..."
            # Try without dev dependencies
            if uv pip install -e "." 2>/dev/null; then
                print_status "Dependencies installed from pyproject.toml"
            else
                print_error "pyproject.toml installation failed. Common causes:"
                print_error "  • Build configuration issues (check [tool.hatch.build] section)"
                print_error "  • Dependency conflicts (check NumPy/pandas versions)"
                print_error "  • Missing system dependencies"
                print_warning "Trying requirements.txt as fallback..."
                
                if [ -f "requirements.txt" ] && uv pip install -r requirements.txt; then
                    print_status "Dependencies installed from requirements.txt (fallback)"
                else
                    print_error "All dependency installation methods failed"
                    print_info "Please check the error messages above and:"
                    print_info "  • Ensure Python version compatibility"
                    print_info "  • Install system dependencies (build-essential, python3-dev)"
                    print_info "  • Check network connectivity"
                    exit 1
                fi
            fi
        fi
        
    elif [ -f "requirements.txt" ]; then
        print_info "Found requirements.txt"
        if uv pip install -r requirements.txt; then
            print_status "Dependencies installed from requirements.txt"
        else
            print_error "Failed to install dependencies from requirements.txt"
            exit 1
        fi
        
        # Also check for dev requirements
        if [ -f "requirements-dev.txt" ]; then
            print_info "Installing dev requirements..."
            if uv pip install -r requirements-dev.txt; then
                print_status "Dev dependencies installed"
            else
                print_warning "Failed to install dev dependencies (non-critical)"
            fi
        fi
        
    elif [ -f "Pipfile" ]; then
        print_warning "Found Pipfile. Converting to requirements.txt..."
        # Check if pipfile-requirements is available
        if command_exists pipfile-requirements; then
            pipfile-requirements > requirements.txt
            uv pip install -r requirements.txt
            print_status "Dependencies installed from converted Pipfile"
        else
            print_warning "pipfile-requirements not found. Installing it first..."
            uv pip install pipfile-requirements
            pipfile-requirements > requirements.txt
            uv pip install -r requirements.txt
            print_status "Dependencies installed from converted Pipfile"
        fi
        
    else
        print_warning "No dependency file found (pyproject.toml, requirements.txt, or Pipfile)"
        print_info "Skipping dependency installation"
        print_info "You may need to install dependencies manually later"
    fi
    
    # Install common dev tools if not already present
    print_info "Installing essential development tools..."
    local dev_tools="pip setuptools wheel"
    for tool in $dev_tools; do
        if ! "$VENV_DIR/bin/python" -c "import $tool" &>/dev/null; then
            uv pip install --upgrade "$tool" 2>/dev/null || true
        fi
    done
    
    # Optional dev tools (fail silently)
    print_info "Installing optional development tools..."
    local optional_tools="black flake8 pytest mypy pre-commit"
    for tool in $optional_tools; do
        uv pip install "$tool" 2>/dev/null || true
    done
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test basic Python functionality
    if ! "$VENV_DIR/bin/python" -c "print('Python is working')" &>/dev/null; then
        print_error "Python installation verification failed"
        return 1
    fi
    
    # Test key imports based on what's likely to be in the project
    local test_imports=""
    if [ -f "pyproject.toml" ] && grep -q "flask" pyproject.toml; then
        test_imports="$test_imports flask"
    fi
    if [ -f "pyproject.toml" ] && grep -q "pandas" pyproject.toml; then
        test_imports="$test_imports pandas"
    fi
    if [ -f "requirements.txt" ] && grep -q "flask" requirements.txt; then
        test_imports="$test_imports flask"
    fi
    if [ -f "requirements.txt" ] && grep -q "pandas" requirements.txt; then
        test_imports="$test_imports pandas"
    fi
    
    if [ -n "$test_imports" ]; then
        local import_test=""
        for package in $test_imports; do
            import_test="$import_test import $package;"
        done
        
        if "$VENV_DIR/bin/python" -c "$import_test print('✅ Key packages imported successfully')" 2>/dev/null; then
            print_status "Installation verified - key packages working"
            
            # Show versions of installed packages
            print_info "Installed package versions:"
            for package in $test_imports; do
                local version=$("$VENV_DIR/bin/python" -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
                echo "   • $package: $version"
            done
        else
            print_warning "Some packages may not have installed correctly"
            print_info "This might not be critical if you haven't defined dependencies yet"
        fi
    else
        print_info "No specific package requirements found - basic Python environment ready"
    fi
    
    return 0
}

# Setup git hooks
setup_git_hooks() {
    if [ -f ".pre-commit-config.yaml" ] && command_exists git; then
        print_info "Setting up pre-commit hooks..."
        if "$VENV_DIR/bin/pre-commit" install 2>/dev/null; then
            print_status "Pre-commit hooks installed"
        else
            print_warning "Failed to install pre-commit hooks (non-critical)"
        fi
    fi
}

# Show completion message
show_completion() {
    echo
    echo -e "${GREEN}🎉 Installation Complete!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo
    echo -e "${GREEN}📋 Next Steps:${NC}"
    echo "   1. Activate the virtual environment:"
    echo -e "      ${BLUE}source $VENV_DIR/bin/activate${NC}"
    echo
    echo "   2. Start development:"
    if [ -f "app.py" ] || [ -f "main.py" ] || [ -f "run.py" ]; then
        echo "      • Run your application"
    else
        echo "      • Create your Python application files"
    fi
    echo
    echo -e "${GREEN}💡 Tips:${NC}"
    echo "   • Virtual environment created at: $VENV_DIR"
    echo "   • Python version: $("$VENV_DIR/bin/python" --version 2>&1)"
    echo "   • To deactivate later: deactivate"
    echo "   • To remove environment: rm -rf $VENV_DIR"
    echo
    echo -e "${GREEN}🛡️  Environment ready for development!${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# Main execution
main() {
    print_header
    
    # Allow custom configuration via environment variables
    VENV_DIR="${VENV_DIR:-$VENV_DIR}"
    PYTHON_VERSION="${PYTHON_VERSION:-$PYTHON_VERSION}"
    
    print_info "Starting Python environment setup..."
    print_info "Target Python version: $PYTHON_VERSION"
    print_info "Virtual environment location: $VENV_DIR"
    echo
    
    # Step 1: Check/Install uv
    print_info "🔧 Step 1: Checking uv package manager..."
    check_uv
    echo
    
    # Step 2: Check if virtual environment exists
    print_info "🔧 Step 2: Checking virtual environment..."
    if ! check_venv; then
        # Step 3: Create virtual environment
        print_info "🔧 Step 3: Creating virtual environment..."
        create_venv
    else
        print_info "🔧 Step 3: Virtual environment already exists and is valid"
    fi
    echo
    
    # Step 4: Configure environment for uv
    print_info "🔧 Step 4: Configuring environment..."
    activate_venv
    echo
    
    # Step 5: Install dependencies
    print_info "🔧 Step 5: Installing dependencies..."
    install_dependencies
    echo
    
    # Step 6: Verify installation
    print_info "🔧 Step 6: Verifying installation..."
    verify_installation
    echo
    
    # Step 7: Setup additional tools
    print_info "🔧 Step 7: Setting up development tools..."
    setup_git_hooks
    echo
    
    # Step 8: Show completion message
    show_completion
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Cyber Assessment Reviewer - Advanced Installation Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --version           Show version information"
        echo ""
        echo "Environment Variables:"
        echo "  VENV_DIR           Virtual environment directory (default: .venv)"
        echo "  PYTHON_VERSION     Python version to use (default: 3.10.11)"
        echo ""
        echo "Examples:"
        echo "  $0                              # Standard installation"
        echo "  PYTHON_VERSION=3.11.0 $0       # Use specific Python version"
        echo "  VENV_DIR=myenv $0               # Use custom environment name"
        exit 0
        ;;
    --version)
        echo "Cyber Assessment Reviewer Installation Script v1.0"
        echo "uv-based Python environment setup with advanced error handling"
        exit 0
        ;;
    --*)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

# Trap errors and cleanup
trap 'print_error "Installation interrupted or failed"' ERR
trap 'print_warning "Installation interrupted by user"' INT

# Run main function
main "$@"