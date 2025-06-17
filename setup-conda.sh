#!/bin/bash
# Conda Environment Setup Script for Unix/Linux/macOS
# Cyber Assessment Reviewer

set -e  # Exit on any error

echo "========================================"
echo "Cyber Assessment Reviewer - Conda Setup"
echo "========================================"
echo

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

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda not found in PATH"
    echo
    echo "Please install Anaconda or Miniconda:"
    echo "- Anaconda: https://www.anaconda.com/products/distribution"
    echo "- Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

print_status "Conda detected"
conda --version
echo

# Initialize conda for shell (if not already done)
if [[ "$SHELL" == *"bash"* ]]; then
    conda init bash >/dev/null 2>&1 || true
elif [[ "$SHELL" == *"zsh"* ]]; then
    conda init zsh >/dev/null 2>&1 || true
fi

# Source conda to make sure it's available
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

# Check if environment already exists
if conda env list | grep -q "cyber-assessment-env"; then
    print_warning "Environment 'cyber-assessment-env' already exists"
    echo
    read -p "Do you want to remove and recreate it? (y/N): " choice
    case "$choice" in 
        y|Y ) 
            echo "Removing existing environment..."
            conda env remove -n cyber-assessment-env -y
            print_status "Environment removed"
            ;;
        * ) 
            echo "Using existing environment"
            ;;
    esac
fi

# Create environment
if [[ ! $(conda env list | grep "cyber-assessment-env") ]]; then
    echo "Creating conda environment..."
    
    if [[ -f "environment.yml" ]]; then
        echo "Creating environment from environment.yml..."
        if conda env create -f environment.yml; then
            print_status "Environment created from environment.yml"
        else
            print_warning "Failed to create from environment.yml, trying manual creation..."
            create_manual=true
        fi
    else
        print_warning "environment.yml not found, creating environment manually..."
        create_manual=true
    fi
    
    if [[ "$create_manual" == "true" ]]; then
        echo "Creating environment manually..."
        conda create -n cyber-assessment-env python=3.10 -y
        
        echo "Installing conda packages..."
        conda install -n cyber-assessment-env -c conda-forge \
            flask pandas numpy requests python-docx openpyxl pypdf2 \
            python-pptx transformers torch scikit-learn matplotlib \
            seaborn jupyter ipykernel -y || print_warning "Some conda packages failed to install"
        
        echo "Installing pip dependencies..."
        conda run -n cyber-assessment-env pip install ollama || print_warning "Some pip packages failed to install"
    fi
    
    print_status "Environment setup complete"
fi

echo
echo "========================================"
echo "🎉 Setup Complete!"
echo "========================================"
echo
echo "To use the Cyber Assessment Reviewer:"
echo
echo "1. Activate the environment:"
echo "   conda activate cyber-assessment-env"
echo
echo "2. Run the application:"
echo "   python app.py"
echo
echo "3. Or run tests:"
echo "   python test_ai_accuracy.py"
echo
echo "To deactivate the environment:"
echo "   conda deactivate"
echo
echo "Environment management:"
echo "- List environments: conda env list"
echo "- Export environment: conda env export -n cyber-assessment-env -f environment.yml"
echo "- Remove environment: conda env remove -n cyber-assessment-env"
echo

# Ask if user wants to activate environment
read -p "Do you want to activate the environment now? (Y/n): " choice
case "$choice" in 
    n|N ) 
        echo
        echo "Remember to activate the environment before using the application:"
        echo "conda activate cyber-assessment-env"
        ;;
    * ) 
        echo
        echo "Activating environment..."
        echo "Note: You may need to run 'conda activate cyber-assessment-env' manually"
        echo "      if this doesn't work in your current shell."
        echo
        
        # Try to activate (may not work in all shells)
        conda activate cyber-assessment-env 2>/dev/null || {
            echo "Please run: conda activate cyber-assessment-env"
        }
        ;;
esac

echo
print_status "Setup script completed!"
