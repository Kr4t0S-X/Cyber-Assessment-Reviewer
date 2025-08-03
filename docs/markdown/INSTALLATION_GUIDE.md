# 🐍 Installation Guide - Cyber Assessment Reviewer

This guide provides comprehensive installation instructions for the Cyber Assessment Reviewer application with support for both **Anaconda/Miniconda** and **pip-based** environments.

## 📋 Prerequisites

- **Python 3.10.11** (preferred) or **Python 3.10+**
- **Git** (for cloning the repository)
- **Internet connection** (for downloading uv and dependencies)
- **uv** (automatically installed by setup scripts) or **Anaconda/Miniconda** (alternative)

## 🚀 Quick Start

### Option 1: Enhanced Setup Scripts (Recommended)

#### Linux/macOS with uv
```bash
# Clone the repository
git clone https://github.com/Kr4t0S-X/Cyber-Assessment-Reviewer.git
cd Cyber-Assessment-Reviewer

# Run enhanced setup (installs uv + Python 3.10.11 + dependencies)
./setup.sh

# Or use the advanced installation script
./install.sh
```

#### Windows with uv
```powershell
# Clone the repository
git clone https://github.com/Kr4t0S-X/Cyber-Assessment-Reviewer.git
cd Cyber-Assessment-Reviewer

# Run PowerShell installation script
.\install.ps1
```

#### Cross-Platform Python Setup
```bash
# Alternative: Python-based setup
python setup.py
```

### Option 2: Platform-Specific Scripts (Legacy)

#### Windows (Conda)
```cmd
setup-conda.bat
```

#### Linux/macOS (Conda)
```bash
./setup-conda.sh
```

## ⚡ uv-Based Installation (Recommended)

### Why Use uv?

- **10x faster** dependency resolution compared to pip
- **Better conflict resolution** for complex ML/AI dependencies
- **Cross-platform consistency** with locked versions
- **Lightweight and fast** installation process
- **Automatic virtual environment management**

### Features of Enhanced Setup Scripts

#### setup.sh / setup.py / install.sh / install.ps1

- **Python 3.10.11 target version** with fallback support
- **Automatic uv installation** and version checking
- **Smart virtual environment detection** and validation
- **Corrupted environment recovery** - automatically recreates broken environments
- **Multiple dependency file support** (pyproject.toml priority, requirements.txt fallback)
- **Cross-platform compatibility** (Linux, macOS, Windows)
- **Comprehensive error handling** and helpful error messages
- **Installation verification** with package version reporting

### Installation Process

1. **uv Installation**: Automatically downloads and installs uv package manager
2. **Version Validation**: Ensures uv >= 0.4.0 for optimal performance
3. **Virtual Environment**: Creates `.venv` with Python 3.10.11 (or best available)
4. **Environment Health Check**: Validates existing environments, recreates if corrupted
5. **Dependency Installation**: Uses pyproject.toml (preferred) or requirements.txt
6. **Verification**: Tests core package imports and displays versions
7. **Setup Completion**: Provides clear next steps and activation instructions

### Usage Examples

#### Basic Installation
```bash
# Linux/macOS
./setup.sh

# Windows
.\install.ps1
```

#### Advanced Installation with Custom Settings
```bash
# Custom virtual environment directory
VENV_DIR="my-env" ./install.sh

# Specific Python version (if available)
PYTHON_VERSION="3.10.9" ./install.sh

# Help and options
./install.sh --help
.\install.ps1 -Help
```

#### Manual Activation After Installation
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## 🐍 Anaconda/Miniconda Installation (Alternative)

### Why Use Conda?

- **Better dependency management** and conflict resolution
- **Isolated environments** prevent package conflicts
- **Cross-platform compatibility** (Windows, macOS, Linux)
- **Scientific package optimization** for ML/AI libraries
- **Easy environment sharing** and reproduction

### Step 1: Install Anaconda or Miniconda

#### Anaconda (Full Distribution)
- **Download**: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- **Size**: ~500MB download, ~3GB installed
- **Includes**: Python + 250+ pre-installed packages + Anaconda Navigator

#### Miniconda (Minimal Distribution)
- **Download**: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Size**: ~50MB download, ~200MB installed
- **Includes**: Python + conda package manager only

### Step 2: Verify Conda Installation

```bash
# Check conda version
conda --version

# Check conda info
conda info
```

### Step 3: Create Environment

#### Method A: Using environment.yml (Recommended)
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate cyber-assessment-env
```

#### Method B: Manual Creation
```bash
# Create new environment
conda create -n cyber-assessment-env python=3.10 -y

# Activate environment
conda activate cyber-assessment-env

# Install dependencies
conda install -c conda-forge flask pandas numpy requests python-docx openpyxl pypdf2 python-pptx transformers torch scikit-learn matplotlib seaborn jupyter ipykernel -y

# Install pip-only packages
pip install ollama
```

### Step 4: Verify Installation

```bash
# Test key imports
python -c "import flask, pandas, transformers; print('✅ Installation verified')"

# Run application
python app.py
```

## 📦 Pip Installation (Alternative)

### Step 1: Create Virtual Environment

#### Windows
```cmd
# Create virtual environment
python -m venv cyber-assessment-env

# Activate environment
cyber-assessment-env\Scripts\activate
```

#### Linux/macOS
```bash
# Create virtual environment
python -m venv cyber-assessment-env

# Activate environment
source cyber-assessment-env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "import flask, pandas, transformers; print('✅ Installation verified')"

# Run application
python app.py
```

## 🔧 Environment Management

### Conda Commands

```bash
# List environments
conda env list

# Activate environment
conda activate cyber-assessment-env

# Deactivate environment
conda deactivate

# Export environment
conda env export -n cyber-assessment-env -f environment.yml

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n cyber-assessment-env
```

### Pip/Venv Commands

```bash
# Activate environment (Windows)
cyber-assessment-env\Scripts\activate

# Activate environment (Linux/macOS)
source cyber-assessment-env/bin/activate

# Deactivate environment
deactivate

# Export requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## 🛠️ Advanced Setup Options

### Custom Environment Name

```bash
# Using setup script with custom name
python setup_environment.py --env-name my-custom-env

# Using conda directly
conda create -n my-custom-env python=3.10
```

### Force Pip Installation

```bash
# Skip conda detection and use pip
python setup_environment.py --pip
```

### Development Setup

```bash
# Install development dependencies
conda install -c conda-forge pytest black flake8 mypy

# Or with pip
pip install pytest black flake8 mypy
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Conda Not Found
**Problem**: `conda: command not found`

**Solutions**:
- Restart terminal after Anaconda/Miniconda installation
- Add conda to PATH manually
- Use Anaconda Prompt on Windows

#### 2. Environment Creation Fails
**Problem**: `CondaError: environment already exists`

**Solutions**:
```bash
# Remove existing environment
conda env remove -n cyber-assessment-env

# Or use different name
conda create -n cyber-assessment-env-v2 python=3.10
```

#### 3. Package Installation Fails
**Problem**: Some packages fail to install

**Solutions**:
```bash
# Try different channels
conda install -c conda-forge -c defaults package-name

# Use pip fallback
pip install package-name

# Update conda
conda update conda
```

#### 4. Import Errors
**Problem**: `ModuleNotFoundError` when running application

**Solutions**:
```bash
# Verify environment is activated
conda info --envs

# Reinstall problematic package
conda install --force-reinstall package-name

# Check Python path
python -c "import sys; print(sys.path)"
```

### Environment Detection Script

```bash
# Check environment status
python setup_environment.py --info
```

### Clean Installation

```bash
# Remove environment and start fresh
python setup_environment.py --clean
python setup_environment.py
```

## 📊 Performance Comparison

| Method | Setup Time | Disk Space | Dependency Resolution | Isolation | Speed |
|--------|------------|------------|---------------------|-----------|-------|
| **uv + venv** | 1-3 min | ~300MB | Excellent | Excellent | 10x faster |
| **Conda** | 5-10 min | ~2GB | Excellent | Excellent | Standard |
| **Pip + venv** | 2-5 min | ~500MB | Good | Good | Standard |
| **Global pip** | 1-2 min | ~200MB | Poor | None | Standard |

### Setup Script Comparison

| Script | Platform | Features | Best For |
|--------|----------|----------|----------|
| **setup.sh** | Linux/macOS | Enhanced uv integration, health checks | Production setup |
| **install.sh** | Linux/macOS | Advanced error handling, color output | Development setup |
| **install.ps1** | Windows | PowerShell native, full feature parity | Windows development |
| **setup.py** | Cross-platform | Python-based, consistent behavior | CI/CD pipelines |

## 🎯 Recommended Workflows

### For Development (Recommended)
```bash
# Use uv for fast, reliable development setup
./install.sh  # Linux/macOS
# or
.\install.ps1  # Windows

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### For Production
```bash
# Use enhanced setup for production deployment
./setup.sh  # Optimized for production use
# or
python setup.py  # For automated deployment scripts
```

### For CI/CD
```bash
# Use Python-based setup for consistency
python setup.py
source .venv/bin/activate
python -m pytest  # Run tests
```

### For Conda Users (Alternative)
```bash
# Use conda for better dependency management
conda create -n cyber-assessment-dev python=3.10
conda activate cyber-assessment-dev
conda install -c conda-forge --file requirements-dev.txt
```

## 📚 Additional Resources

- **uv Documentation**: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **Python 3.10.11 Downloads**: [https://www.python.org/downloads/release/python-31011/](https://www.python.org/downloads/release/python-31011/)
- **Virtual Environments Guide**: [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)
- **Conda Documentation**: [https://docs.conda.io/](https://docs.conda.io/)
- **Package Management Best Practices**: [https://packaging.python.org/](https://packaging.python.org/)

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs**: Look for error messages in the terminal output
2. **Verify prerequisites**: Ensure Python 3.10+ is installed
3. **Try clean installation**: Remove environment and recreate
4. **Check system resources**: Ensure sufficient disk space and memory
5. **Consult documentation**: Review conda/pip documentation for specific errors

## 🎉 Next Steps

After successful installation:

1. **Activate your environment**
2. **Run the application**: `python app.py`
3. **Access the web interface**: Open `http://localhost:5000`
4. **Run tests**: `python test_ai_accuracy.py`
5. **Explore features**: Upload controls and evidence files

The Cyber Assessment Reviewer is now ready to help you with cybersecurity compliance assessments! 🛡️✨
