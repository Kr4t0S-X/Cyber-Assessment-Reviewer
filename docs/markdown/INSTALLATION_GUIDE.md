# 🐍 Installation Guide - Cyber Assessment Reviewer

This guide provides comprehensive installation instructions for the Cyber Assessment Reviewer application with support for both **Anaconda/Miniconda** and **pip-based** environments.

## 📋 Prerequisites

- **Python 3.10 or higher**
- **Git** (for cloning the repository)
- **Anaconda/Miniconda** (recommended) or **pip** with virtual environment support

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/Kr4t0S-X/Cyber-Assessment-Reviewer.git
cd Cyber-Assessment-Reviewer

# Run automated setup
python setup_environment.py
```

### Option 2: Platform-Specific Scripts

#### Windows (Conda)
```cmd
setup-conda.bat
```

#### Linux/macOS (Conda)
```bash
./setup-conda.sh
```

## 🐍 Anaconda/Miniconda Installation (Recommended)

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

| Method | Setup Time | Disk Space | Dependency Resolution | Isolation |
|--------|------------|------------|---------------------|-----------|
| **Conda** | 5-10 min | ~2GB | Excellent | Excellent |
| **Pip + venv** | 2-5 min | ~500MB | Good | Good |
| **Global pip** | 1-2 min | ~200MB | Poor | None |

## 🎯 Recommended Workflows

### For Development
```bash
# Use conda for better dependency management
conda create -n cyber-assessment-dev python=3.10
conda activate cyber-assessment-dev
conda install -c conda-forge --file requirements-dev.txt
```

### For Production
```bash
# Use conda with locked versions
conda env create -f environment-prod.yml
conda activate cyber-assessment-prod
```

### For Testing
```bash
# Use pip for lightweight testing
python -m venv test-env
source test-env/bin/activate  # or test-env\Scripts\activate on Windows
pip install -r requirements.txt
```

## 📚 Additional Resources

- **Conda Documentation**: [https://docs.conda.io/](https://docs.conda.io/)
- **Virtual Environments Guide**: [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)
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
