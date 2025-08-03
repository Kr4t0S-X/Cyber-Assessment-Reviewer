#!/usr/bin/env python3
"""
Cyber Assessment Reviewer - One-Command Setup
Automatically installs uv and all dependencies for the application.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# Configuration
VENV_DIR = ".venv"
PYTHON_VERSION = "3.10.11"  # Exact Python version required
UV_MIN_VERSION = "0.4.0"    # Minimum uv version required


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🛡️  Cyber Assessment Reviewer - Setup")
    print("=" * 60)
    print()


def print_step(step_num, total_steps, description):
    """Print setup step"""
    print(f"📋 Step {step_num}/{total_steps}: {description}")


def run_command(command, description, check=True):
    """Run a shell command with error handling"""
    print(f"   🔧 {description}...")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        else:
            result = subprocess.run(command, capture_output=True, text=True, check=check)
        
        if result.returncode == 0:
            print(f"   ✅ {description} completed successfully")
            return True
        else:
            print(f"   ❌ {description} failed")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed with error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error during {description}: {e}")
        return False


def version_ge(version1, version2):
    """Compare version numbers (returns True if version1 >= version2)"""
    try:
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        return v1_parts >= v2_parts
    except:
        return False

def get_uv_version():
    """Get current uv version"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            import re
            match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
            return match.group(1) if match else "0.0.0"
        return "0.0.0"
    except:
        return "0.0.0"

def check_venv_health():
    """Check if virtual environment is valid and has correct Python version"""
    venv_path = Path(VENV_DIR)
    
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        activate_script = venv_path / "Scripts" / "activate"
    else:
        python_exe = venv_path / "bin" / "python"
        activate_script = venv_path / "bin" / "activate"
    
    if venv_path.exists() and activate_script.exists():
        # Check if Python executable exists and works
        if not python_exe.exists():
            print("   ⚠️  Virtual environment appears corrupted")
            return False
        
        try:
            # Check Python version in venv
            result = subprocess.run([str(python_exe), "--version"], 
                                 capture_output=True, text=True, check=True)
            import re
            match = re.search(r'(\d+\.\d+\.\d+)', result.stdout)
            if match:
                venv_python_version = match.group(1)
                required_major_minor = '.'.join(PYTHON_VERSION.split('.')[:2])
                venv_major_minor = '.'.join(venv_python_version.split('.')[:2])
                
                if venv_major_minor != required_major_minor:
                    print(f"   ⚠️  Virtual environment has Python {venv_python_version}, but {required_major_minor}.x is required")
                    return False
                
                print(f"   ✅ Virtual environment is healthy with Python {venv_python_version}")
                return True
        except:
            print("   ⚠️  Virtual environment appears corrupted")
            return False
    
    return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, 6, "Checking Python version")
    
    version = sys.version_info
    current_version = f"{version.major}.{version.minor}.{version.micro}"
    required_major_minor = '.'.join(PYTHON_VERSION.split('.')[:2])
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"   ❌ Python {current_version} is not supported")
        print(f"   📋 This application requires Python 3.10 or higher")
        print(f"   💡 Please install Python 3.10+ from https://python.org")
        sys.exit(1)
    
    print(f"   ✅ Python {current_version} is compatible (target: {PYTHON_VERSION})")
    return True


def install_uv():
    """Install uv if not present"""
    print_step(2, 6, "Installing/Updating uv (ultra-fast Python package manager)")
    
    # Check if uv is already installed and check version
    if shutil.which("uv"):
        current_version = get_uv_version()
        if version_ge(current_version, UV_MIN_VERSION):
            print(f"   ✅ uv {current_version} is already installed and up to date")
            return True
        else:
            print(f"   ⚠️  uv {current_version} is older than required {UV_MIN_VERSION}. Updating...")
    else:
        print("   🔧 Installing uv...")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Windows installation
        command = "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    else:
        # Linux/macOS installation
        command = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    success = run_command(command, "Installing/Updating uv", check=False)
    
    if not success:
        print("   ⚠️  Automatic uv installation failed")
        print("   📋 Please install uv manually:")
        print("   💻 Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print("   🐧 Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("   🌐 Or visit: https://github.com/astral-sh/uv")
        sys.exit(1)
    
    # Add uv to PATH for current session
    if system != "windows":
        os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ.get('PATH', '')}"
    else:
        # For Windows, add cargo bin to PATH
        cargo_bin = os.path.expanduser("~/.cargo/bin")
        if cargo_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{cargo_bin};{os.environ.get('PATH', '')}"
    
    # Verify installation
    if shutil.which("uv"):
        installed_version = get_uv_version()
        print(f"   ✅ uv {installed_version} installed successfully")
    else:
        print("   ❌ uv installation verification failed")
        sys.exit(1)
    
    return True


def create_virtual_environment():
    """Create virtual environment with uv"""
    print_step(3, 6, "Creating/Validating virtual environment")
    
    if check_venv_health():
        print("   ✅ Virtual environment is ready")
        return True
    
    venv_path = Path(VENV_DIR)
    if venv_path.exists():
        print("   🔧 Removing corrupted virtual environment...")
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print(f"   ⚠️  Could not remove corrupted environment: {e}")
    
    print(f"   🔧 Creating virtual environment with Python {PYTHON_VERSION}...")
    
    # Try with specific Python version first
    if run_command(f"uv venv {VENV_DIR} --python {PYTHON_VERSION}", 
                   f"Creating virtual environment with Python {PYTHON_VERSION}", check=False):
        print(f"   ✅ Virtual environment created with Python {PYTHON_VERSION}")
        return True
    
    # Fallback to Python 3.10
    print(f"   ⚠️  Python {PYTHON_VERSION} not found, trying Python 3.10...")
    if run_command(f"uv venv {VENV_DIR} --python python3.10", 
                   "Creating virtual environment with Python 3.10", check=False):
        print("   ✅ Virtual environment created with Python 3.10")
        return True
    
    # Final fallback to system Python
    print("   ⚠️  Python 3.10 not found, trying system Python...")
    if run_command(f"uv venv {VENV_DIR}", "Creating virtual environment with system Python"):
        print("   ✅ Virtual environment created with system Python")
        return True
    
    print("   ❌ Failed to create virtual environment with any Python version")
    return False


def install_dependencies():
    """Install project dependencies"""
    print_step(4, 6, "Installing project dependencies")
    
    # Configure uv to use the virtual environment
    venv_abs_path = os.path.abspath(VENV_DIR)
    os.environ["VIRTUAL_ENV"] = venv_abs_path
    os.environ["UV_PROJECT_ENVIRONMENT"] = venv_abs_path
    
    print("   🔧 Installing dependencies with uv...")
    
    # Check for various dependency files and install accordingly
    if Path("pyproject.toml").exists():
        print("   💡 Using pyproject.toml for dependencies")
        
        # Try to install with dev dependencies first
        print("   🔧 Attempting installation with dev dependencies...")
        success = run_command("uv pip install -e .[dev]", "Installing project dependencies with dev extras", check=False)
        
        if success:
            print("   ✅ Dependencies installed from pyproject.toml (with dev extras)")
        else:
            print("   ⚠️  Dev dependencies installation failed, trying without dev extras...")
            # Try without dev dependencies
            success = run_command("uv pip install -e .", "Installing project dependencies", check=False)
            
            if success:
                print("   ✅ Dependencies installed from pyproject.toml")
            else:
                print("   ❌ pyproject.toml installation failed. This may be due to:")
                print("   • Build configuration issues")
                print("   • Dependency conflicts")
                print("   • Missing system dependencies")
                print("   ⚠️  Trying requirements.txt as fallback...")
                
                if Path("requirements.txt").exists():
                    success = run_command("uv pip install -r requirements.txt", "Installing from requirements.txt")
                    if success:
                        print("   ✅ Dependencies installed from requirements.txt (fallback)")
                    else:
                        print("   ❌ All dependency installation methods failed")
                        print("   📋 Please check:")
                        print("   • Python version compatibility")
                        print("   • System dependencies (gcc, python3-dev, etc.)")
                        print("   • Network connectivity")
                        sys.exit(1)
                else:
                    print("   ❌ No fallback dependency file found")
                    sys.exit(1)
                
    elif Path("requirements.txt").exists():
        print("   💡 Using requirements.txt for dependencies")
        success = run_command("uv pip install -r requirements.txt", "Installing project dependencies")
        if success:
            print("   ✅ Dependencies installed from requirements.txt")
        else:
            print("   ❌ Failed to install dependencies")
            sys.exit(1)
    else:
        print("   ❌ No dependency file found (pyproject.toml or requirements.txt)")
        sys.exit(1)
    
    return True


def activate_venv():
    """Activate virtual environment"""
    system = platform.system().lower()
    
    if system == "windows":
        activate_script = Path(".venv/Scripts/activate")
    else:
        activate_script = Path(".venv/bin/activate")
    
    if activate_script.exists():
        print("   💡 Virtual environment is ready")
        return True
    
    return False


def verify_installation():
    """Verify that key packages are installed"""
    print_step(5, 6, "Verifying installation")
    
    # Get the Python executable from the virtual environment
    venv_path = Path(VENV_DIR)
    if platform.system() == "Windows":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    print("   🔧 Testing core package imports...")
    
    try:
        # Test imports using the virtual environment Python
        test_command = f'"{python_exe}" -c "import flask, pandas, transformers; print(\'✅ Core packages imported successfully\')"'
        result = subprocess.run(test_command, shell=True, capture_output=True, text=True, check=True)
        
        print("   ✅ Installation verified successfully")
        
        # Show versions of key packages
        print("   📦 Installed package versions:")
        
        version_commands = [
            ("Flask", f'"{python_exe}" -c "import flask; print(flask.__version__)"'),
            ("Pandas", f'"{python_exe}" -c "import pandas; print(pandas.__version__)"'),
            ("Transformers", f'"{python_exe}" -c "import transformers; print(transformers.__version__)"')
        ]
        
        for package_name, cmd in version_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                version = result.stdout.strip()
                print(f"   • {package_name}: {version}")
            except:
                print(f"   • {package_name}: unknown")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Installation verification failed")
        print(f"   💡 Some packages may not have installed correctly")
        return False
    except Exception as e:
        print(f"   ❌ Installation verification failed: {e}")
        return False


def show_completion_message():
    """Show setup completion message"""
    print_step(6, 6, "Setup completed successfully!")
    print()
    print("🎉 Installation Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print(f"   1. Activate the virtual environment:")
    
    system = platform.system().lower()
    if system == "windows":
        print(f"      • Windows: {VENV_DIR}\\Scripts\\activate")
    else:
        print(f"      • Linux/macOS: source {VENV_DIR}/bin/activate")
    
    print()
    print("   2. Run the application:")
    if system == "windows":
        print("      • Windows: python run.py  (or run.bat)")
    else:
        print("      • Linux/macOS: python run.py  (or ./run.sh)")
    
    print()
    print("   3. Access the web interface:")
    print("      • Open your browser to: http://localhost:5000")
    print()
    print("💡 Tips:")
    print(f"   • Virtual environment with Python {PYTHON_VERSION} is ready")
    print("   • For better performance, install Ollama: https://ollama.com")
    print("   • First run may take longer as models download")
    print("   • Check README.md for detailed usage instructions")
    print()
    print("🛡️  Ready to analyze cybersecurity assessments!")
    print("=" * 60)


def main():
    """Main setup function"""
    print_header()
    
    try:
        # Setup steps
        check_python_version()
        install_uv()
        create_virtual_environment()
        install_dependencies()
        verify_installation()
        show_completion_message()
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        print("📋 Please check the error messages above and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()