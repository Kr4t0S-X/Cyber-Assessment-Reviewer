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


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, 6, "Checking Python version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"   ❌ Python {version.major}.{version.minor} is not supported")
        print(f"   📋 This application requires Python 3.10 or higher")
        print(f"   💡 Please install Python 3.10+ from https://python.org")
        sys.exit(1)
    
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_uv():
    """Install uv if not present"""
    print_step(2, 6, "Installing uv (ultra-fast Python package manager)")
    
    # Check if uv is already installed
    if shutil.which("uv"):
        print("   ✅ uv is already installed")
        return True
    
    system = platform.system().lower()
    
    if system == "windows":
        # Windows installation
        command = "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    else:
        # Linux/macOS installation
        command = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    success = run_command(command, "Installing uv", check=False)
    
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
    
    return True


def create_virtual_environment():
    """Create virtual environment with uv"""
    print_step(3, 6, "Creating virtual environment")
    
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("   ✅ Virtual environment already exists")
        return True
    
    return run_command("uv venv", "Creating virtual environment")


def install_dependencies():
    """Install project dependencies"""
    print_step(4, 6, "Installing project dependencies")
    
    # Install dependencies using uv
    success = run_command("uv pip install -e .", "Installing project dependencies")
    
    if not success:
        print("   ⚠️  Failed to install dependencies with uv")
        print("   🔄 Trying fallback pip installation...")
        
        # Fallback to pip if uv fails
        activate_venv()
        success = run_command("pip install -e .", "Installing with pip (fallback)")
        
        if not success:
            print("   ❌ Dependency installation failed completely")
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
    
    try:
        # Test imports
        import flask
        import pandas
        import transformers
        print("   ✅ Core packages imported successfully")
        
        # Check versions
        print(f"   📦 Flask: {flask.__version__}")
        print(f"   📦 Pandas: {pandas.__version__}")
        print(f"   📦 Transformers: {transformers.__version__}")
        
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def show_completion_message():
    """Show setup completion message"""
    print_step(6, 6, "Setup completed successfully!")
    print()
    print("🎉 Installation Complete!")
    print("=" * 60)
    print()
    print("📋 Next Steps:")
    print("   1. Run the application:")
    
    system = platform.system().lower()
    if system == "windows":
        print("      • Windows: python run.py  (or run.bat)")
    else:
        print("      • Linux/macOS: python run.py  (or ./run.sh)")
    
    print()
    print("   2. Access the web interface:")
    print("      • Open your browser to: http://localhost:5000")
    print()
    print("💡 Tips:")
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