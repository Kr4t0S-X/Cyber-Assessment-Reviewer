#!/usr/bin/env python3
"""
Cyber Assessment Reviewer - Simple Run Script
Activates virtual environment and starts the application
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def print_header():
    """Print startup header"""
    print("=" * 60)
    print("🛡️  Cyber Assessment Reviewer - Starting Application")
    print("=" * 60)
    print()


def check_virtual_environment():
    """Check if virtual environment exists"""
    venv_path = Path(".venv")
    
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("📋 Please run setup first:")
        print("   • python setup.py")
        print("   • ./setup.sh (Linux/macOS)")
        print("   • setup.bat (Windows)")
        sys.exit(1)
    
    print("✅ Virtual environment found")
    return True


def activate_and_run():
    """Activate virtual environment and run the application"""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows activation
        activate_script = Path(".venv/Scripts/activate.bat")
        python_exe = Path(".venv/Scripts/python.exe")
    else:
        # Linux/macOS activation
        activate_script = Path(".venv/bin/activate")
        python_exe = Path(".venv/bin/python")
    
    if not activate_script.exists():
        print("❌ Virtual environment activation script not found")
        print("📋 Please run setup again to fix the environment")
        sys.exit(1)
    
    if not python_exe.exists():
        print("❌ Python executable not found in virtual environment")
        print("📋 Please run setup again to fix the environment")
        sys.exit(1)
    
    print("🚀 Starting Cyber Assessment Reviewer...")
    print("💡 Access the application at: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run the application using the virtual environment's Python
        subprocess.run([str(python_exe), "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Application failed to start: {e}")
        print("📋 Common solutions:")
        print("   • Run setup again: python setup.py")
        print("   • Check if all dependencies are installed")
        print("   • Ensure no other application is using port 5000")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        print("Thank you for using Cyber Assessment Reviewer!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Main run function"""
    print_header()
    
    try:
        check_virtual_environment()
        activate_and_run()
    except KeyboardInterrupt:
        print("\n\n👋 Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()