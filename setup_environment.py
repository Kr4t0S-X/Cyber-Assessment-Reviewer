#!/usr/bin/env python3
"""
Environment Setup Script for Cyber Assessment Reviewer
Automatically detects and sets up either Conda or pip-based environment
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional

from conda_manager import CondaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Handles environment setup for the Cyber Assessment Reviewer"""
    
    def __init__(self):
        self.conda_manager = CondaManager()
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        
    def setup_environment(self, force_pip: bool = False) -> bool:
        """Set up the environment using conda or pip"""
        
        print("🚀 Cyber Assessment Reviewer - Environment Setup")
        print("=" * 60)
        
        # Check if user wants to force pip installation
        if force_pip:
            print("🐍 Forced pip installation mode")
            return self._setup_pip_environment()
        
        # Try conda first if available
        if self.conda_manager.is_conda_available():
            print(f"🐍 Conda detected: {self.conda_manager.conda_type} {self.conda_manager.conda_version}")
            return self._setup_conda_environment()
        else:
            print("🐍 Conda not detected, using pip installation")
            return self._setup_pip_environment()
    
    def _setup_conda_environment(self) -> bool:
        """Set up conda environment"""
        
        print(f"\n📦 Setting up Conda environment: {self.conda_manager.env_name}")
        print("-" * 40)
        
        # Step 1: Create environment if it doesn't exist
        if not self.conda_manager.environment_exists():
            print("Creating new conda environment...")
            if not self.conda_manager.create_environment():
                print("❌ Failed to create conda environment")
                print("🔄 Falling back to pip installation...")
                return self._setup_pip_environment()
            print("✅ Conda environment created successfully")
        else:
            print("✅ Conda environment already exists")
        
        # Step 2: Install dependencies
        print("\nInstalling dependencies...")
        if not self.conda_manager.install_dependencies(str(self.requirements_file)):
            print("⚠️  Some dependencies failed to install via conda")
            print("🔄 Trying pip fallback...")
            
            # Try pip installation in the conda environment
            if not self._install_pip_in_conda():
                print("❌ Failed to install dependencies")
                return False
        
        print("✅ Dependencies installed successfully")
        
        # Step 3: Verify installation
        if self._verify_conda_installation():
            print("\n🎉 Conda environment setup completed successfully!")
            self._print_conda_usage_instructions()
            return True
        else:
            print("❌ Installation verification failed")
            return False
    
    def _setup_pip_environment(self) -> bool:
        """Set up pip-based environment"""
        
        print("\n📦 Setting up pip environment")
        print("-" * 40)
        
        # Check if we're in a virtual environment
        in_venv = self._check_virtual_environment()
        
        if not in_venv:
            print("⚠️  Not in a virtual environment")
            print("💡 Recommendation: Create a virtual environment first")
            print("   python -m venv cyber-assessment-env")
            print("   # Activate it, then run this script again")
            
            response = input("\nContinue with global installation? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Installation cancelled")
                return False
        
        # Install dependencies
        print("Installing dependencies with pip...")
        if not self._install_pip_dependencies():
            print("❌ Failed to install dependencies with pip")
            return False
        
        print("✅ Dependencies installed successfully")
        
        # Verify installation
        if self._verify_pip_installation():
            print("\n🎉 Pip environment setup completed successfully!")
            self._print_pip_usage_instructions()
            return True
        else:
            print("❌ Installation verification failed")
            return False
    
    def _install_pip_in_conda(self) -> bool:
        """Install dependencies using pip in conda environment"""
        try:
            if not self.requirements_file.exists():
                print("⚠️  requirements.txt not found")
                return True  # No requirements to install
            
            result = self.conda_manager.run_in_environment([
                'pip', 'install', '-r', str(self.requirements_file)
            ])
            
            if result.returncode == 0:
                print("✅ Pip dependencies installed in conda environment")
                return True
            else:
                print(f"❌ Pip installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing pip dependencies in conda: {e}")
            return False
    
    def _install_pip_dependencies(self) -> bool:
        """Install dependencies using pip"""
        try:
            if not self.requirements_file.exists():
                print("⚠️  requirements.txt not found")
                return True
            
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return True
            else:
                print(f"❌ Pip installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing pip dependencies: {e}")
            return False
    
    def _check_virtual_environment(self) -> bool:
        """Check if we're running in a virtual environment"""
        return (
            hasattr(sys, 'real_prefix') or  # virtualenv
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
            'CONDA_DEFAULT_ENV' in os.environ  # conda
        )
    
    def _verify_conda_installation(self) -> bool:
        """Verify conda environment installation"""
        try:
            # Test importing key packages
            test_imports = [
                'flask',
                'pandas',
                'transformers'
            ]
            
            for package in test_imports:
                result = self.conda_manager.run_in_environment([
                    'python', '-c', f'import {package}; print(f"{package} OK")'
                ])
                
                if result.returncode != 0:
                    print(f"❌ Failed to import {package}")
                    return False
            
            print("✅ Key packages verified")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying conda installation: {e}")
            return False
    
    def _verify_pip_installation(self) -> bool:
        """Verify pip installation"""
        try:
            # Test importing key packages
            test_imports = ['flask', 'pandas', 'transformers']
            
            for package in test_imports:
                try:
                    __import__(package)
                    print(f"✅ {package} imported successfully")
                except ImportError:
                    print(f"❌ Failed to import {package}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying pip installation: {e}")
            return False
    
    def _print_conda_usage_instructions(self):
        """Print instructions for using conda environment"""
        activation_cmd = self.conda_manager.activate_environment_command()
        
        print("\n" + "=" * 60)
        print("🎯 CONDA ENVIRONMENT USAGE INSTRUCTIONS")
        print("=" * 60)
        print(f"Environment Name: {self.conda_manager.env_name}")
        print(f"Activation Command: {activation_cmd}")
        print()
        print("To use the Cyber Assessment Reviewer:")
        print(f"1. Activate the environment: {activation_cmd}")
        print("2. Run the application: python app.py")
        print("3. Or run tests: python test_ai_accuracy.py")
        print()
        print("To deactivate: conda deactivate")
        print()
        print("Environment management:")
        print(f"- Export environment: conda env export -n {self.conda_manager.env_name} -f environment.yml")
        print(f"- Remove environment: conda env remove -n {self.conda_manager.env_name}")
        print("=" * 60)
    
    def _print_pip_usage_instructions(self):
        """Print instructions for using pip environment"""
        print("\n" + "=" * 60)
        print("🎯 PIP ENVIRONMENT USAGE INSTRUCTIONS")
        print("=" * 60)
        
        if self._check_virtual_environment():
            print("✅ Virtual environment detected")
        else:
            print("⚠️  Global Python environment (consider using virtual environment)")
        
        print()
        print("To use the Cyber Assessment Reviewer:")
        print("1. Run the application: python app.py")
        print("2. Or run tests: python test_ai_accuracy.py")
        print()
        
        if not self._check_virtual_environment():
            print("💡 For better isolation, consider using a virtual environment:")
            print("   python -m venv cyber-assessment-env")
            print("   # Activate and reinstall dependencies")
        
        print("=" * 60)
    
    def export_conda_environment(self) -> bool:
        """Export conda environment for sharing"""
        if not self.conda_manager.is_conda_available():
            print("❌ Conda not available")
            return False
        
        return self.conda_manager.export_environment()
    
    def clean_environment(self) -> bool:
        """Clean up the conda environment"""
        if not self.conda_manager.is_conda_available():
            print("❌ Conda not available")
            return False
        
        if self.conda_manager.environment_exists():
            print(f"Removing conda environment: {self.conda_manager.env_name}")
            return self.conda_manager.remove_environment()
        else:
            print("Environment doesn't exist")
            return True

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup environment for Cyber Assessment Reviewer")
    parser.add_argument('--pip', action='store_true', help='Force pip installation (skip conda)')
    parser.add_argument('--clean', action='store_true', help='Clean up conda environment')
    parser.add_argument('--export', action='store_true', help='Export conda environment')
    parser.add_argument('--info', action='store_true', help='Show environment information')
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.info:
        # Show environment information
        info = setup.conda_manager.get_environment_info()
        print("🔍 Environment Information")
        print("=" * 30)
        for key, value in info.items():
            print(f"{key}: {value}")
        return
    
    if args.clean:
        # Clean up environment
        if setup.clean_environment():
            print("✅ Environment cleaned successfully")
        else:
            print("❌ Failed to clean environment")
        return
    
    if args.export:
        # Export environment
        if setup.export_conda_environment():
            print("✅ Environment exported successfully")
        else:
            print("❌ Failed to export environment")
        return
    
    # Setup environment
    success = setup.setup_environment(force_pip=args.pip)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
