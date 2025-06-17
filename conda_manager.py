#!/usr/bin/env python3
"""
Anaconda/Miniconda Environment Manager for Cyber Assessment Reviewer
Provides automatic detection, environment creation, and dependency management
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CondaManager:
    """Manages Anaconda/Miniconda environments for the Cyber Assessment Reviewer"""
    
    def __init__(self, env_name: str = "cyber-assessment-env"):
        self.env_name = env_name
        self.conda_executable = None
        self.conda_available = False
        self.conda_version = None
        self.conda_type = None  # 'anaconda', 'miniconda', or 'conda'
        
        # Detect conda installation
        self._detect_conda()
        
        # Define conda-preferred packages (available in conda-forge)
        self.conda_packages = {
            'python': '3.10',
            'flask': None,
            'pandas': None,
            'numpy': None,
            'requests': None,
            'python-docx': None,
            'openpyxl': None,
            'pypdf2': None,
            'python-pptx': None,
            'transformers': None,
            'torch': None,
            'scikit-learn': None,
            'matplotlib': None,
            'seaborn': None,
            'jupyter': None,
            'ipykernel': None
        }
        
        # Packages that must be installed via pip
        self.pip_only_packages = [
            'ollama',
            'schedule'  # If we add it back later
        ]
        
        logger.info(f"CondaManager initialized. Conda available: {self.conda_available}")
    
    def _detect_conda(self) -> bool:
        """Detect if conda is available and get version info"""
        
        # Try different conda executable names
        conda_commands = ['conda', 'conda.exe']
        
        for cmd in conda_commands:
            try:
                # Check if conda is in PATH
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    self.conda_executable = cmd
                    self.conda_available = True
                    self.conda_version = result.stdout.strip()
                    
                    # Detect conda type
                    self._detect_conda_type()
                    
                    logger.info(f"Conda detected: {self.conda_type} {self.conda_version}")
                    return True
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        logger.info("Conda not detected in system PATH")
        return False
    
    def _detect_conda_type(self):
        """Detect if this is Anaconda, Miniconda, or standalone conda"""
        try:
            result = subprocess.run([self.conda_executable, 'info'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                info_text = result.stdout.lower()
                if 'anaconda' in info_text:
                    self.conda_type = 'anaconda'
                elif 'miniconda' in info_text:
                    self.conda_type = 'miniconda'
                else:
                    self.conda_type = 'conda'
            else:
                self.conda_type = 'conda'
                
        except Exception as e:
            logger.warning(f"Could not detect conda type: {e}")
            self.conda_type = 'conda'
    
    def is_conda_available(self) -> bool:
        """Check if conda is available"""
        return self.conda_available
    
    def environment_exists(self) -> bool:
        """Check if the conda environment already exists"""
        if not self.conda_available:
            return False
        
        try:
            result = subprocess.run([self.conda_executable, 'env', 'list', '--json'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                env_data = json.loads(result.stdout)
                env_paths = env_data.get('envs', [])
                
                # Check if our environment exists
                for env_path in env_paths:
                    if self.env_name in env_path:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking environment existence: {e}")
            return False
    
    def create_environment(self, python_version: str = "3.10") -> bool:
        """Create a new conda environment"""
        if not self.conda_available:
            logger.error("Conda not available, cannot create environment")
            return False
        
        if self.environment_exists():
            logger.info(f"Environment '{self.env_name}' already exists")
            return True
        
        try:
            logger.info(f"Creating conda environment '{self.env_name}' with Python {python_version}")
            
            cmd = [
                self.conda_executable, 'create', 
                '-n', self.env_name, 
                f'python={python_version}',
                '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully created environment '{self.env_name}'")
                return True
            else:
                logger.error(f"Failed to create environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating conda environment: {e}")
            return False
    
    def install_dependencies(self, requirements_file: str = "requirements.txt") -> bool:
        """Install dependencies using conda and pip as fallback"""
        if not self.conda_available:
            logger.error("Conda not available, cannot install dependencies")
            return False
        
        if not self.environment_exists():
            logger.error(f"Environment '{self.env_name}' does not exist")
            return False
        
        success = True
        
        # Read requirements from file if it exists
        requirements = []
        if Path(requirements_file).exists():
            requirements = self._parse_requirements_file(requirements_file)
        
        # Install conda packages first
        conda_packages_to_install = []
        pip_packages_to_install = []
        
        for req in requirements:
            package_name = self._extract_package_name(req)
            
            if package_name in self.conda_packages:
                conda_packages_to_install.append(req)
            elif package_name in self.pip_only_packages:
                pip_packages_to_install.append(req)
            else:
                # Try conda first, fallback to pip
                conda_packages_to_install.append(req)
        
        # Install conda packages
        if conda_packages_to_install:
            success &= self._install_conda_packages(conda_packages_to_install)
        
        # Install pip-only packages
        if pip_packages_to_install:
            success &= self._install_pip_packages(pip_packages_to_install)
        
        # Install any remaining packages that failed with conda via pip
        if not success:
            logger.info("Some conda installations failed, trying pip fallback")
            success = self._install_pip_packages(requirements)
        
        return success
    
    def _parse_requirements_file(self, requirements_file: str) -> List[str]:
        """Parse requirements.txt file"""
        requirements = []
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        requirements.append(line)
        except Exception as e:
            logger.error(f"Error reading requirements file: {e}")
        
        return requirements
    
    def _extract_package_name(self, requirement: str) -> str:
        """Extract package name from requirement string"""
        # Handle various requirement formats: package, package==1.0, package>=1.0, etc.
        import re
        match = re.match(r'^([a-zA-Z0-9_-]+)', requirement)
        return match.group(1) if match else requirement
    
    def _install_conda_packages(self, packages: List[str]) -> bool:
        """Install packages using conda"""
        try:
            logger.info(f"Installing conda packages: {packages}")
            
            # Use conda-forge channel for better package availability
            cmd = [
                self.conda_executable, 'install', 
                '-n', self.env_name,
                '-c', 'conda-forge',
                '-y'
            ] + packages
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Successfully installed conda packages")
                return True
            else:
                logger.warning(f"Some conda packages failed to install: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing conda packages: {e}")
            return False
    
    def _install_pip_packages(self, packages: List[str]) -> bool:
        """Install packages using pip in the conda environment"""
        try:
            logger.info(f"Installing pip packages: {packages}")
            
            # Use conda run to execute pip in the environment
            cmd = [
                self.conda_executable, 'run', 
                '-n', self.env_name,
                'pip', 'install'
            ] + packages
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Successfully installed pip packages")
                return True
            else:
                logger.error(f"Failed to install pip packages: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing pip packages: {e}")
            return False
    
    def activate_environment_command(self) -> str:
        """Get the command to activate the conda environment"""
        if not self.conda_available:
            return ""
        
        if os.name == 'nt':  # Windows
            return f"conda activate {self.env_name}"
        else:  # Unix-like systems
            return f"conda activate {self.env_name}"
    
    def get_python_executable(self) -> Optional[str]:
        """Get the Python executable path for the conda environment"""
        if not self.conda_available or not self.environment_exists():
            return None
        
        try:
            cmd = [self.conda_executable, 'run', '-n', self.env_name, 'python', '-c', 
                   'import sys; print(sys.executable)']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting Python executable: {e}")
            return None
    
    def run_in_environment(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command in the conda environment"""
        if not self.conda_available:
            raise RuntimeError("Conda not available")
        
        cmd = [self.conda_executable, 'run', '-n', self.env_name] + command
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def export_environment(self, output_file: str = "environment.yml") -> bool:
        """Export the conda environment to a YAML file"""
        if not self.conda_available or not self.environment_exists():
            return False
        
        try:
            cmd = [self.conda_executable, 'env', 'export', '-n', self.env_name, 
                   '-f', output_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Environment exported to {output_file}")
                return True
            else:
                logger.error(f"Failed to export environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting environment: {e}")
            return False
    
    def remove_environment(self) -> bool:
        """Remove the conda environment"""
        if not self.conda_available:
            return False
        
        try:
            cmd = [self.conda_executable, 'env', 'remove', '-n', self.env_name, '-y']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Successfully removed environment '{self.env_name}'")
                return True
            else:
                logger.error(f"Failed to remove environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing environment: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Union[str, bool, List[str]]]:
        """Get comprehensive information about the conda setup"""
        info = {
            'conda_available': self.conda_available,
            'conda_executable': self.conda_executable,
            'conda_version': self.conda_version,
            'conda_type': self.conda_type,
            'environment_name': self.env_name,
            'environment_exists': self.environment_exists(),
            'python_executable': self.get_python_executable(),
            'activation_command': self.activate_environment_command()
        }
        
        return info

def main():
    """Main function for testing conda manager"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🐍 Conda Manager for Cyber Assessment Reviewer")
    print("=" * 50)
    
    manager = CondaManager()
    info = manager.get_environment_info()
    
    print(f"Conda Available: {info['conda_available']}")
    if info['conda_available']:
        print(f"Conda Type: {info['conda_type']}")
        print(f"Conda Version: {info['conda_version']}")
        print(f"Environment Name: {info['environment_name']}")
        print(f"Environment Exists: {info['environment_exists']}")
        print(f"Activation Command: {info['activation_command']}")
        
        if info['python_executable']:
            print(f"Python Executable: {info['python_executable']}")
    else:
        print("Conda not detected. Using pip-based installation.")

if __name__ == "__main__":
    main()
