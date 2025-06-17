#!/usr/bin/env python3
"""
Docker-Optimized Conda Manager for Cyber Assessment Reviewer
Specialized conda management for containerized environments
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from conda_manager import CondaManager

logger = logging.getLogger(__name__)

class DockerCondaManager(CondaManager):
    """Docker-optimized conda manager for containerized environments"""
    
    def __init__(self, env_name: str = "cyber-assessment-env"):
        # Initialize with container-specific settings
        super().__init__(env_name)
        
        # Container-specific paths
        self.container_conda_path = "/opt/conda"
        self.container_env_path = f"/opt/conda/envs/{env_name}"
        
        # Override conda executable for container
        if self.is_in_container():
            self.conda_executable = f"{self.container_conda_path}/bin/conda"
            self.conda_available = os.path.exists(self.conda_executable)
        
        # Container-specific package preferences
        self.container_optimized_packages = {
            # Use conda-forge for better container compatibility
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
            # Container-optimized versions
            'gunicorn': None,  # WSGI server for production
            'supervisor': None,  # Process management
        }
        
        logger.info(f"DockerCondaManager initialized. Container mode: {self.is_in_container()}")
    
    def is_in_container(self) -> bool:
        """Check if running inside a Docker container"""
        return (
            os.path.exists("/.dockerenv") or
            os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read() or
            os.environ.get("DOCKER_BUILD") == "true" or
            os.environ.get("CONDA_DEFAULT_ENV") is not None
        )
    
    def setup_container_environment(self) -> bool:
        """Set up conda environment optimized for container deployment"""
        
        if not self.conda_available:
            logger.error("Conda not available in container")
            return False
        
        logger.info("Setting up container-optimized conda environment")
        
        # Use mamba if available for faster package resolution
        mamba_available = self._check_mamba_available()
        package_manager = "mamba" if mamba_available else "conda"
        
        # Create environment with container-optimized settings
        if not self.environment_exists():
            success = self._create_container_environment(package_manager)
            if not success:
                return False
        
        # Install container-specific dependencies
        success = self._install_container_dependencies(package_manager)
        if not success:
            return False
        
        # Optimize environment for container
        self._optimize_container_environment()
        
        logger.info("Container environment setup completed")
        return True
    
    def _check_mamba_available(self) -> bool:
        """Check if mamba is available for faster package resolution"""
        try:
            mamba_path = f"{self.container_conda_path}/bin/mamba"
            return os.path.exists(mamba_path)
        except Exception:
            return False
    
    def _create_container_environment(self, package_manager: str) -> bool:
        """Create conda environment optimized for containers"""
        try:
            logger.info(f"Creating container environment with {package_manager}")
            
            # Use environment.yml if available
            if Path("/app/environment.yml").exists():
                cmd = [
                    f"{self.container_conda_path}/bin/{package_manager}",
                    "env", "create", "-f", "/app/environment.yml"
                ]
            else:
                # Fallback to manual creation
                cmd = [
                    f"{self.container_conda_path}/bin/{package_manager}",
                    "create", "-n", self.env_name,
                    "python=3.10", "-y"
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("Container environment created successfully")
                return True
            else:
                logger.error(f"Failed to create container environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating container environment: {e}")
            return False
    
    def _install_container_dependencies(self, package_manager: str) -> bool:
        """Install dependencies optimized for container deployment"""
        try:
            logger.info("Installing container-optimized dependencies")
            
            # Install conda packages
            conda_packages = [
                "flask", "pandas", "numpy", "requests",
                "python-docx", "openpyxl", "pypdf2", "python-pptx",
                "transformers", "torch", "scikit-learn",
                "gunicorn"  # Production WSGI server
            ]
            
            cmd = [
                f"{self.container_conda_path}/bin/{package_manager}",
                "install", "-n", self.env_name,
                "-c", "conda-forge", "-y"
            ] + conda_packages
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode != 0:
                logger.warning(f"Some conda packages failed: {result.stderr}")
            
            # Install pip-only packages
            pip_packages = ["ollama"]
            
            pip_cmd = [
                f"{self.container_env_path}/bin/pip",
                "install", "--no-cache-dir"
            ] + pip_packages
            
            result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("Container dependencies installed successfully")
                return True
            else:
                logger.error(f"Failed to install pip packages: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing container dependencies: {e}")
            return False
    
    def _optimize_container_environment(self):
        """Optimize conda environment for container deployment"""
        try:
            logger.info("Optimizing environment for container deployment")
            
            # Clean conda cache
            cleanup_cmd = [f"{self.container_conda_path}/bin/conda", "clean", "-afy"]
            subprocess.run(cleanup_cmd, capture_output=True, timeout=60)
            
            # Remove unnecessary files to reduce image size
            cleanup_paths = [
                f"{self.container_env_path}/lib/python*/site-packages/*/tests",
                f"{self.container_env_path}/lib/python*/site-packages/*/__pycache__",
                f"{self.container_conda_path}/pkgs/cache",
            ]
            
            for pattern in cleanup_paths:
                try:
                    subprocess.run(["find", pattern, "-type", "f", "-delete"], 
                                 capture_output=True, timeout=30)
                except Exception:
                    pass  # Ignore cleanup errors
            
            logger.info("Container environment optimization completed")
            
        except Exception as e:
            logger.warning(f"Environment optimization failed: {e}")
    
    def activate_environment_in_container(self) -> str:
        """Get activation command for container environment"""
        if self.is_in_container():
            return f"source {self.container_conda_path}/etc/profile.d/conda.sh && conda activate {self.env_name}"
        else:
            return super().activate_environment_command()
    
    def get_container_python_executable(self) -> str:
        """Get Python executable path in container"""
        if self.is_in_container():
            return f"{self.container_env_path}/bin/python"
        else:
            return self.get_python_executable()
    
    def run_in_container_environment(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run command in container conda environment"""
        if self.is_in_container():
            # Use conda run for proper environment activation
            cmd = [
                f"{self.container_conda_path}/bin/conda", "run",
                "-n", self.env_name
            ] + command
            return subprocess.run(cmd, capture_output=True, text=True)
        else:
            return super().run_in_environment(command)
    
    def verify_container_installation(self) -> bool:
        """Verify conda installation in container"""
        try:
            # Test key imports
            test_imports = [
                'flask',
                'pandas', 
                'transformers',
                'gunicorn'
            ]
            
            for package in test_imports:
                result = self.run_in_container_environment([
                    'python', '-c', f'import {package}; print(f"{package} OK")'
                ])
                
                if result.returncode != 0:
                    logger.error(f"Failed to import {package} in container")
                    return False
            
            logger.info("Container installation verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Container verification failed: {e}")
            return False
    
    def get_container_environment_info(self) -> Dict[str, Union[str, bool, List[str]]]:
        """Get container-specific environment information"""
        base_info = super().get_environment_info()
        
        container_info = {
            'in_container': self.is_in_container(),
            'container_conda_path': self.container_conda_path,
            'container_env_path': self.container_env_path,
            'container_python_executable': self.get_container_python_executable(),
            'container_activation_command': self.activate_environment_in_container()
        }
        
        # Merge with base info
        base_info.update(container_info)
        return base_info
    
    def create_conda_pack_archive(self, output_path: str = "/tmp/conda-env.tar.gz") -> bool:
        """Create conda-pack archive for environment portability"""
        try:
            # Install conda-pack if not available
            install_cmd = [
                f"{self.container_env_path}/bin/pip",
                "install", "conda-pack"
            ]
            subprocess.run(install_cmd, capture_output=True, timeout=120)
            
            # Create packed environment
            pack_cmd = [
                f"{self.container_env_path}/bin/conda-pack",
                "-n", self.env_name,
                "-o", output_path,
                "--compress"
            ]
            
            result = subprocess.run(pack_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Conda environment packed to {output_path}")
                return True
            else:
                logger.error(f"Failed to pack environment: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating conda-pack archive: {e}")
            return False

def setup_docker_conda_environment():
    """Main function to set up conda environment in Docker container"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🐳🐍 Setting up Docker Conda Environment")
    print("=" * 50)
    
    manager = DockerCondaManager()
    
    # Check if we're in a container
    if not manager.is_in_container():
        print("⚠️  Not running in a container")
        return False
    
    print("✅ Container environment detected")
    
    # Set up environment
    if manager.setup_container_environment():
        print("✅ Conda environment setup completed")
        
        # Verify installation
        if manager.verify_container_installation():
            print("✅ Installation verified")
            
            # Show environment info
            info = manager.get_container_environment_info()
            print(f"Environment: {info['environment_name']}")
            print(f"Python: {info['container_python_executable']}")
            print(f"Activation: {info['container_activation_command']}")
            
            return True
        else:
            print("❌ Installation verification failed")
            return False
    else:
        print("❌ Environment setup failed")
        return False

if __name__ == "__main__":
    success = setup_docker_conda_environment()
    sys.exit(0 if success else 1)
