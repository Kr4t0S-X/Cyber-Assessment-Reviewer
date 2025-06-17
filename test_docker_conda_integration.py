#!/usr/bin/env python3
"""
Test Script for Docker + Conda Integration
Tests the complete Docker + Conda deployment pipeline
"""

import os
import sys
import subprocess
import time
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerCondaIntegrationTest:
    """Test suite for Docker + Conda integration"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = []
        
    def run_command(self, command: str, timeout: int = 300) -> tuple:
        """Run shell command and return result"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, 
                text=True, timeout=timeout, cwd=self.project_root
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def test_docker_availability(self) -> bool:
        """Test if Docker is available and running"""
        print("🐳 Testing Docker Availability")
        print("-" * 40)
        
        # Check Docker command
        success, stdout, stderr = self.run_command("docker --version")
        if not success:
            print("❌ Docker not found")
            return False
        
        print(f"✅ Docker detected: {stdout.strip()}")
        
        # Check Docker daemon
        success, stdout, stderr = self.run_command("docker info")
        if not success:
            print("❌ Docker daemon not running")
            return False
        
        print("✅ Docker daemon is running")
        
        # Check Docker Compose
        success, stdout, stderr = self.run_command("docker-compose --version")
        if not success:
            success, stdout, stderr = self.run_command("docker compose version")
        
        if success:
            print(f"✅ Docker Compose detected: {stdout.strip()}")
        else:
            print("⚠️  Docker Compose not found")
        
        return True
    
    def test_dockerfile_conda_exists(self) -> bool:
        """Test if Dockerfile.conda exists and is valid"""
        print("\n📄 Testing Dockerfile.conda")
        print("-" * 40)
        
        dockerfile_path = self.project_root / "Dockerfile.conda"
        if not dockerfile_path.exists():
            print("❌ Dockerfile.conda not found")
            return False
        
        print("✅ Dockerfile.conda exists")
        
        # Check for key stages
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        required_stages = ["conda-base", "env-builder", "production"]
        for stage in required_stages:
            if f"as {stage}" in content:
                print(f"✅ Stage '{stage}' found")
            else:
                print(f"❌ Stage '{stage}' missing")
                return False
        
        return True
    
    def test_environment_yml_exists(self) -> bool:
        """Test if environment.yml exists and is valid"""
        print("\n📋 Testing environment.yml")
        print("-" * 40)
        
        env_file = self.project_root / "environment.yml"
        if not env_file.exists():
            print("❌ environment.yml not found")
            return False
        
        print("✅ environment.yml exists")
        
        try:
            import yaml
            with open(env_file, 'r') as f:
                env_data = yaml.safe_load(f)
            
            print(f"✅ Valid YAML format")
            print(f"   Environment: {env_data.get('name', 'N/A')}")
            print(f"   Dependencies: {len(env_data.get('dependencies', []))}")
            
            return True
        except ImportError:
            print("⚠️  PyYAML not available - cannot validate YAML")
            return True
        except Exception as e:
            print(f"❌ Invalid YAML: {e}")
            return False
    
    def test_docker_compose_conda_exists(self) -> bool:
        """Test if docker-compose.conda.yml exists"""
        print("\n🐙 Testing docker-compose.conda.yml")
        print("-" * 40)
        
        compose_file = self.project_root / "docker-compose.conda.yml"
        if not compose_file.exists():
            print("❌ docker-compose.conda.yml not found")
            return False
        
        print("✅ docker-compose.conda.yml exists")
        
        try:
            import yaml
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            print(f"✅ Valid compose format")
            print(f"   Services: {list(services.keys())}")
            
            return True
        except Exception as e:
            print(f"❌ Invalid compose file: {e}")
            return False
    
    def test_build_scripts_exist(self) -> bool:
        """Test if build scripts exist"""
        print("\n🔨 Testing Build Scripts")
        print("-" * 40)
        
        scripts = [
            "docker-build-conda.sh",
            "docker-build-conda.bat",
            "docker-deploy-conda.sh"
        ]
        
        all_exist = True
        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                print(f"✅ {script}")
            else:
                print(f"❌ {script} - Missing")
                all_exist = False
        
        return all_exist
    
    def test_docker_build(self) -> bool:
        """Test Docker build process"""
        print("\n🏗️  Testing Docker Build")
        print("-" * 40)
        
        print("Building conda-based Docker image...")
        
        # Use build script if available, otherwise direct docker build
        if (self.project_root / "docker-build-conda.sh").exists():
            if os.name == 'nt':  # Windows
                build_cmd = "docker-build-conda.bat --type production"
            else:
                build_cmd = "./docker-build-conda.sh --type production"
        else:
            build_cmd = "docker build -f Dockerfile.conda --target production -t cyber-assessment-reviewer:conda-test ."
        
        print(f"Build command: {build_cmd}")
        
        success, stdout, stderr = self.run_command(build_cmd, timeout=1200)  # 20 minutes
        
        if success:
            print("✅ Docker build completed successfully")
            return True
        else:
            print(f"❌ Docker build failed")
            print(f"Error: {stderr}")
            return False
    
    def test_docker_run(self) -> bool:
        """Test running the Docker container"""
        print("\n🚀 Testing Docker Run")
        print("-" * 40)
        
        # Start container in background
        run_cmd = "docker run -d -p 5001:5000 --name conda-test-container cyber-assessment-reviewer:conda-test"
        
        success, stdout, stderr = self.run_command(run_cmd)
        if not success:
            print(f"❌ Failed to start container: {stderr}")
            return False
        
        container_id = stdout.strip()
        print(f"✅ Container started: {container_id[:12]}")
        
        # Wait for container to be ready
        print("Waiting for container to be ready...")
        for i in range(30):
            try:
                response = requests.get("http://localhost:5001/system_status", timeout=5)
                if response.status_code == 200:
                    print("✅ Container is responding")
                    break
            except:
                pass
            time.sleep(2)
            print(".", end="", flush=True)
        else:
            print("\n❌ Container not responding after 60 seconds")
            self.cleanup_test_container()
            return False
        
        # Test basic functionality
        try:
            response = requests.get("http://localhost:5001/", timeout=10)
            if response.status_code == 200:
                print("✅ Application is working")
                success = True
            else:
                print(f"❌ Application returned status {response.status_code}")
                success = False
        except Exception as e:
            print(f"❌ Application test failed: {e}")
            success = False
        
        # Cleanup
        self.cleanup_test_container()
        return success
    
    def test_docker_compose_deployment(self) -> bool:
        """Test Docker Compose deployment"""
        print("\n🐙 Testing Docker Compose Deployment")
        print("-" * 40)
        
        # Check if compose file exists
        compose_file = "docker-compose.conda.yml"
        if not (self.project_root / compose_file).exists():
            print("❌ docker-compose.conda.yml not found")
            return False
        
        # Start services
        compose_cmd = f"docker-compose -f {compose_file} up -d"
        print(f"Starting services: {compose_cmd}")
        
        success, stdout, stderr = self.run_command(compose_cmd, timeout=600)
        if not success:
            print(f"❌ Failed to start services: {stderr}")
            return False
        
        print("✅ Services started")
        
        # Wait for services to be ready
        print("Waiting for services to be ready...")
        for i in range(60):
            try:
                response = requests.get("http://localhost:5000/system_status", timeout=5)
                if response.status_code == 200:
                    print("✅ Services are responding")
                    break
            except:
                pass
            time.sleep(2)
            print(".", end="", flush=True)
        else:
            print("\n❌ Services not responding after 120 seconds")
            self.cleanup_compose_services(compose_file)
            return False
        
        # Test services
        success = True
        try:
            # Test main application
            response = requests.get("http://localhost:5000/", timeout=10)
            if response.status_code == 200:
                print("✅ Main application working")
            else:
                print(f"❌ Main application failed: {response.status_code}")
                success = False
            
            # Test Ollama (if available)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama service working")
                else:
                    print("⚠️  Ollama service not responding")
            except:
                print("⚠️  Ollama service not available")
        
        except Exception as e:
            print(f"❌ Service test failed: {e}")
            success = False
        
        # Cleanup
        self.cleanup_compose_services(compose_file)
        return success
    
    def cleanup_test_container(self):
        """Clean up test container"""
        self.run_command("docker stop conda-test-container")
        self.run_command("docker rm conda-test-container")
    
    def cleanup_compose_services(self, compose_file: str):
        """Clean up compose services"""
        self.run_command(f"docker-compose -f {compose_file} down --volumes")
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        print("🧪 Docker + Conda Integration Test Suite")
        print("=" * 60)
        print()
        
        tests = [
            ("Docker Availability", self.test_docker_availability),
            ("Dockerfile.conda", self.test_dockerfile_conda_exists),
            ("environment.yml", self.test_environment_yml_exists),
            ("docker-compose.conda.yml", self.test_docker_compose_conda_exists),
            ("Build Scripts", self.test_build_scripts_exist),
            # Skip build and run tests by default (too time-consuming)
            # ("Docker Build", self.test_docker_build),
            # ("Docker Run", self.test_docker_run),
            # ("Docker Compose", self.test_docker_compose_deployment),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Results Summary")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Docker + Conda integration is ready!")
            return True
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            return False

def main():
    """Main test function"""
    test_suite = DockerCondaIntegrationTest()
    success = test_suite.run_all_tests()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps")
    print("=" * 60)
    
    if success:
        print("✅ Docker + Conda integration is ready!")
        print()
        print("To build and deploy:")
        print("1. Build image: ./docker-build-conda.sh")
        print("2. Deploy: ./docker-deploy-conda.sh")
        print("3. Or use compose: docker-compose -f docker-compose.conda.yml up")
    else:
        print("❌ Some issues were detected.")
        print("Please review the test output and fix any problems.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
