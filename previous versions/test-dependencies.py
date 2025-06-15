#!/usr/bin/env python3
# test_dependencies.py - Test script to check what's installed

import sys
import importlib.util
import subprocess

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("-" * 60)

# Test pip
print("\nChecking pip...")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ pip is installed:", result.stdout.strip())
    else:
        print("❌ pip error:", result.stderr)
except Exception as e:
    print("❌ pip not found:", e)

print("-" * 60)

# Test core modules
modules_to_test = [
    ('flask', 'Flask'),
    ('pandas', 'pandas'),
    ('openpyxl', 'openpyxl'),
    ('PyPDF2', 'PyPDF2'),
    ('docx', 'python-docx'),
    ('pptx', 'python-pptx'),
    ('werkzeug', 'werkzeug'),
    ('requests', 'requests'),
    ('numpy', 'numpy'),
]

print("\nChecking core modules:")
for module_name, package_name in modules_to_test:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"✅ {module_name:<15} (package: {package_name})")
    else:
        print(f"❌ {module_name:<15} (package: {package_name}) - NOT INSTALLED")

print("-" * 60)

# Test Ollama
print("\nChecking Ollama...")
try:
    import requests
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code == 200:
        print("✅ Ollama is running")
        models = response.json().get("models", [])
        print(f"   Available models: {len(models)}")
        for model in models[:5]:  # Show first 5
            print(f"   - {model.get('name', 'unknown')}")
    else:
        print("⚠️  Ollama responded but with status:", response.status_code)
except requests.exceptions.ConnectionError:
    print("❌ Ollama is not running (connection refused)")
except ImportError:
    print("❌ requests module not available to check Ollama")
except Exception as e:
    print("❌ Error checking Ollama:", e)

print("-" * 60)

# Test optional modules
print("\nChecking optional modules (for Transformers):")
optional_modules = ['torch', 'transformers', 'accelerate', 'bitsandbytes']
for module in optional_modules:
    spec = importlib.util.find_spec(module)
    if spec is not None:
        print(f"✅ {module}")
    else:
        print(f"⚠️  {module} - not installed (OK if using Ollama)")

print("-" * 60)
print("\nDiagnostics complete!")
print("\nTo install missing dependencies manually:")
print("pip install flask pandas openpyxl PyPDF2 python-docx python-pptx werkzeug requests numpy")
