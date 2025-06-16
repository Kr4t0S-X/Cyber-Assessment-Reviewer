"""
Utility functions and helpers for Cyber Assessment Reviewer
Contains common utilities, JSON handling, dependency management, and helper functions
"""

import sys
import subprocess
import importlib.util
import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from werkzeug.utils import secure_filename

# Configure logging
logger = logging.getLogger(__name__)

class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NaN values and other edge cases"""
    
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return ''  # Convert NaN/Inf to empty string
        return super().default(obj)
    
    def encode(self, obj):
        # Pre-process the object to handle NaN values
        obj = self._clean_nan_values(obj)
        return super().encode(obj)
    
    def _clean_nan_values(self, obj):
        """Recursively clean NaN values from nested structures"""
        if isinstance(obj, dict):
            return {k: self._clean_nan_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan_values(item) for item in obj]
        elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return ''
        elif hasattr(obj, 'item') and callable(obj.item):
            # Handle numpy scalars
            try:
                val = obj.item()
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return ''
                return val
            except:
                return str(obj)
        else:
            return obj

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> None:
    """Safely dump object to JSON file with NaN handling"""
    with open(file_path, 'w') as f:
        json.dump(obj, f, cls=SafeJSONEncoder, **kwargs)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely serialize object to JSON string with NaN handling"""
    return json.dumps(obj, cls=SafeJSONEncoder, **kwargs)

def clean_sample_control(control: Dict[str, Any]) -> Dict[str, Any]:
    """Clean sample control data for JSON serialization"""
    if not control:
        return None
    
    cleaned = control.copy()
    for key, value in cleaned.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = ''
    return cleaned

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def secure_save_file(file, directory: Path, prefix: str = "") -> Path:
    """Securely save uploaded file with proper naming"""
    filename = secure_filename(file.filename)
    if prefix:
        filename = f"{prefix}_{filename}"
    filepath = directory / filename
    file.save(str(filepath))
    return filepath

def check_and_install_dependencies():
    """Check for required dependencies and install missing ones"""
    
    # Check if we should skip dependency checking
    if os.environ.get('SKIP_DEP_CHECK', '').lower() in ['1', 'true', 'yes']:
        print("Skipping dependency check (SKIP_DEP_CHECK is set)")
        return
    
    print("Checking dependencies...")
    
    # Core dependencies that are always needed
    core_dependencies = {
        'flask': 'flask==3.0.0',
        'pandas': 'pandas==2.1.4',
        'openpyxl': 'openpyxl==3.1.2',
        'PyPDF2': 'PyPDF2==3.0.1',
        'docx': 'python-docx==1.1.0',
        'pptx': 'python-pptx==0.6.23',
        'werkzeug': 'werkzeug==3.0.1',
        'requests': 'requests==2.31.0',
        'numpy': 'numpy==1.24.3'
    }
    
    # Optional dependencies for Transformers mode
    transformers_dependencies = {
        'torch': 'torch==2.1.0',
        'transformers': 'transformers==4.36.0',
        'accelerate': 'accelerate==0.25.0',
        'sentencepiece': 'sentencepiece==0.1.99',
        'bitsandbytes': 'bitsandbytes==0.41.3',
        'scipy': 'scipy==1.11.4',
        'safetensors': 'safetensors==0.4.1'
    }
    
    missing_core = []
    missing_transformers = []
    
    # Check core dependencies
    for module, package in core_dependencies.items():
        if importlib.util.find_spec(module) is None:
            missing_core.append(package)
    
    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_available = response.status_code == 200
    except:
        ollama_available = False
    
    # If Ollama is not available, check Transformers dependencies
    if not ollama_available:
        print("Ollama not detected. Checking Transformers dependencies...")
        for module, package in transformers_dependencies.items():
            if importlib.util.find_spec(module) is None:
                missing_transformers.append(package)
    
    # Install missing packages
    if missing_core or missing_transformers:
        print("\nMissing dependencies detected. Installing...")
        
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install missing core dependencies
        if missing_core:
            print(f"\nInstalling core dependencies: {', '.join(missing_core)}")
            for package in missing_core:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✓ Installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to install {package}: {e}")
                    sys.exit(1)
        
        # Install missing Transformers dependencies if needed
        if missing_transformers and not ollama_available:
            print(f"\nInstalling Transformers dependencies: {', '.join(missing_transformers)}")
            
            # Special handling for PyTorch
            if 'torch' in [pkg.split('==')[0] for pkg in missing_transformers]:
                print("\nInstalling PyTorch... This may take a few minutes.")
                try:
                    # Try to install PyTorch with CUDA support
                    if sys.platform.startswith('linux') or sys.platform == 'win32':
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "torch", "torchvision", "torchaudio", 
                            "--index-url", "https://download.pytorch.org/whl/cu118"
                        ])
                    else:
                        # macOS or other platforms
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "torch", "torchvision", "torchaudio"
                        ])
                    print("✓ Installed PyTorch")
                except subprocess.CalledProcessError:
                    print("✗ Failed to install PyTorch with CUDA, trying CPU-only version...")
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "torch", "torchvision", "torchaudio"
                        ])
                        print("✓ Installed PyTorch (CPU-only)")
                    except subprocess.CalledProcessError as e:
                        print(f"✗ Failed to install PyTorch: {e}")
                        sys.exit(1)
            
            # Install other Transformers dependencies
            for package in missing_transformers:
                if not package.startswith('torch'):
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                        print(f"✓ Installed {package}")
                    except subprocess.CalledProcessError as e:
                        print(f"✗ Failed to install {package}: {e}")
                        # Don't exit for optional dependencies
        
        print("\n✅ All required dependencies installed!")
        print("Please restart the application for changes to take effect.")
        sys.exit(0)
    else:
        if ollama_available:
            print("✅ All core dependencies satisfied. Using Ollama backend.")
        else:
            print("✅ All dependencies satisfied.")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def validate_framework(framework: str, available_frameworks: Dict[str, str]) -> bool:
    """Validate if framework is supported"""
    return framework in available_frameworks

def calculate_file_size_mb(filepath: Path) -> float:
    """Calculate file size in MB"""
    return filepath.stat().st_size / (1024 * 1024)

def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def get_available_wsgi_server():
    """Detect available WSGI servers and return the best option"""
    import platform

    # Check for Waitress (cross-platform, good for Windows)
    try:
        import waitress
        return 'waitress'
    except ImportError:
        pass

    # Check for Gunicorn (Unix-only, but very robust)
    if platform.system() != 'Windows':
        try:
            import gunicorn
            return 'gunicorn'
        except ImportError:
            pass

    # Fallback to Flask development server
    return 'flask'

def install_wsgi_server():
    """Install appropriate WSGI server for the platform"""
    import platform

    print("🔧 Installing production WSGI server...")

    try:
        if platform.system() == 'Windows':
            # Install Waitress for Windows
            subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress==3.0.0"])
            print("✅ Installed Waitress (Windows-compatible WSGI server)")
        else:
            # Install Gunicorn for Unix-like systems
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn==21.2.0"])
            print("✅ Installed Gunicorn (Unix WSGI server)")

        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install WSGI server: {e}")
        return False
