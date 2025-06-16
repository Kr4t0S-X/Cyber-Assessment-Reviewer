#!/usr/bin/env python3
"""
Cyber Assessment Reviewer - Main Application Entry Point
A modular, AI-powered cybersecurity control analysis system

This is the main entry point that orchestrates all modules and starts the Flask application.
"""

import sys
from pathlib import Path
from flask import Flask

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration and utilities
from config import get_config, Config
from utils import check_and_install_dependencies, setup_logging

def create_app(config_name: str = None) -> Flask:
    """Application factory function"""
    
    # Get configuration
    config = get_config(config_name)
    
    # Validate configuration
    config.validate_config()
    
    # Setup logging
    setup_logging(config.LOG_LEVEL, config.LOG_FILE)
    
    # Create required directories
    config.create_directories()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Set Flask configuration
    app.secret_key = config.SECRET_KEY
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS
    
    return app, config

def initialize_ai_backend(config: Config):
    """Initialize the AI backend"""
    from ai_backend import CyberAssessmentReviewer
    
    try:
        # Try Ollama first (recommended)
        reviewer = CyberAssessmentReviewer(config, use_ollama=True)
        print("✅ Using Ollama backend")
        return reviewer
    except Exception as e:
        print(f"⚠️  Failed to initialize with Ollama: {e}")
        print("🔄 Falling back to Transformers mode...")
        try:
            reviewer = CyberAssessmentReviewer(config, use_ollama=False)
            print("✅ Using Transformers backend")
            return reviewer
        except Exception as e2:
            print(f"❌ Failed to initialize reviewer: {e2}")
            print("Please install Ollama (recommended) or required Python packages")
            raise

def register_routes(app: Flask, config: Config, reviewer):
    """Register all application routes"""
    from routes import create_routes
    create_routes(app, config, reviewer)

def main():
    """Main application function"""
    
    print("=" * 60)
    print("🛡️  Cyber Assessment Reviewer - Modular Edition")
    print("=" * 60)
    print("\n🔧 Initializing system...")
    
    # Check and install dependencies first
    check_and_install_dependencies()
    
    # Create Flask app and get configuration
    app, config = create_app()
    
    print(f"📋 Configuration: {config.__class__.__name__}")
    print(f"🔧 Debug mode: {config.DEBUG}")

    # Show the correct accessible URL
    if config.HOST == '0.0.0.0':
        accessible_url = f"http://localhost:{config.PORT}"
    else:
        accessible_url = f"http://{config.HOST}:{config.PORT}"
    print(f"🌐 Server will be accessible at: {accessible_url}")
    
    # Initialize AI backend
    print("\n🤖 Initializing AI backend...")
    reviewer = initialize_ai_backend(config)
    
    if reviewer.use_ollama:
        print(f"   Model: {reviewer.model_name}")
        print("   💡 Tip: Ensure Ollama is running with the required model")
    else:
        print("   📦 Using Transformers backend")
        print("   ⚠️  Note: First run will download model (~13GB)")
    
    # Register routes
    print("\n🌐 Setting up web routes...")
    register_routes(app, config, reviewer)
    
    print("\n✅ System initialization complete!")
    print(f"\n🌐 Starting server...")
    print(f"   📍 Access the application at: {accessible_url}")
    print("\n💡 Tips:")
    print("   • For easier setup, install Ollama from https://ollama.com")
    print("   • Supported file types: PDF, DOCX, XLSX, PPTX")
    print("   • Maximum file size: 50MB")
    print("=" * 60)
    
    # Start the Flask application
    try:
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
