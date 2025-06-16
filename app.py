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
from utils import check_and_install_dependencies, setup_logging, get_available_wsgi_server, install_wsgi_server

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

def run_with_wsgi_server(app: Flask, config: Config):
    """Run the application with appropriate WSGI server"""

    if not config.USE_PRODUCTION_SERVER:
        print("🔧 Using Flask development server (not recommended for production)")
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid double initialization
        )
        return

    # Detect available WSGI server
    wsgi_server = get_available_wsgi_server()

    if wsgi_server == 'flask':
        print("⚠️  No production WSGI server found. Installing...")
        if install_wsgi_server():
            wsgi_server = get_available_wsgi_server()
        else:
            print("❌ Failed to install WSGI server. Using Flask development server.")
            app.run(
                debug=config.DEBUG,
                host=config.HOST,
                port=config.PORT,
                threaded=True
            )
            return

    # Run with production WSGI server
    if wsgi_server == 'waitress':
        run_with_waitress(app, config)
    elif wsgi_server == 'gunicorn':
        run_with_gunicorn(app, config)
    else:
        # Fallback to Flask development server
        print("⚠️  Falling back to Flask development server")
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            threaded=True
        )

def run_with_waitress(app: Flask, config: Config):
    """Run with Waitress WSGI server"""
    try:
        from waitress import serve
        print(f"🚀 Starting Waitress WSGI server...")
        print(f"   Workers: {config.WSGI_THREADS} threads")
        print(f"   Timeout: {config.WSGI_TIMEOUT}s")

        serve(
            app,
            host=config.HOST,
            port=config.PORT,
            threads=config.WSGI_THREADS,
            connection_limit=1000,
            cleanup_interval=30,
            channel_timeout=config.WSGI_TIMEOUT
        )
    except ImportError:
        print("❌ Waitress not available. Install with: pip install waitress")
        raise
    except Exception as e:
        print(f"❌ Waitress server error: {e}")
        raise

def run_with_gunicorn(app: Flask, config: Config):
    """Run with Gunicorn WSGI server"""
    try:
        import gunicorn.app.base

        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()

            def load_config(self):
                config = {key: value for key, value in self.options.items()
                         if key in self.cfg.settings and value is not None}
                for key, value in config.items():
                    self.cfg.set(key.lower(), value)

            def load(self):
                return self.application

        print(f"🚀 Starting Gunicorn WSGI server...")
        print(f"   Workers: {config.WSGI_WORKERS}")
        print(f"   Threads: {config.WSGI_THREADS}")
        print(f"   Timeout: {config.WSGI_TIMEOUT}s")

        options = {
            'bind': f'{config.HOST}:{config.PORT}',
            'workers': config.WSGI_WORKERS,
            'threads': config.WSGI_THREADS,
            'timeout': config.WSGI_TIMEOUT,
            'keepalive': 5,
            'max_requests': 1000,
            'max_requests_jitter': 100,
            'worker_class': 'gthread',
            'worker_connections': 1000,
            'preload_app': True
        }

        StandaloneApplication(app, options).run()

    except ImportError:
        print("❌ Gunicorn not available. Install with: pip install gunicorn")
        raise
    except Exception as e:
        print(f"❌ Gunicorn server error: {e}")
        raise

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
    
    # Start the application with appropriate server
    try:
        run_with_wsgi_server(app, config)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
