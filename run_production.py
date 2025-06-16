#!/usr/bin/env python3
"""
Production runner for Cyber Assessment Reviewer
Sets production environment variables and starts the application
"""

import os
import sys

# Set production environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['DEBUG'] = 'False'
os.environ['USE_PRODUCTION_SERVER'] = 'True'
os.environ['LOG_LEVEL'] = 'INFO'

# Ensure SECRET_KEY is set for production
if not os.environ.get('SECRET_KEY'):
    print("⚠️  WARNING: SECRET_KEY not set. Using default (not secure for production)")
    print("   Set SECRET_KEY environment variable for production deployment")
    os.environ['SECRET_KEY'] = 'production-secret-key-change-me'

# Import and run the main application
if __name__ == '__main__':
    from app import main
    print("🚀 Starting Cyber Assessment Reviewer in PRODUCTION mode")
    print("=" * 60)
    main()
