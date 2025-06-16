"""
Configuration module for Cyber Assessment Reviewer
Centralized configuration management for the application
"""

import os
from pathlib import Path
from typing import Dict, Set

class Config:
    """Application configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')  # 0.0.0.0 allows access from other devices on network
    PORT = int(os.environ.get('PORT', 5000))
    
    # File Upload Configuration
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS: Set[str] = {'xlsx', 'pdf', 'ppt', 'pptx', 'docx'}
    
    # Session Configuration
    SESSION_FOLDER = 'sessions'
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    
    # AI Model Configuration
    DEFAULT_MODEL_NAME = "mistral:7b-instruct"
    USE_OLLAMA = os.environ.get('USE_OLLAMA', 'True').lower() == 'true'
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # Processing Limits
    MAX_CONTROLS_DEFAULT = 20
    MAX_PAGES_PDF = 50
    MAX_TEXT_LENGTH_PER_SECTION = 2000
    MAX_EVIDENCE_SECTIONS = 5
    
    # Cybersecurity Frameworks
    FRAMEWORKS: Dict[str, str] = {
        "NIST": "NIST Cybersecurity Framework",
        "ISO27001": "ISO/IEC 27001:2022",
        "SOC2": "SOC 2 Type II",
        "CIS": "CIS Controls v8",
        "PCI-DSS": "PCI-DSS v4.0"
    }
    
    # Directory Structure
    REQUIRED_DIRECTORIES = [
        'uploads',
        'sessions', 
        'static',
        'models',
        'logs'
    ]
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'logs/cyber_assessment_reviewer.log'
    
    # Excel Column Mapping for Assessment Files
    COLUMN_MAPPING = {
        'control': ['control', 'control_id', 'id', 'control id', 'control_number'],
        'name': ['name', 'control_name', 'title', 'control name', 'description'],
        'requirement': ['requirement', 'requirements', 'description', 'control_description'],
        'answer': ['answer', 'response', 'supplier_answer', 'supplier response', 'implementation'],
        'status': ['status', 'implementation_status', 'compliance_status']
    }
    
    # Risk Assessment Configuration
    RISK_WEIGHTS = {
        "Critical": 10,
        "High": 7,
        "Medium": 4,
        "Low": 1,
        "Unknown": 5
    }
    
    # LLM Generation Parameters
    LLM_TEMPERATURE = 0.3
    LLM_MAX_NEW_TOKENS = 1024
    LLM_TOP_P = 0.9
    LLM_REPETITION_PENALTY = 1.1
    
    @classmethod
    def create_directories(cls):
        """Create required directories if they don't exist"""
        for directory in cls.REQUIRED_DIRECTORIES:
            Path(directory).mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        if cls.MAX_CONTENT_LENGTH <= 0:
            errors.append("MAX_CONTENT_LENGTH must be positive")
        
        if cls.PORT < 1 or cls.PORT > 65535:
            errors.append("PORT must be between 1 and 65535")
        
        if not cls.ALLOWED_EXTENSIONS:
            errors.append("ALLOWED_EXTENSIONS cannot be empty")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        return True

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    HOST = os.environ.get('HOST', 'localhost')  # Use localhost for development by default

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production
    
    @classmethod
    def validate_config(cls):
        """Additional validation for production"""
        super().validate_config()
        
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'your-secret-key-here-change-in-production':
            raise ValueError("SECRET_KEY must be set to a secure value in production")

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB for testing

# Configuration factory
def get_config(env: str = None) -> Config:
    """Get configuration based on environment"""
    env = env or os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)
