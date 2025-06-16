# Cyber Assessment Reviewer - Modular Edition

A modular, AI-powered cybersecurity control analysis system that helps organizations review and assess their cybersecurity controls against various compliance frameworks.

## 🚀 Features

- **AI-Powered Analysis**: Uses Mistral 7B or other LLMs for intelligent control assessment
- **Multiple Backends**: Supports both Ollama (recommended) and Transformers
- **Framework Support**: NIST, ISO 27001, SOC 2, CIS Controls, PCI DSS
- **File Processing**: Supports PDF, DOCX, XLSX, PPTX evidence files
- **Risk Assessment**: Comprehensive risk scoring and compliance metrics
- **Excel Reports**: Detailed analysis reports in Excel format
- **Modular Architecture**: Clean, maintainable code structure

## 📁 Project Structure

```
cyber-assessment-reviewer/
├── app.py                 # Main application entry point
├── config.py             # Configuration management
├── models.py             # Data models and classes
├── utils.py              # Utility functions and helpers
├── file_processors.py    # File processing logic
├── ai_backend.py         # AI/LLM backend management
├── routes.py             # Flask routes and web logic
├── templates.py          # HTML template generation
├── requirements.txt      # Full dependencies
├── requirements-core.txt # Core dependencies only
└── README.md            # This file
```

## 🛠️ Installation

### Option 1: With Ollama (Recommended)

1. **Install Ollama** (easier setup, better performance):
   ```bash
   # Visit https://ollama.com and install Ollama
   # Then pull the required model:
   ollama pull mistral:7b-instruct
   ```

2. **Install core dependencies**:
   ```bash
   pip install -r requirements-core.txt
   ```

### Option 2: With Transformers (No Ollama)

1. **Install all dependencies** (includes PyTorch and Transformers):
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Starting the Application

```bash
python app.py
```

The application will:
- Check and install missing dependencies automatically
- Initialize the AI backend (Ollama or Transformers)
- Start the web server on http://localhost:5000

### Using the Web Interface

1. **Select Framework**: Choose your compliance framework (NIST, ISO 27001, etc.)
2. **Upload Assessment**: Upload your control assessment file (Excel format)
3. **Upload Evidence**: Upload supporting evidence files (PDF, DOCX, XLSX, PPTX)
4. **Analyze**: Click "Analyze Assessment" to start the AI analysis
5. **Download Report**: Get detailed Excel report with findings and recommendations

## 📊 Supported File Formats

### Assessment Files
- **Excel (.xlsx)**: Control matrices with columns for control ID, name, requirements, answers, status

### Evidence Files
- **PDF (.pdf)**: Policy documents, procedures, screenshots
- **Word (.docx)**: Documentation, procedures, reports
- **PowerPoint (.pptx)**: Presentations, training materials
- **Excel (.xlsx)**: Additional data, logs, configurations

## 🚀 Quick Start

**For development (with production WSGI server):**
```bash
python app.py
```

**For production deployment:**
```bash
python run_production.py
```

**For the original monolithic version:**
```bash
python cyber-assessment-reviewerv37.py
```

## 🏭 Production WSGI Server

The application now uses **production-ready WSGI servers** instead of Flask's development server:

- **Waitress** (Windows/Cross-platform) - Automatically installed and used
- **Gunicorn** (Linux/Unix) - Used on Unix-like systems
- **Automatic detection** - Chooses the best server for your platform

### Benefits:
- ✅ **No more development server warnings**
- ✅ **Better performance and stability**
- ✅ **Production-ready out of the box**
- ✅ **Automatic installation if missing**

### Configuration:
```bash
# Environment variables for WSGI server tuning
export WSGI_WORKERS=4        # Number of worker processes (Gunicorn)
export WSGI_THREADS=4        # Number of threads per worker
export WSGI_TIMEOUT=120      # Request timeout in seconds
export USE_PRODUCTION_SERVER=true  # Force production server (default: true)
```
