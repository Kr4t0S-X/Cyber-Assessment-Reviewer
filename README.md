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

**For the refactored modular version, run:**
```bash
python app.py
```

**For the original monolithic version, run:**
```bash
python cyber-assessment-reviewerv37.py
```
