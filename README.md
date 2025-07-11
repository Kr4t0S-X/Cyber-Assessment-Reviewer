# Cyber Assessment Reviewer - Simplified Edition

A comprehensive, AI-powered cybersecurity control analysis system that helps organizations review and assess their cybersecurity controls against various compliance frameworks with advanced accuracy metrics, evidence analysis, and risk assessment capabilities.

## 🚀 Key Features

- **AI-Powered Analysis**: Uses Mistral 7B or other LLMs for intelligent control assessment
- **Multiple Backends**: Supports both Ollama (recommended) and Transformers
- **Framework Support**: NIST, ISO 27001, SOC 2, CIS Controls, PCI DSS
- **File Processing**: Supports PDF, DOCX, XLSX, PPTX evidence files
- **Advanced Risk Assessment**: Dynamic risk scoring with contextual factors and interdependency analysis
- **Enhanced Accuracy Engine**: Multi-dimensional accuracy scoring with framework-specific baselines
- **Evidence Quality Analysis**: Sophisticated evidence review with completeness and relevance scoring
- **Technical Findings Generation**: Detailed technical findings with specificity and actionability
- **Excel Reports**: Comprehensive analysis reports in Excel format
- **Modular Architecture**: Clean, maintainable code structure with enhanced components

## ⚡ Quick Start (One-Command Setup)

### Prerequisites
- Python 3.10 or higher
- Internet connection for dependency installation

### Installation & Setup

Choose your platform:

#### 🐧 Linux/macOS
```bash
git clone <repository-url>
cd cyber-assessment-reviewer
./setup.sh
```

#### 🪟 Windows
```batch
git clone <repository-url>
cd cyber-assessment-reviewer
setup.bat
```

#### 🐍 Cross-Platform (Python)
```bash
git clone <repository-url>
cd cyber-assessment-reviewer
python setup.py
```

### Running the Application

#### 🐧 Linux/macOS
```bash
./run.sh
```

#### 🪟 Windows
```batch
run.bat
```

#### 🐍 Cross-Platform (Python)
```bash
python run.py
```

The application will be available at: **http://localhost:5000**

## 🔧 What the Setup Does

The setup script automatically:
1. ✅ Checks Python version compatibility (3.10+)
2. ✅ Installs `uv` (ultra-fast Python package manager)
3. ✅ Creates an isolated virtual environment
4. ✅ Installs all required dependencies
5. ✅ Verifies the installation
6. ✅ Provides next steps

**Total setup time: ~2-3 minutes** (depending on internet speed)

## 🔍 Troubleshooting

### Common Issues & Solutions

#### "Python not found" or "Python version too old"
- **Solution**: Install Python 3.10+ from [python.org](https://python.org)
- **Windows**: Make sure to check "Add Python to PATH" during installation
- **Linux**: `sudo apt-get install python3 python3-pip` (Ubuntu/Debian)
- **macOS**: `brew install python3` (with Homebrew)

#### "uv installation failed"
- **Solution**: Install uv manually:
  - **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - **Linux/macOS**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

#### "Virtual environment creation failed"
- **Solution**: Clear any existing `.venv` folder and run setup again:
  ```bash
  rm -rf .venv  # Linux/macOS
  rmdir /s .venv  # Windows
  ```

#### "Dependencies installation failed"
- **Solution**: The setup script automatically falls back to pip if uv fails
- **Manual fallback**: Activate the virtual environment and install manually:
  ```bash
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate.bat  # Windows
  pip install -e .
  ```

#### "Application won't start"
- **Solution**: Check that you're in the correct directory and virtual environment exists
- **Port conflict**: If port 5000 is in use, the app will show an error
- **Permissions**: Make sure scripts are executable (`chmod +x *.sh` on Linux/macOS)

### Performance Tips

- **Install Ollama**: For faster AI responses, install [Ollama](https://ollama.com) and run:
  ```bash
  ollama pull mistral:7b
  ```
- **First run**: The first run downloads AI models (~13GB) - this is normal
- **Memory**: Ensure you have at least 8GB RAM for optimal performance

## 📁 Project Structure

```
cyber-assessment-reviewer/
├── app.py                    # Main application entry point
├── config.py                # Configuration management
├── models.py                # Data models and classes
├── utils.py                 # Utility functions and helpers
├── file_processors.py       # File processing logic
├── ai_backend.py            # AI/LLM backend management
├── routes.py                # Flask routes and web logic
├── templates.py             # HTML template generation
├── accuracy_engine.py       # Enhanced accuracy calculation system
├── evidence_analyzer.py     # Advanced evidence review system
├── findings_generator.py    # Technical findings generation
├── risk_engine.py           # Dynamic risk assessment engine
├── ai_validation.py         # AI response validation
├── analysis_workflow.py     # Optimized analysis workflow
├── docs/                    # Documentation
│   └── markdown/           # Organized documentation files
├── pyproject.toml          # Modern Python dependencies
├── setup.py                # Cross-platform setup script
├── setup.sh                # Linux/macOS setup script
├── setup.bat               # Windows setup script
├── run.py                  # Cross-platform run script
├── run.sh                  # Linux/macOS run script
├── run.bat                 # Windows run script
└── README.md               # This file
```

## 🔧 Enhanced System Components

### Accuracy Engine
- **Multi-dimensional scoring**: Compliance, risk assessment, finding quality, evidence utilization
- **Framework-specific baselines**: NIST, ISO 27001, SOC 2, CIS Controls, PCI DSS
- **Confidence-weighted accuracy**: Adjusts accuracy based on assessment confidence
- **Improvement tracking**: Monitors accuracy trends over time

### Evidence Analyzer
- **Quality scoring**: Completeness, relevance, freshness, technical depth, credibility
- **Evidence classification**: Policy documents, configurations, audit logs, procedures
- **Gap detection**: Identifies missing or insufficient evidence
- **Contradiction analysis**: Detects conflicting evidence statements

### Findings Generator
- **Technical specificity**: Detailed technical findings with actionable insights
- **Severity classification**: Critical, High, Medium, Low, Informational
- **Correlation analysis**: Identifies relationships between findings
- **Remediation prioritization**: Effort estimates and priority rankings

### Risk Engine
- **Dynamic scoring**: Contextual factors, temporal analysis, threat landscape
- **Quantitative metrics**: Value at Risk, Expected Loss, Risk Exposure
- **Interdependency analysis**: Risk relationships between controls
- **Treatment recommendations**: Mitigate, Transfer, Avoid, Accept strategies

## 📚 Documentation

Comprehensive documentation is available in the `docs/markdown/` folder:

- **[Installation Guide](./docs/markdown/INSTALLATION_GUIDE.md)** - Complete setup instructions
- **[Docker Setup](./docs/markdown/DOCKER_SETUP.md)** - Container deployment
- **[AI Optimization](./docs/markdown/AI_OPTIMIZATION_GUIDE.md)** - Performance tuning
- **[Full Documentation Index](./docs/markdown/README.md)** - Complete documentation catalog

## 🛠️ Installation

### 🚀 Quick Setup (Automated)

**Recommended**: Use the automated setup script that detects your environment:

```bash
# Clone the repository
git clone https://github.com/Kr4t0S-X/Cyber-Assessment-Reviewer.git
cd Cyber-Assessment-Reviewer

# Run automated setup (detects conda/pip automatically)
python setup_environment.py
```

### 🐍 Conda Installation (Recommended)

**Best for**: Better dependency management and environment isolation

#### Windows:
```cmd
setup-conda.bat
```

#### Linux/macOS:
```bash
./setup-conda.sh
```

#### Manual Conda Setup:
```bash
# Create environment from file
conda env create -f environment.yml
conda activate cyber-assessment-env

# Or create manually
conda create -n cyber-assessment-env python=3.10 -y
conda activate cyber-assessment-env
conda install -c conda-forge flask pandas numpy requests python-docx openpyxl pypdf2 python-pptx transformers torch scikit-learn -y
pip install ollama
```

### 📦 Traditional Installation

#### Option 1: With Ollama (Recommended)

1. **Install Ollama** (easier setup, better performance):
   ```bash
   # Visit https://ollama.com and install Ollama
   # Then pull the required model:
   ollama pull mistral:7b-instruct
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   python -m venv cyber-assessment-env
   # Windows: cyber-assessment-env\Scripts\activate
   # Linux/macOS: source cyber-assessment-env/bin/activate
   pip install -r requirements-core.txt
   ```

#### Option 2: With Transformers (No Ollama)

1. **Create virtual environment and install all dependencies**:
   ```bash
   python -m venv cyber-assessment-env
   # Windows: cyber-assessment-env\Scripts\activate
   # Linux/macOS: source cyber-assessment-env/bin/activate
   pip install -r requirements.txt
   ```

> 📖 **Detailed Installation Guide**: See [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) for comprehensive setup instructions, troubleshooting, and advanced options.

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

## 🐳 Docker Deployment

The application is fully containerized with Docker for easy deployment and scaling.

### 🚀 Quick Start with Docker

**1. Build the Docker images:**
```bash
chmod +x docker-build.sh
./docker-build.sh
```

**2. Deploy with Ollama (Recommended):**
```bash
chmod +x docker-deploy.sh
./docker-deploy.sh --mode ollama
```

**3. Deploy with Transformers (Standalone):**
```bash
./docker-deploy.sh --mode transformers
```

### 📦 Docker Images

The project provides two optimized Docker images:

- **`cyber-assessment-reviewer:latest`** - Production image with Ollama support (smaller, faster)
- **`cyber-assessment-reviewer:transformers`** - Full image with Transformers support (larger, self-contained)

### 🔧 Docker Compose Configurations

**Standard deployment (with Ollama):**
```bash
docker-compose up -d
```

**Transformers-only deployment:**
```bash
docker-compose -f docker-compose.transformers.yml up -d
```

### 🌐 Services

| Service | Port | Description |
|---------|------|-------------|
| **cyber-assessment-reviewer** | 5000 | Main application |
| **ollama** | 11434 | AI model server (Ollama mode only) |

### 📁 Persistent Data

Docker volumes are automatically created for:
- `./data/uploads` - Uploaded assessment and evidence files
- `./data/sessions` - User sessions and analysis results
- `./data/logs` - Application logs
- `./data/models` - Cached AI models
- `./data/ollama` - Ollama model storage (Ollama mode)
- `./data/transformers_cache` - Transformers model cache (Transformers mode)

### ⚙️ Environment Configuration

Copy and customize the environment file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

**Key environment variables:**
```bash
# Security
SECRET_KEY=your-super-secret-key-change-this

# AI Backend
USE_OLLAMA=true                    # true for Ollama, false for Transformers
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_MODEL_NAME=mistral:7b-instruct

# Performance
WSGI_WORKERS=4
WSGI_THREADS=4
MAX_CONTROLS_DEFAULT=20

# Resources
MAX_CONTENT_LENGTH=52428800        # 50MB file upload limit
```

### 🔍 Health Checks

Both services include comprehensive health checks:
- **Application**: `http://localhost:5000/system_status`
- **Ollama**: `http://localhost:11434/api/tags`

### 📊 Resource Requirements

**Minimum requirements:**
- **RAM**: 4GB (Ollama mode), 8GB (Transformers mode)
- **Storage**: 10GB for models and data
- **CPU**: 2 cores minimum, 4+ recommended

**Recommended for production:**
- **RAM**: 8GB (Ollama), 16GB (Transformers)
- **Storage**: 50GB SSD
- **CPU**: 4+ cores
- **GPU**: Optional but recommended for Transformers mode

### 🐛 Docker Troubleshooting

**Check service status:**
```bash
docker-compose ps
docker-compose logs -f
```

**Restart services:**
```bash
docker-compose restart
```

**Clean rebuild:**
```bash
docker-compose down
docker system prune -f
./docker-build.sh
./docker-deploy.sh
```

**Check Ollama models:**
```bash
docker-compose exec ollama ollama list
```

**Monitor resource usage:**
```bash
docker stats
```

### 🔒 Production Security

For production deployments:

1. **Set a strong SECRET_KEY**:
   ```bash
   export SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Use HTTPS with reverse proxy** (nginx/traefik)

3. **Limit file upload sizes** and types

4. **Regular security updates**:
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

5. **Monitor logs and metrics**

### 🌍 Scaling and Load Balancing

For high-availability deployments:

```yaml
# docker-compose.prod.yml
services:
  cyber-assessment-reviewer:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### 🔧 GPU Support

For NVIDIA GPU acceleration (Transformers mode):

1. **Install NVIDIA Container Toolkit**
2. **Uncomment GPU configuration** in docker-compose.yml
3. **Deploy with GPU support**:
   ```bash
   docker-compose -f docker-compose.transformers.yml up -d
   ```
