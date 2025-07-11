# 🪟 Windows Quick Reference - Cyber Assessment Reviewer

## 🚀 Quick Start Commands

### Prerequisites Check
```cmd
# Check if Docker is installed and running
docker --version
docker info

# Check WSL2 status
wsl --list --verbose
```

### One-Command Deployment
```cmd
# Clone and deploy in one go
git clone https://github.com/Kr4t0S-X/Cyber-Assessment-Reviewer.git
cd Cyber-Assessment-Reviewer
docker-build-conda.bat
docker-compose -f docker-compose.conda.yml up -d
```

### Access Application
- **Main Application**: http://localhost:5000
- **Development Version**: http://localhost:5001 (with `--profile dev`)
- **Jupyter Notebook**: http://localhost:8888 (with `--profile dev`)

---

## 🔧 Essential Commands

### Build Commands
```cmd
# Production build
docker-build-conda.bat

# Development build
docker-build-conda.bat --type development

# Clean build (no cache)
docker-build-conda.bat --no-cache
```

### Deployment Commands
```cmd
# Start production environment
docker-compose -f docker-compose.conda.yml up -d

# Start development environment
docker-compose -f docker-compose.conda.yml --profile dev up -d

# Stop all services
docker-compose -f docker-compose.conda.yml down

# View logs
docker-compose -f docker-compose.conda.yml logs -f
```

### Container Management
```cmd
# Check container status
docker-compose -f docker-compose.conda.yml ps

# Restart services
docker-compose -f docker-compose.conda.yml restart

# Access container shell
docker exec -it cyber-assessment-reviewer-conda bash

# View resource usage
docker stats
```

---

## 🐛 Quick Troubleshooting

### Docker Issues
```cmd
# Restart Docker Desktop
# Close Docker Desktop → Restart as Administrator

# Clean Docker system
docker system prune -a -f

# Reset WSL2
wsl --shutdown
# Wait 10 seconds, then restart Docker Desktop
```

### Build Issues
```cmd
# Check disk space (need 20GB+)
dir C:\

# Clean build cache
docker builder prune -f

# Use alternative build
docker build -f Dockerfile.conda -t cyber-assessment-reviewer:conda-latest .
```

### Application Issues
```cmd
# Check application logs
docker logs cyber-assessment-reviewer-conda

# Test application health
curl http://localhost:5000/system_status

# Restart application only
docker restart cyber-assessment-reviewer-conda
```

---

## 📊 System Requirements

### Minimum
- Windows 10 Build 19041+ or Windows 11
- 8GB RAM
- 20GB free disk space
- Docker Desktop with WSL2

### Recommended
- Windows 11 Pro/Enterprise
- 16GB+ RAM
- 50GB+ free disk space (SSD preferred)
- 4+ CPU cores

---

## 🔍 Health Check Commands

```cmd
# Complete system check
docker --version && docker info && wsl --list --verbose

# Application health check
curl http://localhost:5000/system_status

# Container health check
docker-compose -f docker-compose.conda.yml ps

# Resource usage check
docker stats --no-stream
```

---

## 🛠️ Development Shortcuts

### Start Development Environment
```cmd
docker-compose -f docker-compose.conda.yml --profile dev up -d
```

### Run Tests
```cmd
docker exec cyber-assessment-reviewer-conda-dev python -m pytest test_*.py -v
```

### Code Quality
```cmd
# Format code
docker exec cyber-assessment-reviewer-conda-dev black *.py

# Check style
docker exec cyber-assessment-reviewer-conda-dev flake8 *.py
```

### Access Development Tools
```cmd
# Jupyter notebook
start http://localhost:8888

# Development app (with debug)
start http://localhost:5001

# Container shell
docker exec -it cyber-assessment-reviewer-conda-dev bash
```

---

## 📁 Important File Locations

### Configuration Files
- `docker-compose.conda.yml` - Main orchestration
- `Dockerfile.conda` - Multi-stage build definition
- `environment.yml` - Conda environment specification
- `.env` - Environment variables

### Build Scripts
- `docker-build-conda.bat` - Windows build script
- `docker-deploy-conda.sh` - Deployment script (use in WSL2)

### Data Directories
- `data/uploads/` - Uploaded files
- `data/sessions/` - User sessions
- `data/logs/` - Application logs
- `data/models/` - AI models
- `data/ollama/` - Ollama model storage

---

## 🚨 Emergency Commands

### Complete Reset
```cmd
# Stop everything
docker-compose -f docker-compose.conda.yml down --volumes

# Remove all containers and images
docker system prune -a -f

# Reset Docker Desktop
# Docker Desktop → Settings → Troubleshoot → Reset to factory defaults
```

### Quick Recovery
```cmd
# Restart from scratch
git pull origin main
docker-build-conda.bat --no-cache
docker-compose -f docker-compose.conda.yml up -d
```

---

## 📞 Support Resources

### Documentation
- `WINDOWS_DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `INSTALLATION_GUIDE.md` - Alternative installation methods
- `DOCKER_CONDA_INTEGRATION_ANALYSIS.md` - Technical analysis

### Test Scripts
- `test_conda_integration.py` - Test conda functionality
- `test_docker_conda_integration.py` - Test Docker integration

### Log Locations
```cmd
# Application logs
docker logs cyber-assessment-reviewer-conda

# Docker Desktop logs
# %APPDATA%\Docker\log\

# WSL2 logs
wsl --list --verbose
```

---

## 💡 Pro Tips

### Performance Optimization
1. **Move project to WSL2**: Better file I/O performance
2. **Increase Docker resources**: 8GB+ RAM, 4+ CPU cores
3. **Use SSD storage**: Faster container startup
4. **Enable BuildKit**: Set `DOCKER_BUILDKIT=1`

### Security Best Practices
1. **Change default secrets**: Update `.env` file
2. **Use non-root user**: Already configured in containers
3. **Enable Windows Defender exclusions**: For Docker directories
4. **Regular updates**: Keep Docker Desktop updated

### Development Workflow
1. **Use development profile**: `--profile dev` for debugging
2. **Mount source code**: Already configured for live reloading
3. **Use Jupyter**: For data analysis and experimentation
4. **Run tests frequently**: Automated testing in containers

---

## 🎯 Common Use Cases

### Daily Development
```cmd
# Start development environment
docker-compose -f docker-compose.conda.yml --profile dev up -d

# Make code changes (auto-reloads)
# Test in browser: http://localhost:5001

# Run tests
docker exec cyber-assessment-reviewer-conda-dev python test_ai_accuracy.py
```

### Production Deployment
```cmd
# Build production image
docker-build-conda.bat --type production

# Deploy production stack
docker-compose -f docker-compose.conda.yml up -d

# Monitor health
curl http://localhost:5000/system_status
```

### Troubleshooting Session
```cmd
# Check system health
docker system df
docker-compose -f docker-compose.conda.yml ps

# View logs
docker-compose -f docker-compose.conda.yml logs --tail=50

# Access container for debugging
docker exec -it cyber-assessment-reviewer-conda bash
```

---

**🎉 You're ready to use the Cyber Assessment Reviewer with conda-optimized Docker containers on Windows!**

For detailed instructions, see the complete `WINDOWS_DEPLOYMENT_GUIDE.md`.
