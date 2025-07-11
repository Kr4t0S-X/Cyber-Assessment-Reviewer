# 🐳 Docker Setup for Cyber Assessment Reviewer

This document provides a complete guide for containerizing and deploying the Cyber Assessment Reviewer using Docker.

## 📋 Overview

The Docker setup includes:
- **Multi-stage Dockerfile** with optimized builds
- **Docker Compose** configurations for different deployment modes
- **Production-ready** WSGI server (Waitress/Gunicorn)
- **Ollama integration** for AI model serving
- **Persistent data volumes** for uploads, sessions, and logs
- **Health checks** and monitoring
- **Security hardening** and best practices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                          │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │  Nginx Proxy    │    │   Application   │               │
│  │   (Optional)    │◄──►│ cyber-assessment│               │
│  │   Port: 80/443  │    │   Port: 5000    │               │
│  └─────────────────┘    └─────────────────┘               │
│                                   │                        │
│                          ┌─────────────────┐               │
│                          │     Ollama      │               │
│                          │  AI Model Server│               │
│                          │   Port: 11434   │               │
│                          └─────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
cyber-assessment-reviewer/
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml             # Main compose file (Ollama mode)
├── docker-compose.transformers.yml # Transformers mode
├── docker-compose.prod.yml        # Production overrides
├── .dockerignore                   # Docker build exclusions
├── .env.example                    # Environment template
├── docker-build.sh                # Build script (Linux/Mac)
├── docker-deploy.sh               # Deployment script (Linux/Mac)
├── docker-test.sh                 # Testing script (Linux/Mac)
├── docker-manage.bat              # Management script (Windows)
├── Makefile                       # Cross-platform commands
├── nginx/
│   └── nginx.conf                 # Nginx reverse proxy config
└── data/                          # Persistent data (created automatically)
    ├── uploads/                   # File uploads
    ├── sessions/                  # User sessions
    ├── logs/                      # Application logs
    ├── models/                    # AI model cache
    ├── ollama/                    # Ollama model storage
    └── transformers_cache/        # Transformers model cache
```

## 🚀 Quick Start

### 1. Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **4GB RAM** minimum (8GB recommended)
- **10GB disk space** for models and data

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (set SECRET_KEY, etc.)
nano .env
```

### 3. Build and Deploy

**Option A: Using Make (Linux/Mac/Windows with make)**
```bash
make setup    # Initial setup
make build    # Build images
make deploy   # Deploy with Ollama
```

**Option B: Using Scripts**
```bash
# Linux/Mac
./docker-build.sh
./docker-deploy.sh --mode ollama

# Windows
docker-manage.bat build
docker-manage.bat deploy
```

**Option C: Manual Docker Compose**
```bash
# Ollama mode (recommended)
docker-compose up -d

# Transformers mode (standalone)
docker-compose -f docker-compose.transformers.yml up -d
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Security
SECRET_KEY=your-super-secret-key-change-this

# AI Backend
USE_OLLAMA=true
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_MODEL_NAME=mistral:7b-instruct

# Performance
WSGI_WORKERS=4
WSGI_THREADS=4
MAX_CONTROLS_DEFAULT=20

# Resources
MAX_CONTENT_LENGTH=52428800  # 50MB
```

### Deployment Modes

| Mode | Command | Description | Resource Usage |
|------|---------|-------------|----------------|
| **Ollama** | `docker-compose up -d` | Recommended, uses external Ollama | 4GB RAM |
| **Transformers** | `docker-compose -f docker-compose.transformers.yml up -d` | Standalone with built-in models | 8GB RAM |
| **Production** | `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d` | Production hardened | 6GB RAM |

## 🏭 Production Deployment

### 1. Production Configuration

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 2. Security Hardening

- **Set strong SECRET_KEY**
- **Use HTTPS with nginx reverse proxy**
- **Enable resource limits**
- **Configure log rotation**
- **Regular security updates**

### 3. Monitoring

The setup includes optional monitoring with:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Health checks** for all services

## 🔍 Health Checks

All services include comprehensive health checks:

```bash
# Check application health
curl http://localhost:5000/system_status

# Check Ollama health
curl http://localhost:11434/api/tags

# Check container health
docker ps
```

## 📊 Resource Management

### Resource Limits

Production configuration includes:
- **Memory limits**: 4GB per service
- **CPU limits**: 2 cores per service
- **Disk quotas**: Configurable volumes
- **Network limits**: Rate limiting via nginx

### Scaling

For high-availability:
```yaml
services:
  cyber-assessment-reviewer:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
```

## 🐛 Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker-compose logs cyber-assessment-reviewer
   ```

2. **Ollama connection failed**
   ```bash
   docker-compose logs ollama
   curl http://localhost:11434/api/tags
   ```

3. **Out of memory**
   ```bash
   docker stats
   # Reduce WSGI_WORKERS or use Ollama mode
   ```

4. **Permission denied**
   ```bash
   # Fix data directory permissions
   sudo chown -R 1000:1000 data/
   ```

### Useful Commands

```bash
# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Restart services
docker-compose restart

# Clean rebuild
docker-compose down
docker system prune -f
docker-compose up -d --build

# Shell access
docker-compose exec cyber-assessment-reviewer bash
```

## 🔄 Updates and Maintenance

### Regular Updates

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d

# Clean old images
docker image prune -f
```

### Backup and Restore

```bash
# Backup data
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# Restore data
tar -xzf backup-20241217.tar.gz
```

## 🌐 Network Configuration

### Port Mapping

| Service | Internal Port | External Port | Description |
|---------|---------------|---------------|-------------|
| Application | 5000 | 5000 | Main web interface |
| Ollama | 11434 | 11434 | AI model API |
| Nginx | 80/443 | 80/443 | Reverse proxy (optional) |

### Custom Networks

The setup creates a dedicated Docker network:
- **Name**: `cyber-assessment-network`
- **Type**: Bridge network
- **Isolation**: Services isolated from other containers

## 📈 Performance Optimization

### For Better Performance

1. **Use SSD storage** for Docker volumes
2. **Allocate sufficient RAM** (8GB+ recommended)
3. **Use Ollama mode** for better resource efficiency
4. **Enable GPU support** for Transformers mode
5. **Configure nginx caching** for static assets

### GPU Support

For NVIDIA GPU acceleration:
```yaml
services:
  cyber-assessment-reviewer:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 🔒 Security Best Practices

1. **Change default SECRET_KEY**
2. **Use HTTPS in production**
3. **Regular security updates**
4. **Monitor logs for suspicious activity**
5. **Limit file upload sizes**
6. **Use non-root containers** (already implemented)
7. **Network segmentation**
8. **Regular backups**

## 📞 Support

For issues with the Docker setup:
1. Check the logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `curl http://localhost:5000/system_status`
4. Review resource usage: `docker stats`

The Docker setup is production-ready and includes all necessary components for a secure, scalable deployment of the Cyber Assessment Reviewer.
