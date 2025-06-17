# Multi-stage Dockerfile for Cyber Assessment Reviewer
# Stage 1: Base Python environment with dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-core.txt requirements.txt ./

# Stage 2: Core dependencies (minimal for Ollama mode)
FROM base as core-deps

# Install core dependencies
RUN pip install --no-cache-dir -r requirements-core.txt

# Stage 3: Full dependencies (includes Transformers)
FROM base as full-deps

# Install all dependencies including Transformers
RUN pip install --no-cache-dir -r requirements.txt

# Stage 4: Production image (default to core dependencies)
FROM core-deps as production

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p uploads sessions logs static models && \
    chown -R appuser:appuser /app

# Remove test files and development artifacts
RUN rm -rf tests/ *.md requirements-*.txt .git/ .gitignore

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/system_status || exit 1

# Set production environment variables
ENV FLASK_ENV=production \
    DEBUG=False \
    USE_PRODUCTION_SERVER=true \
    HOST=0.0.0.0 \
    PORT=5000 \
    LOG_LEVEL=INFO

# Default command
CMD ["python", "app.py"]

# Stage 5: Full image with Transformers (for non-Ollama deployments)
FROM full-deps as transformers

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p uploads sessions logs static models && \
    chown -R appuser:appuser /app

# Remove test files and development artifacts
RUN rm -rf tests/ *.md requirements-*.txt .git/ .gitignore

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/system_status || exit 1

# Set production environment variables
ENV FLASK_ENV=production \
    DEBUG=False \
    USE_PRODUCTION_SERVER=true \
    HOST=0.0.0.0 \
    PORT=5000 \
    LOG_LEVEL=INFO \
    USE_OLLAMA=false

# Default command
CMD ["python", "app.py"]
