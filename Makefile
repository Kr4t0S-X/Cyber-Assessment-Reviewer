# Makefile for Cyber Assessment Reviewer Docker Management
# Cross-platform Docker management commands

.PHONY: help build deploy deploy-transformers test clean logs status restart stop down

# Default target
help:
	@echo "Cyber Assessment Reviewer - Docker Management"
	@echo "============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build              - Build Docker images"
	@echo "  make deploy             - Deploy with Ollama backend"
	@echo "  make deploy-transformers - Deploy with Transformers backend"
	@echo "  make test               - Run deployment tests"
	@echo "  make logs               - View application logs"
	@echo "  make status             - Show container status"
	@echo "  make restart            - Restart services"
	@echo "  make stop               - Stop services"
	@echo "  make down               - Stop and remove containers"
	@echo "  make clean              - Clean up Docker resources"
	@echo "  make setup              - Initial setup (create .env)"
	@echo ""
	@echo "Production commands:"
	@echo "  make deploy-prod        - Deploy with production configuration"
	@echo "  make deploy-nginx       - Deploy with nginx reverse proxy"
	@echo ""

# Setup environment
setup:
	@echo "Setting up environment..."
	@if not exist .env copy .env.example .env
	@if not exist data mkdir data
	@if not exist data\uploads mkdir data\uploads
	@if not exist data\sessions mkdir data\sessions
	@if not exist data\logs mkdir data\logs
	@if not exist data\models mkdir data\models
	@if not exist data\ollama mkdir data\ollama
	@if not exist data\transformers_cache mkdir data\transformers_cache
	@echo "Setup complete! Please edit .env file with your configuration."

# Build Docker images
build: setup
	@echo "Building Docker images..."
	docker build --target production --tag cyber-assessment-reviewer:latest --tag cyber-assessment-reviewer:ollama .
	docker build --target transformers --tag cyber-assessment-reviewer:transformers .
	@echo "Build complete!"

# Deploy with Ollama (default)
deploy: build
	@echo "Deploying with Ollama backend..."
	docker-compose up -d
	@echo "Deployment complete! Application available at http://localhost:5000"

# Deploy with Transformers
deploy-transformers: build
	@echo "Deploying with Transformers backend..."
	docker-compose -f docker-compose.transformers.yml up -d
	@echo "Deployment complete! Application available at http://localhost:5000"

# Deploy with production configuration
deploy-prod: build
	@echo "Deploying with production configuration..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production deployment complete!"

# Deploy with nginx reverse proxy
deploy-nginx: build
	@echo "Deploying with nginx reverse proxy..."
	@if not exist nginx mkdir nginx
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Deployment with nginx complete! Application available at http://localhost"

# Test deployment
test:
	@echo "Testing deployment..."
	@echo "Checking if containers are running..."
	docker ps | findstr cyber-assessment || echo "No containers found"
	@echo "Testing application endpoint..."
	curl -f http://localhost:5000/system_status || echo "Application not responding"
	@echo "Test complete!"

# View logs
logs:
	docker-compose logs -f

# Show container status
status:
	@echo "Container Status:"
	@echo "=================="
	docker-compose ps
	@echo ""
	@echo "Resource Usage:"
	@echo "==============="
	docker stats --no-stream

# Restart services
restart:
	@echo "Restarting services..."
	docker-compose restart
	@echo "Services restarted!"

# Stop services
stop:
	@echo "Stopping services..."
	docker-compose stop
	@echo "Services stopped!"

# Stop and remove containers
down:
	@echo "Stopping and removing containers..."
	docker-compose down
	docker-compose -f docker-compose.transformers.yml down
	@echo "Containers removed!"

# Clean up Docker resources
clean: down
	@echo "Cleaning up Docker resources..."
	docker system prune -f
	docker volume prune -f
	@echo "Cleanup complete!"

# Development commands
dev-logs:
	docker-compose logs -f cyber-assessment-reviewer

dev-shell:
	docker-compose exec cyber-assessment-reviewer /bin/bash

dev-ollama-shell:
	docker-compose exec ollama /bin/bash

# Backup data
backup:
	@echo "Creating backup..."
	@if not exist backups mkdir backups
	tar -czf backups\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.tar.gz data\
	@echo "Backup created in backups\ directory"

# Restore from backup (specify BACKUP_FILE=filename)
restore:
	@echo "Restoring from backup..."
	@if "$(BACKUP_FILE)"=="" (echo "Please specify BACKUP_FILE=filename" && exit 1)
	tar -xzf $(BACKUP_FILE) -C .
	@echo "Restore complete!"

# Update images
update:
	@echo "Updating Docker images..."
	docker-compose pull
	docker-compose up -d
	@echo "Update complete!"

# Health check
health:
	@echo "Health Check:"
	@echo "============="
	@curl -f http://localhost:5000/system_status && echo "✓ Application healthy" || echo "✗ Application unhealthy"
	@curl -f http://localhost:11434/api/tags && echo "✓ Ollama healthy" || echo "✗ Ollama unhealthy"

# Quick start (build and deploy)
start: build deploy
	@echo "Quick start complete!"
	@echo "Application is available at: http://localhost:5000"
	@echo "Use 'make logs' to view logs"
	@echo "Use 'make status' to check status"
	@echo "Use 'make stop' to stop services"
