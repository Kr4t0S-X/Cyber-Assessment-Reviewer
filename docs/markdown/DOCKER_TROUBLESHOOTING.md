# 🐛 Docker Troubleshooting Guide

## Common Docker Desktop Issues on Windows

### Error: `error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping"`

This error indicates Docker Desktop connection issues. Here's how to fix it:

## 🔧 **Quick Fixes (Try in Order)**

### **1. Restart Docker Desktop**
```cmd
# Close Docker Desktop completely
# Then restart it from Start Menu or Desktop
```

### **2. Check Docker Desktop Status**
```cmd
# Check if Docker is running
docker version

# Check Docker daemon status
docker info
```

### **3. Restart Docker Service**
```cmd
# Run as Administrator
net stop com.docker.service
net start com.docker.service
```

### **4. Reset Docker Desktop**
```cmd
# In Docker Desktop:
# Settings → Troubleshoot → Reset to factory defaults
```

## 🛠️ **Detailed Solutions**

### **Solution 1: Restart Docker Desktop Properly**

1. **Close Docker Desktop completely**:
   - Right-click Docker icon in system tray
   - Select "Quit Docker Desktop"
   - Wait 30 seconds

2. **Restart Docker Desktop**:
   - Open from Start Menu
   - Wait for "Docker Desktop is running" message

3. **Verify it's working**:
   ```cmd
   docker run hello-world
   ```

### **Solution 2: Check Windows Features**

Ensure required Windows features are enabled:

1. **Open "Turn Windows features on or off"**
2. **Enable these features**:
   - ✅ Hyper-V
   - ✅ Windows Subsystem for Linux
   - ✅ Virtual Machine Platform
   - ✅ Containers

3. **Restart computer** after enabling

### **Solution 3: WSL2 Backend Issues**

If using WSL2 backend:

1. **Update WSL2**:
   ```cmd
   wsl --update
   wsl --shutdown
   ```

2. **Check WSL2 status**:
   ```cmd
   wsl --list --verbose
   ```

3. **Set default WSL version**:
   ```cmd
   wsl --set-default-version 2
   ```

### **Solution 4: Docker Desktop Settings**

1. **Open Docker Desktop Settings**
2. **General tab**:
   - ✅ Use WSL 2 based engine
   - ✅ Start Docker Desktop when you log in

3. **Resources → WSL Integration**:
   - ✅ Enable integration with my default WSL distro
   - ✅ Enable integration with additional distros

### **Solution 5: Clean Docker Installation**

If other solutions don't work:

1. **Uninstall Docker Desktop**:
   - Control Panel → Programs → Uninstall Docker Desktop

2. **Clean remaining files**:
   ```cmd
   # Delete Docker folders (if they exist)
   rmdir /s "C:\Program Files\Docker"
   rmdir /s "%USERPROFILE%\.docker"
   rmdir /s "%APPDATA%\Docker"
   ```

3. **Download and reinstall** latest Docker Desktop from docker.com

## 🚀 **Alternative: Use Docker without Docker Desktop**

If Docker Desktop continues to have issues, you can use Docker in WSL2:

### **Install Docker in WSL2**

1. **Install WSL2 Ubuntu**:
   ```cmd
   wsl --install -d Ubuntu
   ```

2. **Install Docker in WSL2**:
   ```bash
   # In WSL2 terminal
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

3. **Start Docker service**:
   ```bash
   sudo service docker start
   ```

4. **Use from WSL2**:
   ```bash
   # Navigate to your project in WSL2
   cd /mnt/c/Users/Dark-/CascadeProjects/Cyber-Assessment-Reviewer
   
   # Build and run
   docker build -t cyber-assessment-reviewer .
   docker-compose up -d
   ```

## 🔍 **Diagnostic Commands**

Use these to diagnose Docker issues:

```cmd
# Check Docker version
docker --version

# Check Docker info (detailed)
docker info

# Check Docker daemon status
docker system info

# Test Docker with hello-world
docker run hello-world

# Check running containers
docker ps

# Check Docker Desktop logs
# Docker Desktop → Troubleshoot → View logs
```

## 🎯 **For Cyber Assessment Reviewer**

Once Docker is working, try building again:

### **Method 1: Using our scripts**
```cmd
# Windows
docker-manage.bat build

# Or if you have make
make build
```

### **Method 2: Manual build**
```cmd
# Simple build test
docker build --target production -t cyber-assessment-test .

# If successful, build both images
docker build --target production -t cyber-assessment-reviewer:latest .
docker build --target transformers -t cyber-assessment-reviewer:transformers .
```

### **Method 3: Step-by-step debugging**
```cmd
# Test basic Docker functionality
docker run hello-world

# Test Python base image
docker run python:3.10-slim python --version

# Test our Dockerfile stages
docker build --target base -t cyber-assessment-base .
docker build --target core-deps -t cyber-assessment-core .
docker build --target production -t cyber-assessment-prod .
```

## 🆘 **If Nothing Works**

### **Alternative 1: Use GitHub Codespaces**
- Fork the repository to GitHub
- Open in Codespaces
- Docker works out of the box

### **Alternative 2: Use Virtual Machine**
- Install VirtualBox or VMware
- Create Ubuntu VM
- Install Docker in VM

### **Alternative 3: Use Cloud Instance**
- AWS EC2, Google Cloud, or Azure VM
- Pre-installed Docker images available

## 📞 **Getting Help**

If you're still having issues:

1. **Check Docker Desktop logs**:
   - Docker Desktop → Troubleshoot → View logs

2. **Check Windows Event Viewer**:
   - Look for Docker-related errors

3. **Docker Community**:
   - Docker Desktop GitHub issues
   - Docker Community Forums
   - Stack Overflow

4. **System Information**:
   ```cmd
   # Gather system info for support
   systeminfo
   docker version
   docker info
   wsl --list --verbose
   ```

## ✅ **Prevention**

To avoid future issues:

1. **Keep Docker Desktop updated**
2. **Keep Windows updated**
3. **Don't run Docker Desktop as Administrator** (unless necessary)
4. **Allocate sufficient resources** in Docker Desktop settings
5. **Regular restarts** of Docker Desktop
6. **Monitor disk space** (Docker images can be large)

The most common fix is simply restarting Docker Desktop properly. Try that first!
