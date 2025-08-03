# Cyber Assessment Reviewer - Advanced Windows Installation Script
# PowerShell version of install.sh with equivalent functionality

param(
    [string]$VenvDir = ".venv",
    [string]$PythonVersion = "3.10.11",
    [string]$UvVersion = "0.4.0",
    [switch]$Help,
    [switch]$Version
)

# Set error handling
$ErrorActionPreference = "Stop"

# Color functions for Windows PowerShell
function Write-Status {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[✗] $Message" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[i] $Message" -ForegroundColor Blue
}

function Write-Header {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host "🛡️  Cyber Assessment Reviewer - Advanced Installation" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host ""
}

# Check if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Compare version numbers
function Compare-Version {
    param(
        [string]$Version1,
        [string]$Version2
    )
    try {
        $v1 = [System.Version]$Version1
        $v2 = [System.Version]$Version2
        return $v1 -ge $v2
    }
    catch {
        return $false
    }
}

# Install uv for Windows
function Install-Uv {
    Write-Info "Installing uv package manager..."
    
    try {
        # Use PowerShell to download and install uv
        Write-Info "Downloading uv installer..."
        $uvInstaller = Invoke-RestMethod https://astral.sh/uv/install.ps1
        
        Write-Info "Running uv installer..."
        Invoke-Expression $uvInstaller
        
        # Add to PATH for current session
        $uvPath = "$env:USERPROFILE\.cargo\bin"
        if (Test-Path $uvPath) {
            $env:PATH = "$uvPath;$env:PATH"
        }
        
        # Verify installation
        if (Test-Command "uv") {
            $installedVersion = (uv --version 2>$null | Select-String -Pattern '\d+\.\d+\.\d+').Matches[0].Value
            Write-Status "uv $installedVersion installed successfully"
        }
        else {
            throw "uv installation verification failed"
        }
    }
    catch {
        Write-Error-Custom "Failed to install uv: $_"
        Write-Info "Please install uv manually from: https://github.com/astral-sh/uv"
        exit 1
    }
}

# Check uv installation
function Test-Uv {
    if (-not (Test-Command "uv")) {
        Write-Warning-Custom "uv not found. Installing..."
        Install-Uv
    }
    else {
        # Check uv version
        try {
            $currentVersion = (uv --version 2>$null | Select-String -Pattern '\d+\.\d+\.\d+').Matches[0].Value
            if (-not (Compare-Version $currentVersion $UvVersion)) {
                Write-Warning-Custom "uv version $currentVersion is older than required $UvVersion. Updating..."
                Install-Uv
            }
            else {
                Write-Status "uv $currentVersion is already installed"
            }
        }
        catch {
            Write-Warning-Custom "Could not determine uv version. Reinstalling..."
            Install-Uv
        }
    }
}

# Check if virtual environment exists and is valid
function Test-VirtualEnvironment {
    $activateScript = Join-Path $VenvDir "Scripts\activate.ps1"
    $pythonExe = Join-Path $VenvDir "Scripts\python.exe"
    
    if ((Test-Path $VenvDir) -and (Test-Path $activateScript)) {
        Write-Info "Virtual environment found at $VenvDir"
        
        # Verify it's a valid Python environment
        if (-not (Test-Path $pythonExe)) {
            Write-Warning-Custom "Virtual environment appears corrupted. Recreating..."
            Remove-Item -Recurse -Force $VenvDir
            return $false
        }
        
        try {
            # Check Python version in venv
            $venvPythonOutput = & $pythonExe --version 2>&1
            $venvPythonVersion = ($venvPythonOutput | Select-String -Pattern '\d+\.\d+\.\d+').Matches[0].Value
            $requiredMajorMinor = ($PythonVersion -split '\.')[0..1] -join '.'
            $venvMajorMinor = ($venvPythonVersion -split '\.')[0..1] -join '.'
            
            if ($venvMajorMinor -ne $requiredMajorMinor) {
                Write-Warning-Custom "Virtual environment has Python $venvPythonVersion, but Python $requiredMajorMinor.x is required. Recreating..."
                Remove-Item -Recurse -Force $VenvDir
                return $false
            }
            
            Write-Status "Virtual environment is valid with Python $venvPythonVersion"
            return $true
        }
        catch {
            Write-Warning-Custom "Could not verify virtual environment Python version. Recreating..."
            Remove-Item -Recurse -Force $VenvDir
            return $false
        }
    }
    else {
        Write-Info "No virtual environment found"
        return $false
    }
}

# Create virtual environment using uv
function New-VirtualEnvironment {
    Write-Info "Creating virtual environment with Python $PythonVersion..."
    
    try {
        # Try with specific Python version first
        $result = uv venv $VenvDir --python $PythonVersion 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Virtual environment created with Python $PythonVersion"
            return
        }
        
        # Fallback to Python 3.10
        Write-Warning-Custom "Specific version $PythonVersion not found, trying Python 3.10..."
        $result = uv venv $VenvDir --python python3.10 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Virtual environment created with Python 3.10"
            return
        }
        
        # Final fallback to system Python
        Write-Warning-Custom "Python 3.10 not found, trying system Python..."
        $result = uv venv $VenvDir 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = Join-Path $VenvDir "Scripts\python.exe"
            $systemPythonOutput = & $pythonExe --version 2>&1
            $systemPythonVersion = ($systemPythonOutput | Select-String -Pattern '\d+\.\d+\.\d+').Matches[0].Value
            Write-Status "Virtual environment created with system Python $systemPythonVersion"
            
            # Warn if system Python is too old
            if (-not (Compare-Version $systemPythonVersion "3.10.0")) {
                Write-Warning-Custom "System Python $systemPythonVersion is older than recommended 3.10+"
                Write-Warning-Custom "Some features may not work correctly"
            }
            return
        }
        
        throw "All Python installation attempts failed"
    }
    catch {
        Write-Error-Custom "Failed to create virtual environment: $_"
        exit 1
    }
}

# Configure environment for uv
function Set-UvEnvironment {
    $env:VIRTUAL_ENV = Join-Path $PWD $VenvDir
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $PWD $VenvDir
    Write-Status "Virtual environment configured for uv"
}

# Install dependencies
function Install-Dependencies {
    Write-Info "Installing dependencies in virtual environment..."
    
    # Check for various dependency files
    if (Test-Path "pyproject.toml") {
        Write-Info "Found pyproject.toml"
        
        # Try to install with dev dependencies first, then fallback
        try {
            $result = uv pip install -e ".[dev]" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Status "Dependencies installed from pyproject.toml (with dev extras)"
            }
            else {
                $result = uv pip install -e "." 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Status "Dependencies installed from pyproject.toml"
                }
                else {
                    throw "Failed to install dependencies from pyproject.toml"
                }
            }
        }
        catch {
            Write-Error-Custom "Failed to install dependencies from pyproject.toml: $_"
            exit 1
        }
    }
    elseif (Test-Path "requirements.txt") {
        Write-Info "Found requirements.txt"
        try {
            $result = uv pip install -r requirements.txt 2>&1
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to install from requirements.txt"
            }
            Write-Status "Dependencies installed from requirements.txt"
        }
        catch {
            Write-Error-Custom "Failed to install dependencies from requirements.txt: $_"
            exit 1
        }
        
        # Also check for dev requirements
        if (Test-Path "requirements-dev.txt") {
            Write-Info "Installing dev requirements..."
            try {
                $result = uv pip install -r requirements-dev.txt 2>&1
                if ($LASTEXITCODE -eq 0) {
                    Write-Status "Dev dependencies installed"
                }
                else {
                    Write-Warning-Custom "Failed to install dev dependencies (non-critical)"
                }
            }
            catch {
                Write-Warning-Custom "Failed to install dev dependencies (non-critical): $_"
            }
        }
    }
    elseif (Test-Path "Pipfile") {
        Write-Warning-Custom "Found Pipfile. Pipfile conversion not implemented for Windows."
        Write-Info "Please convert Pipfile to requirements.txt manually or use Linux/macOS installation"
    }
    else {
        Write-Warning-Custom "No dependency file found (pyproject.toml, requirements.txt, or Pipfile)"
        Write-Info "Skipping dependency installation"
        Write-Info "You may need to install dependencies manually later"
    }
    
    # Install common dev tools
    Write-Info "Installing essential development tools..."
    $devTools = @("pip", "setuptools", "wheel")
    foreach ($tool in $devTools) {
        try {
            $result = uv pip install --upgrade $tool 2>&1
            # Don't fail if these don't install
        }
        catch {
            # Silently continue
        }
    }
    
    # Optional dev tools (fail silently)
    Write-Info "Installing optional development tools..."
    $optionalTools = @("black", "flake8", "pytest", "mypy")
    foreach ($tool in $optionalTools) {
        try {
            $result = uv pip install $tool 2>&1
            # Don't fail if these don't install
        }
        catch {
            # Silently continue
        }
    }
}

# Verify installation
function Test-Installation {
    Write-Info "Verifying installation..."
    
    $pythonExe = Join-Path $VenvDir "Scripts\python.exe"
    
    # Test basic Python functionality
    try {
        $result = & $pythonExe -c "print('Python is working')" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Custom "Python installation verification failed"
            return $false
        }
    }
    catch {
        Write-Error-Custom "Python installation verification failed: $_"
        return $false
    }
    
    # Test key imports based on what's likely to be in the project
    $testImports = @()
    
    if ((Test-Path "pyproject.toml") -and (Get-Content "pyproject.toml" | Select-String "flask")) {
        $testImports += "flask"
    }
    if ((Test-Path "pyproject.toml") -and (Get-Content "pyproject.toml" | Select-String "pandas")) {
        $testImports += "pandas"
    }
    if ((Test-Path "requirements.txt") -and (Get-Content "requirements.txt" | Select-String "flask")) {
        $testImports += "flask"
    }
    if ((Test-Path "requirements.txt") -and (Get-Content "requirements.txt" | Select-String "pandas")) {
        $testImports += "pandas"
    }
    
    if ($testImports.Count -gt 0) {
        $importTest = ($testImports | ForEach-Object { "import $_" }) -join "; "
        $importTest += "; print('✅ Key packages imported successfully')"
        
        try {
            $result = & $pythonExe -c $importTest 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Status "Installation verified - key packages working"
                
                # Show versions of installed packages
                Write-Info "Installed package versions:"
                foreach ($package in $testImports) {
                    try {
                        $version = & $pythonExe -c "import $package; print($package.__version__)" 2>&1
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "   • $package`: $version"
                        }
                        else {
                            Write-Host "   • $package`: unknown"
                        }
                    }
                    catch {
                        Write-Host "   • $package`: unknown"
                    }
                }
            }
            else {
                Write-Warning-Custom "Some packages may not have installed correctly"
                Write-Info "This might not be critical if you haven't defined dependencies yet"
            }
        }
        catch {
            Write-Warning-Custom "Package verification failed: $_"
            Write-Info "This might not be critical if you haven't defined dependencies yet"
        }
    }
    else {
        Write-Info "No specific package requirements found - basic Python environment ready"
    }
    
    return $true
}

# Show completion message
function Show-Completion {
    Write-Host ""
    Write-Host "🎉 Installation Complete!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor Green
    Write-Host "   1. Activate the virtual environment:"
    Write-Host "      $VenvDir\Scripts\Activate.ps1" -ForegroundColor Blue
    Write-Host ""
    Write-Host "   2. Start development:"
    if ((Test-Path "app.py") -or (Test-Path "main.py") -or (Test-Path "run.py")) {
        Write-Host "      • Run your application"
    }
    else {
        Write-Host "      • Create your Python application files"
    }
    Write-Host ""
    Write-Host "💡 Tips:" -ForegroundColor Green
    Write-Host "   • Virtual environment created at: $VenvDir"
    
    $pythonExe = Join-Path $VenvDir "Scripts\python.exe"
    if (Test-Path $pythonExe) {
        try {
            $pythonVersionOutput = & $pythonExe --version 2>&1
            Write-Host "   • Python version: $pythonVersionOutput"
        }
        catch {
            Write-Host "   • Python version: Unknown"
        }
    }
    
    Write-Host "   • To deactivate later: deactivate"
    Write-Host "   • To remove environment: Remove-Item -Recurse -Force $VenvDir"
    Write-Host ""
    Write-Host "🛡️  Environment ready for development!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Blue
}

# Main execution function
function Main {
    # Handle help and version flags
    if ($Help) {
        Write-Host "Cyber Assessment Reviewer - Advanced Installation Script"
        Write-Host ""
        Write-Host "Usage: .\install.ps1 [options]"
        Write-Host ""
        Write-Host "Options:"
        Write-Host "  -Help              Show this help message"
        Write-Host "  -Version           Show version information"
        Write-Host "  -VenvDir <path>    Virtual environment directory (default: .venv)"
        Write-Host "  -PythonVersion <v> Python version to use (default: 3.10.11)"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\install.ps1                              # Standard installation"
        Write-Host "  .\install.ps1 -PythonVersion 3.11.0       # Use specific Python version"
        Write-Host "  .\install.ps1 -VenvDir myenv               # Use custom environment name"
        return
    }
    
    if ($Version) {
        Write-Host "Cyber Assessment Reviewer Installation Script v1.0"
        Write-Host "uv-based Python environment setup with advanced error handling"
        return
    }
    
    Write-Header
    
    Write-Info "Starting Python environment setup..."
    Write-Info "Target Python version: $PythonVersion"
    Write-Info "Virtual environment location: $VenvDir"
    Write-Host ""
    
    try {
        # Step 1: Check/Install uv
        Write-Info "🔧 Step 1: Checking uv package manager..."
        Test-Uv
        Write-Host ""
        
        # Step 2: Check if virtual environment exists
        Write-Info "🔧 Step 2: Checking virtual environment..."
        if (-not (Test-VirtualEnvironment)) {
            # Step 3: Create virtual environment
            Write-Info "🔧 Step 3: Creating virtual environment..."
            New-VirtualEnvironment
        }
        else {
            Write-Info "🔧 Step 3: Virtual environment already exists and is valid"
        }
        Write-Host ""
        
        # Step 4: Configure environment for uv
        Write-Info "🔧 Step 4: Configuring environment..."
        Set-UvEnvironment
        Write-Host ""
        
        # Step 5: Install dependencies
        Write-Info "🔧 Step 5: Installing dependencies..."
        Install-Dependencies
        Write-Host ""
        
        # Step 6: Verify installation
        Write-Info "🔧 Step 6: Verifying installation..."
        Test-Installation | Out-Null
        Write-Host ""
        
        # Step 7: Show completion message
        Show-Completion
    }
    catch {
        Write-Error-Custom "Installation failed: $_"
        Write-Host "Please check the error messages above and try again." -ForegroundColor Red
        exit 1
    }
}

# Handle Ctrl+C gracefully
try {
    Main
}
catch [System.Management.Automation.ParameterBindingException] {
    Write-Error-Custom "Invalid parameters. Use -Help for usage information."
    exit 1
}
catch {
    Write-Error-Custom "Installation interrupted or failed: $_"
    exit 1
}