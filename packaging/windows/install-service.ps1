#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install Space Router Home Node as a Windows Service via WinSW.

.DESCRIPTION
    Copies the WinSW wrapper and service XML into the installation directory,
    then installs and starts the Windows Service.

.PARAMETER InstallDir
    Installation directory (default: $env:ProgramFiles\SpaceRouter).
#>
param(
    [string]$InstallDir = "$env:ProgramFiles\SpaceRouter"
)

$ErrorActionPreference = "Stop"

$WinSW = Join-Path $InstallDir "space-router-node-service.exe"
$ServiceXml = Join-Path $InstallDir "space-router-node-service.xml"

# Validate files exist
if (-not (Test-Path $WinSW)) {
    Write-Error "WinSW executable not found at $WinSW"
    exit 1
}
if (-not (Test-Path $ServiceXml)) {
    Write-Error "Service XML not found at $ServiceXml"
    exit 1
}

# Create data directories
$DataDir = "$env:ProgramData\SpaceRouter"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\certs" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\logs" | Out-Null

# Copy default config if not present
$DefaultEnv = Join-Path $InstallDir "spacerouter.env.default"
$ActiveEnv = Join-Path $DataDir "spacerouter.env"
if ((Test-Path $DefaultEnv) -and -not (Test-Path $ActiveEnv)) {
    Copy-Item $DefaultEnv $ActiveEnv
    Write-Host "Created default configuration at $ActiveEnv"
}

# Install and start the service
Write-Host "Installing Space Router Home Node service..."
& $WinSW install
if ($LASTEXITCODE -ne 0) {
    Write-Error "Service installation failed (exit code $LASTEXITCODE)"
    exit 1
}

Write-Host "Starting service..."
& $WinSW start
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Service start failed — check configuration at $ActiveEnv"
} else {
    Write-Host "Service started successfully."
}

Write-Host ""
Write-Host "Space Router Home Node installed as a Windows Service."
Write-Host "Configure: $ActiveEnv"
Write-Host "Logs:      $DataDir\logs\"
Write-Host "Status:    sc query SpaceRouterHomeNode"
