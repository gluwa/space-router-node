#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Uninstall the Space Router Home Node Windows Service.

.DESCRIPTION
    Stops the service if running and removes it via WinSW.
    Preserves configuration and log files in ProgramData.

.PARAMETER InstallDir
    Installation directory (default: $env:ProgramFiles\SpaceRouter).
#>
param(
    [string]$InstallDir = "$env:ProgramFiles\SpaceRouter"
)

$ErrorActionPreference = "Continue"

$WinSW = Join-Path $InstallDir "space-router-node-service.exe"

if (-not (Test-Path $WinSW)) {
    Write-Warning "WinSW executable not found at $WinSW — service may already be removed."
    exit 0
}

# Stop the service first
Write-Host "Stopping Space Router Home Node service..."
& $WinSW stop 2>$null
Start-Sleep -Seconds 3

# Uninstall the service
Write-Host "Removing service..."
& $WinSW uninstall 2>$null

Write-Host ""
Write-Host "Service removed."
Write-Host "Configuration preserved at: $env:ProgramData\SpaceRouter\"
