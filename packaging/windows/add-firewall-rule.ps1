#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Add Windows Firewall inbound rule for Space Router Home Node.

.DESCRIPTION
    Creates an inbound TCP rule allowing traffic to the node's listening port.
    Reads port from SR_NODE_PORT environment variable (default: 9090).

.PARAMETER Port
    Override the listening port (default: reads $env:SR_NODE_PORT or 9090).
#>
param(
    [int]$Port = $(if ($env:SR_NODE_PORT) { [int]$env:SR_NODE_PORT } else { 9090 })
)

$ErrorActionPreference = "Stop"

$RuleName = "Space Router Home Node (TCP-In)"

# Remove existing rule if present (idempotent)
$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
    Remove-NetFirewallRule -DisplayName $RuleName
    Write-Host "Removed existing firewall rule."
}

New-NetFirewallRule `
    -DisplayName $RuleName `
    -Description "Allow inbound TCP traffic to Space Router Home Node on port $Port" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort $Port `
    -Action Allow `
    -Profile Domain,Private,Public `
    -Program "$env:ProgramFiles\SpaceRouter\space-router-node.exe" `
    -Enabled True | Out-Null

Write-Host "Firewall rule added: $RuleName (TCP port $Port)"
