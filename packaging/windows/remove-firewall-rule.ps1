#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Remove the Windows Firewall inbound rule for Space Router Home Node.
#>

$ErrorActionPreference = "Continue"

$RuleName = "Space Router Home Node (TCP-In)"

$existing = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($existing) {
    Remove-NetFirewallRule -DisplayName $RuleName
    Write-Host "Firewall rule removed: $RuleName"
} else {
    Write-Host "Firewall rule not found — nothing to remove."
}
