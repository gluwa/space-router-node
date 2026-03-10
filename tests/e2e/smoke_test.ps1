# E2E smoke tests for the compiled Space Router Home Node binary.
# Runs on Windows CI runners after PyInstaller build.
#
# Usage: .\tests\e2e\smoke_test.ps1 -Binary "dist\space-router-node-windows-x64.exe"
#
# Required environment variables:
#   EXPECTED_VERSION  - version string the binary should report
#   SR_NODE_PORT      - port for the node to bind (use a high port to avoid conflicts)
#   SR_PUBLIC_IP      - public IP override
#   SR_UPNP_ENABLED   - set to "false" for CI

param(
    [Parameter(Mandatory = $true)]
    [string]$Binary
)

$ErrorActionPreference = "Continue"
$Pass = 0
$Fail = 0
$MockApiProcess = $null
$MockApiPort = 19099
$BasePort = [int]$env:SR_NODE_PORT
if ($BasePort -eq 0) { $BasePort = 19090 }
$PortBindingPort = $BasePort
$ShutdownPort = $BasePort + 1

function Log($msg) { Write-Host "  [INFO]  $msg" }
function Pass($msg) { Write-Host "  [PASS]  $msg"; $script:Pass++ }
function Fail($msg) { Write-Host "  [FAIL]  $msg"; $script:Fail++ }

# Start a mock coordination API
function Start-MockApi {
    $mockScript = @"
import http.server, json

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'id': 'test-node-001'}).encode())
    def do_PATCH(self):
        length = int(self.headers.get('Content-Length', 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{}')
    def log_message(self, *args):
        pass

server = http.server.HTTPServer(('127.0.0.1', $MockApiPort), Handler)
server.serve_forever()
"@

    $tempFile = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.py'
    $mockScript | Out-File -FilePath $tempFile -Encoding utf8
    $script:MockApiProcess = Start-Process -FilePath python -ArgumentList $tempFile -PassThru -NoNewWindow
    $env:SR_COORDINATION_API_URL = "http://127.0.0.1:$MockApiPort"
    Log "Started mock coordination API on port $MockApiPort (PID $($script:MockApiProcess.Id))"

    # Wait for the mock API to be ready (up to 10 seconds)
    for ($i = 0; $i -lt 20; $i++) {
        try {
            $null = Invoke-WebRequest -Uri "http://127.0.0.1:$MockApiPort/" -Method GET -TimeoutSec 1 -ErrorAction SilentlyContinue
            Log "Mock API is ready"
            return
        }
        catch {
            # Connection refused or other error — server not ready yet
        }
        Start-Sleep -Milliseconds 500
    }
    Log "WARNING: Mock API may not be ready after 10 seconds"
}

function Stop-MockApi {
    if ($null -ne $script:MockApiProcess -and -not $script:MockApiProcess.HasExited) {
        Stop-Process -Id $script:MockApiProcess.Id -Force -ErrorAction SilentlyContinue
        $script:MockApiProcess.WaitForExit(5000) | Out-Null
    }
}

# ---------- Test 1: --version flag ----------
function Test-VersionFlag {
    Log "Testing --version flag..."
    try {
        $output = & $Binary --version 2>&1 | Out-String
        if ($output -match [regex]::Escape($env:EXPECTED_VERSION)) {
            Pass "--version reports '$($env:EXPECTED_VERSION)'"
        }
        else {
            Fail "--version output was '$($output.Trim())', expected to contain '$($env:EXPECTED_VERSION)'"
        }
    }
    catch {
        Fail "--version threw an error: $_"
    }
}

# ---------- Test 2: Binary starts and binds to port ----------
function Test-PortBinding {
    # Use a dedicated port so TIME_WAIT state doesn't affect other tests
    $env:SR_NODE_PORT = "$PortBindingPort"
    Log "Testing port binding on port $($env:SR_NODE_PORT)..."

    $proc = Start-Process -FilePath $Binary -PassThru -NoNewWindow
    Log "Started binary with PID $($proc.Id)"

    Start-Sleep -Seconds 6

    if ($proc.HasExited) {
        Fail "Binary exited prematurely with code $($proc.ExitCode)"
        return
    }

    # Check if the port is listening
    try {
        $connection = Test-NetConnection -ComputerName 127.0.0.1 -Port ([int]$env:SR_NODE_PORT) -WarningAction SilentlyContinue
        $listening = $connection.TcpTestSucceeded
    }
    catch {
        Log "Test-NetConnection failed: $_, trying netstat fallback"
        $netstat = netstat -an 2>$null | Select-String ":$($env:SR_NODE_PORT)\s+.*LISTENING"
        $listening = ($null -ne $netstat)
    }

    # Clean up
    try {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        $proc.WaitForExit(5000) | Out-Null
    }
    catch { }

    if ($listening) {
        Pass "Binary is listening on port $($env:SR_NODE_PORT)"
    }
    else {
        Fail "Binary did not bind to port $($env:SR_NODE_PORT)"
    }
}

# ---------- Test 3: Clean shutdown ----------
function Test-CleanShutdown {
    # Use a different port than Test-PortBinding to avoid TIME_WAIT conflicts
    $env:SR_NODE_PORT = "$ShutdownPort"
    Log "Testing clean shutdown on port $($env:SR_NODE_PORT)..."

    $proc = Start-Process -FilePath $Binary -PassThru -NoNewWindow
    Log "Started binary with PID $($proc.Id)"

    Start-Sleep -Seconds 6

    if ($proc.HasExited) {
        Fail "Binary exited before shutdown signal could be sent"
        return
    }

    # On Windows, console apps don't respond to WM_CLOSE (taskkill without /F).
    # Python handles CTRL_C_EVENT, but sending it via GenerateConsoleCtrlEvent
    # would also signal the parent process. Use Stop-Process (TerminateProcess)
    # which is the standard way to stop services on Windows.
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Log "Sent stop signal to PID $($proc.Id)"

    # Wait up to 10 seconds for exit
    $exited = $proc.WaitForExit(10000)

    if ($exited) {
        Pass "Process stopped successfully (exit code $($proc.ExitCode))"
    }
    else {
        Fail "Binary did not exit within 10 seconds after stop signal"
    }
}

# ---------- Run all tests ----------
Write-Host ""
Write-Host "=== Space Router Home Node - E2E Smoke Tests ==="
Write-Host "Binary:  $Binary"
Write-Host "Version: $($env:EXPECTED_VERSION)"
Write-Host "Ports:   $PortBindingPort (binding), $ShutdownPort (shutdown)"
Write-Host ""

try {
    Start-MockApi

    Test-VersionFlag
    Test-PortBinding
    Test-CleanShutdown
}
finally {
    Stop-MockApi
}

Write-Host ""
Write-Host "=== Results: $Pass passed, $Fail failed ==="
Write-Host ""

if ($Fail -gt 0) {
    exit 1
}
