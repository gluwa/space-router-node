# Code Signing Setup

This document describes how to configure code signing for Space Router Home Node
release binaries. The CI pipeline includes conditional signing steps that are
**silently skipped** when secrets are not configured — no changes are needed to
enable or disable them.

## macOS Code Signing & Notarization

### Prerequisites

1. An **Apple Developer ID Application** certificate (for signing binaries)
2. An **Apple Developer ID Installer** certificate (for signing `.pkg` installers)
3. An Apple ID enrolled in the Apple Developer Program
4. An **app-specific password** for notarization (generated at appleid.apple.com)

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `APPLE_CERTIFICATE_P12` | Base64-encoded `.p12` file containing the Developer ID Application certificate and private key |
| `APPLE_CERTIFICATE_PASSWORD` | Password used when exporting the `.p12` file |
| `APPLE_INSTALLER_CERTIFICATE_P12` | Base64-encoded `.p12` file containing the Developer ID Installer certificate |
| `APPLE_INSTALLER_CERTIFICATE_PASSWORD` | Password for the installer `.p12` file |
| `APPLE_TEAM_ID` | 10-character Apple Developer Team ID |
| `APPLE_ID` | Apple ID email used for notarization |
| `APPLE_ID_PASSWORD` | App-specific password for notarization |

### Encoding certificates as Base64

```bash
# Export from Keychain Access as .p12, then encode:
base64 -i DeveloperIDApplication.p12 | pbcopy
# Paste into GitHub Secret APPLE_CERTIFICATE_P12

base64 -i DeveloperIDInstaller.p12 | pbcopy
# Paste into GitHub Secret APPLE_INSTALLER_CERTIFICATE_P12
```

### What the CI does

1. Decodes the `.p12` from the secret and imports it into a temporary keychain
2. Signs the binary with `codesign --force --options runtime` (hardened runtime)
3. Signs the `.pkg` with `productsign` (if macOS installer PR is merged)
4. Submits to Apple for notarization via `notarytool submit --wait`
5. Staples the notarization ticket to the binary/pkg via `xcrun stapler`
6. Cleans up the temporary keychain

---

## Windows Code Signing

### Prerequisites

1. A **code signing certificate** (EV or standard) from a trusted CA
   (DigiCert, Sectigo, GlobalSign, etc.)
2. The certificate exported as a `.pfx` / `.p12` file with private key

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `WINDOWS_CERTIFICATE_PFX` | Base64-encoded `.pfx` file containing the signing certificate and private key |
| `WINDOWS_CERTIFICATE_PASSWORD` | Password for the `.pfx` file |

### Encoding the certificate as Base64

```powershell
# PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("certificate.pfx")) | Set-Clipboard

# Or on Linux/macOS
base64 -i certificate.pfx | pbcopy
```

### What the CI does

1. Decodes the `.pfx` from the secret to a temporary file
2. Signs `space-router-node.exe` with `signtool sign` using SHA-256 and a
   DigiCert timestamp server
3. Signs the `-setup.exe` NSIS installer (if Windows installer PR is merged)
4. Deletes the temporary `.pfx` file

### EV certificates with hardware tokens

EV certificates stored on USB tokens (e.g., SafeNet) require special CI
setup and are not supported by the current scaffolding. For EV signing,
consider cloud-based signing services like:

- **Azure Trusted Signing**
- **DigiCert KeyLocker**
- **AWS CloudHSM**

---

## Verifying signatures

### macOS

```bash
# Verify code signature
codesign --verify --verbose=2 space-router-node

# Check notarization
spctl --assess --type execute space-router-node
xcrun stapler validate space-router-node
```

### Windows

```powershell
# Check digital signature
Get-AuthenticodeSignature .\space-router-node.exe

# Or using signtool
signtool verify /pa /v space-router-node.exe
```
