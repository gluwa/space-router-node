; ============================================================================
; Space Router Home Node — NSIS Installer
; ============================================================================
; Builds a Windows installer that:
;   1. Installs binary + WinSW + config to Program Files
;   2. Creates data directories in ProgramData
;   3. Registers & starts the Windows Service via WinSW
;   4. Adds an inbound firewall rule for the node port
;   5. Registers in Add/Remove Programs with uninstaller
;
; Build:
;   makensis /DVERSION=1.0.0 /DBINARY=space-router-node.exe \
;            /DWINSW=WinSW-x64.exe installer.nsi
;
; Required defines:
;   VERSION  - Semantic version (no leading 'v')
;   BINARY   - Path to the compiled binary
;   WINSW    - Path to WinSW executable
; ============================================================================

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef BINARY
  !error "BINARY must be defined (path to space-router-node.exe)"
!endif
!ifndef WINSW
  !error "WINSW must be defined (path to WinSW executable)"
!endif

; ---------------------------------------------------------------------------
; General settings
; ---------------------------------------------------------------------------
Name "Space Router Home Node"
OutFile "space-router-node-${VERSION}-setup.exe"
InstallDir "$PROGRAMFILES64\SpaceRouter"
InstallDirRegKey HKLM "Software\SpaceRouter" "InstallDir"
RequestExecutionLevel admin
ShowInstDetails show
ShowUninstDetails show

; ---------------------------------------------------------------------------
; Version info embedded in the executable
; ---------------------------------------------------------------------------
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Space Router Home Node"
VIAddVersionKey "CompanyName" "Gluwa Inc."
VIAddVersionKey "FileDescription" "Space Router Home Node Installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Gluwa Inc."

; ---------------------------------------------------------------------------
; Pages
; ---------------------------------------------------------------------------
!include "MUI2.nsh"

!define MUI_ABORTWARNING

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ---------------------------------------------------------------------------
; Installer section
; ---------------------------------------------------------------------------
Section "Install" SecInstall
  SetOutPath "$INSTDIR"

  ; --- Stop existing service if upgrading ---
  IfFileExists "$INSTDIR\space-router-node-service.exe" 0 +3
    nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" stop'
    nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" uninstall'

  ; --- Install files ---
  File "${BINARY}"
  File /oname=space-router-node-service.exe "${WINSW}"
  File "space-router-node-service.xml"
  File /oname=spacerouter.env.default "..\spacerouter.env"

  ; --- Create data directories ---
  CreateDirectory "$COMMONPROGRAMDATA\SpaceRouter"
  CreateDirectory "$COMMONPROGRAMDATA\SpaceRouter\certs"
  CreateDirectory "$COMMONPROGRAMDATA\SpaceRouter\logs"

  ; --- Copy default config if not present ---
  IfFileExists "$COMMONPROGRAMDATA\SpaceRouter\spacerouter.env" +2
    CopyFiles /SILENT "$INSTDIR\spacerouter.env.default" "$COMMONPROGRAMDATA\SpaceRouter\spacerouter.env"

  ; --- Install and start the Windows Service ---
  nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" install'
  nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" start'

  ; --- Add firewall rule via netsh (works on all Windows versions) ---
  nsExec::ExecToLog 'netsh advfirewall firewall delete rule name="Space Router Home Node (TCP-In)"'
  nsExec::ExecToLog 'netsh advfirewall firewall add rule name="Space Router Home Node (TCP-In)" dir=in action=allow protocol=TCP localport=9090 program="$INSTDIR\space-router-node.exe" enable=yes'

  ; --- Write uninstaller ---
  WriteUninstaller "$INSTDIR\uninstall.exe"

  ; --- Add/Remove Programs registry entry ---
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "DisplayName" "Space Router Home Node"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "Publisher" "Gluwa Inc."
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode" \
    "NoRepair" 1

  ; --- Remember install dir ---
  WriteRegStr HKLM "Software\SpaceRouter" "InstallDir" "$INSTDIR"
SectionEnd

; ---------------------------------------------------------------------------
; Uninstaller section
; ---------------------------------------------------------------------------
Section "Uninstall"
  ; --- Stop and remove the service ---
  IfFileExists "$INSTDIR\space-router-node-service.exe" 0 +3
    nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" stop'
    nsExec::ExecToLog '"$INSTDIR\space-router-node-service.exe" uninstall'

  ; --- Remove firewall rule ---
  nsExec::ExecToLog 'netsh advfirewall firewall delete rule name="Space Router Home Node (TCP-In)"'

  ; --- Remove installed files ---
  Delete "$INSTDIR\space-router-node.exe"
  Delete "$INSTDIR\space-router-node-service.exe"
  Delete "$INSTDIR\space-router-node-service.xml"
  Delete "$INSTDIR\spacerouter.env.default"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"

  ; --- Remove registry entries ---
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\SpaceRouterHomeNode"
  DeleteRegKey HKLM "Software\SpaceRouter"

  ; --- Preserve user data ---
  ; Configuration and certs in ProgramData are NOT removed.
  ; Users can manually delete %ProgramData%\SpaceRouter if desired.
SectionEnd
