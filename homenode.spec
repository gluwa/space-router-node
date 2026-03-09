# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SpaceRouter Home Node."""

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

hiddenimports = [
    # Conditionally imported at runtime
    "miniupnpc",
    # pydantic v2 uses a Rust-compiled core loaded dynamically
    "pydantic",
    "pydantic_core",
    "pydantic_settings",
    "pydantic_settings.main",
    # dotenv loaded by pydantic-settings
    "dotenv",
    # httpx transport stack
    "httpx",
    "httpcore",
    "h11",
    "certifi",
    "idna",
    "sniffio",
    "anyio",
    "anyio._backends._asyncio",
    # cryptography internals sometimes missed
    "cryptography.hazmat.primitives.asymmetric.rsa",
    "cryptography.hazmat.primitives.hashes",
    "cryptography.hazmat.primitives.serialization",
    "cryptography.x509",
    "cryptography.x509.oid",
]

# Collect all pydantic submodules to handle dynamic imports
hiddenimports += collect_submodules("pydantic")
hiddenimports += collect_submodules("pydantic_core")

a = Analysis(
    ["app/main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "pytest_asyncio",
        "respx",
        "_pytest",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="space-router-node",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
