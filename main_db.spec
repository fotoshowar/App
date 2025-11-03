# main_db.spec
# ...
# --- DATOS A INCLUIR ---
added_files = [
    ('static', 'static'),
    ('app.py', '.'),
]

# --- SECCIÓN BINARIOS (MEJORADA) ---
# PyInstaller buscará estas DLLs en las rutas comunes de instalación de Python y las incluirá.
binaries = [
    ('opencv_videoio_ffmpeg.dll', 'opencv_world.dll', 'opencv_python3.dll'),
]

# --- SECCIÓN HIDDENIMPORTS ---
a = Analysis(
    ['main_db.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'uvicorn',
        'aiosqlite',
        'PIL',
        'numpy',
        'pyparsing',
        'httpx',
        'face_recognition',
        'dlib',
        # Opcional
        'torch',
        'torchvision',
        'chromadb',
        'chromadb.api',
        'chromadb.api.segment',
        'chromadb.db',
        'chromadb.db.index',
        'chromadb.db.index.hnsw',
        'chromadb.db.index.hnsw.lib',
        'chromadb.utils',
        'chromadb.telemetry.product.posthog',
        'chromadb.api.rust',
        'chromadb.api.rust.abi',
        'chromadb.api.rust.utils',
        'chromadb.auth',
        'chromadb.auth.token_provider',
        'chromadb.auth.token_provider.simple_token_provider',
        'chromadb.auth.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='face_recognition_db',
    debug=True,
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
