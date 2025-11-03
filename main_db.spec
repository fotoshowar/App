# main_db.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- DATOS A INCLUIR ---
# Incluimos la carpeta estática y el nuevo archivo app.py
added_files = [
    ('static', 'static'),
    ('app.py', '.'),
]

a = Analysis(
    ['main_db.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'uvicorn',
        'aiosqlite',
        'PIL',
        'opencv-python',
        'cryptography',
        'hkdf',
        'numpy',
        'pyparsing',
        # Módulos de ChromaDB
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
