# main_with_db.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- DATOS A INCLUIR ---
# Incluimos las carpetas necesarias para el funcionamiento
added_files = [
    ('static', 'static'),
    # La carpeta 'uploads' se creará dinámicamente, no es necesario incluirla
]

a = Analysis(
    ['main_with_db.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'uvicorn',
        'uvicorn.lifespan.on',
        'uvicorn.protocols.http.h11_impl',
        'aiosqlite',
        'chromadb',
        'PIL',
        'PIL._imagingtk',
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
    name='face_recognition_with_db',
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
