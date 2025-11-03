# main_db.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- DATOS A INCLUIR ---
# Incluimos la carpeta estática y el nuevo archivo app.py
added_files = [
    ('static', 'static'),
    ('app.py', '.'), # <-- ¡IMPORTANTE! Incluye el archivo de la aplicación
]

a = Analysis(
    ['main_db.py'], # <-- El punto de entrada es el nuevo lanzador
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'uvicorn',
        'aiosqlite',
        'chromadb',
        'PIL',
        'opencv-python',
        'cryptography',
        'hkdf',
        'numpy',
        'pyparsing',
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
    name='face_recognition_db', # <-- Un nuevo nombre para el ejecutable
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
