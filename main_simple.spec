# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- DATOS A INCLUIR ---
# Aquí es donde le decimos a PyInstaller que incluya nuestra carpeta 'static'
# El formato es: ('ruta_origen', 'ruta_destino_en_el_ejecutable')
added_files = [
    ('static', 'static'),
]

a = Analysis(
    ['main_simple.py'],
    pathex=[],
    binaries=[],
    datas=added_files,  # <-- Usamos nuestra lista de archivos
    hiddenimports=[
        'uvicorn',
        'uvicorn.lifespan.on',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
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
    name='face_recognition_simple', # Nombre del ejecutable
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # <-- ¡IMPORTANTE! Mantiene la ventana de consola visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
