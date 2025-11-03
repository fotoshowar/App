# main_simple.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- DATOS A INCLUIR ---
# El formato es: ('ruta_origen', 'ruta_destino_en_el_ejecutable')
# Usamos una ruta absoluta para evitar cualquier duda
import os
import sys

# Obtenemos la ruta del directorio actual donde se ejecuta PyInstaller
current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

added_files = [
    (os.path.join(current_dir, 'static'), 'static'),
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
    name='face_recognition_simple',
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
