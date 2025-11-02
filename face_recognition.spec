# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- CÁLCULO DE RUTAS ANTES DE ANÁLISIS ---
# Aquí sí podemos usar código Python normal para calcular la ruta.
import face_recognition_models
import os
model_source_path = os.path.join(os.path.dirname(face_recognition_models.__file__), 'models')

# --- ANÁLISIS ---
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('static', 'static'),
        ('models', 'models'),
        # --- ¡SOLUCIÓN DEFINITIVA Y GARANTIZADA! ---
        # Usamos la variable que calculamos antes.
        (model_source_path, 'face_recognition_models/models'),
    ],
    hiddenimports=[
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'starlette',
        'face_recognition_models',
        'dlib',
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'sklearn',
        'scipy',
        'scipy.spatial.distance',
        'sklearn.metrics.pairwise',
        'insightface',
        'facenet_pytorch',
        'PIL',
        'aiosqlite',
        'cryptography',
        'hkdf',
        'chromadb',
        'httpx',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
),

# --- CREACIÓN DE LOS COMPONENTES ---
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- ESTRUCTURA DE EJECUTABLE Y CARPETA PORTÁTIL ---
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='face_recognition_app',
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

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='face_recognition_app',
)
