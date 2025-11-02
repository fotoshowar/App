# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- ANÁLISIS ---
# Importamos la librería para poder encontrar su ruta de instalación
import face_recognition_models
import os

# Obtenemos la ruta a la carpeta 'models' DENTRO de la librería instalada
model_source_path = os.path.join(os.path.dirname(face_recognition_models.__file__), 'models')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # --- ARCHIVOS Y CARPETAS DE DATOS ---
        # Estos son los archivos que son parte de TU proyecto.
        ('static', 'static'),
        ('models', 'models'),
        
        # --- ¡SOLUCIÓN DEFINITIVA Y GARANTIZADA! ---
        # Añadimos la carpeta de modelos de la librería usando la ruta que encontramos.
        # Formato: ('ruta_origen_real', 'ruta_destino_en_el_ejecutable')
        (model_source_path, 'face_recognition_models/models'),
    ],
    # --- ¡ELIMINAMOS collect_data! ---
    # Ya no lo necesitamos porque estamos añadiendo los datos manualmente.
    hiddenimports=[
        # ... (el resto de tu lista de hiddenimports se queda igual)
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
# Dividimos la creación en dos partes para mayor fiabilidad y velocidad.

# 1. Creamos el pequeño archivo ejecutable principal.
exe = EXE(
    pyz,
    a.scripts,
    [], # No incluimos binarios o datos aquí directamente.
    exclude_binaries=True, # <-- ¡LÍNEA CLAVE! Excluye las librerías pesadas del .exe.
    name='face_recognition_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # Comprime el ejecutable para que sea más pequeño.
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # <-- ¡MANTENER EN TRUE! Muestra la consola para ver errores y logs.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 2. Creamos la carpeta final (COLLECT) que contiene TODO lo necesario.
coll = COLLECT(
    exe,                     # El ejecutable principal.
    a.binaries,              # Todas las librerías (.dll, .so, .dylib).
    a.zipfiles,              # Archivos comprimidos.
    a.datas,                 # Las carpetas 'static', 'models', 'config.ini', etc.
    strip=False,
    upx=True,
    upx_exclude=[],
    name='face_recognition_app', # <-- Nombre de la carpeta final que se creará.
)
