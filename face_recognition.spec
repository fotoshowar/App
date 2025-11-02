# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- ANÁLISIS ---
# PyInstaller analiza main.py y sigue las importaciones.
# Esta sección encuentra todas las dependencias (librerías, scripts, etc.).
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # --- ARCHIVOS Y CARPETAS DE DATOS ---
        # Aquí le decimos a PyInstaller qué carpetas y archivos adicionales debe empaquetar.
        # Formato: ('ruta_origen', 'ruta_destino_en_el_ejecutable')

        # Copia la carpeta 'static' (con tus archivos HTML) a la raíz del ejecutable.
        ('static', 'static'),

        # Copia la carpeta 'models' (para los modelos de dlib que se descargan) a la raíz.
        ('models', 'models'),

        # Copia el archivo de configuración a la raíz del ejecutable.
        ('config.ini', '.'),

        # --- ¡CORRECCIÓN CLAVE PARA EL ERROR DE MODELOS! ---
        # Le dice a PyInstaller que busque la carpeta 'models' dentro de la librería
        # 'face_recognition_models' y la incluya. Esto resuelve el RuntimeError.
        ('face_recognition_models/models', 'face_recognition_models/models'),
    ],
    hiddenimports=[
        # --- LIBRERÍAS OCULTAS ---
        # PyInstaller a veces no encuentra estas librerías automáticamente.
        # Las listamos aquí para asegurarnos de que se incluyan.

        # Dependencias de FastAPI/Uvicorn
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'starlette',

        # Dependencias de Procesamiento Facial y ML
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
        'PIL', # Pillow, usada por torchvision y otras librerías

        # Dependencias de Base de Datos y Utilidades
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
)

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
