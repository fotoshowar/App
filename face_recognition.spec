# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# --- ANÁLISIS ---
# PyInstaller analiza main.py y sigue las importaciones.
# SÍ, analizará advanced_face_processor.py y sus dependencias como cv2, torch, dlib, etc.
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # --- CRÍTICO PARA advanced_face_processor.py ---
        # Tu procesador espera una carpeta 'models' para guardar los archivos .dat de dlib.
        # Esta línea le dice a PyInstaller: "Copia la carpeta 'models' del proyecto
        # al directorio final de la aplicación".
        ('models', 'models'),

        # También necesitas las carpetas estáticas y de datos.
        ('static', 'static'),
        # Si quieres que las carpetas 'uploads' y 'faces' existan al descomprimir, añádelas aquí.
        # ('uploads', 'uploads'),
        # ('faces', 'faces'),
    ],
    hiddenimports=[
        # --- LA RED DE SEGURIDAD ---
        # Esta es una lista de módulos que PyInstaller podría no encontrar automáticamente.
        # Incluimos aquí las dependencias más problemáticas de advanced_face_processor.py.

        # Dependencias directas de advanced_face_processor.py
        'cv2',
        'numpy',
        'face_recognition',
        'torch',
        'torchvision',
        'dlib',
        'scipy.spatial.distance',
        'sklearn.metrics.pairwise',
        'insightface',
        'facenet_pytorch',
        'PIL', # Pillow, usado por torchvision

        # Dependencias de FastAPI/Uvicorn (para que el servidor web funcione)
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'starlette',

        # Dependencias de base de datos y utilidades
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

# --- CREACIÓN DEL EJECUTABLE ---
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- ¡CAMBIO CLAVE! ---
# Dividimos la creación en dos partes: EXE y COLLECT.

# 1. Creamos el pequeño archivo ejecutable principal
exe = EXE(
    pyz,
    a.scripts,
    [], # <-- OJO: Aquí quitamos a.binaries, a.zipfiles y a.datas
    exclude_binaries=True, # <-- ¡LÍNEA CLAVE! No incluye las librerías pesadas aquí.
    name='face_recognition_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # UPX funciona bien en este modo
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # ¡Mantén en True para ver los logs y errores al depurar!
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 2. Creamos la carpeta final (COLLECT) que contiene todo
coll = COLLECT(
    exe, # El ejecutable que acabamos de crear
    a.binaries, # Todas las librerías (.dll, .so, .dylib)
    a.zipfiles, # Archivos comprimidos
    a.datas, # Las carpetas 'static' y 'models'
    strip=False,
    upx=True,
    upx_exclude=[],
    name='face_recognition_app', # Nombre de la carpeta final
)
