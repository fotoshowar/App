# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Analiza tu script principal y sus dependencias
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Añade aquí las carpetas que tu app necesita leer en tiempo de ejecución
        ('static', 'static'),  # Copia la carpeta 'static' a la carpeta de ejecución
        ('models', 'models'),  # Si tienes modelos pre-descargados
        # Si 'uploads' y 'faces' deben existir al iniciar, añádelos también.
        # ('uploads', 'uploads'),
        # ('faces', 'faces'),
    ],
    hiddenimports=[
        # Librerías que PyInstaller podría no encontrar automáticamente
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'face_recognition_models',
        'dlib',
        'torch',
        'torchvision',
        'numpy',
        'cv2',
        'sklearn',
        'scipy',
        'insightface',
        'facenet_pytorch',
        'aiosqlite',
        'cryptography',
        'hkdf',
        'chromadb',
        'httpx',
        'pydantic',
        'PIL', # Pillow, que usa torchvision
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

# Crea el archivo PYZ (Python Zip)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Crea el ejecutable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='face_recognition_app', # Nombre del ejecutable final
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True, # Comprime el ejecutable (puede dar problemas en algunos antivirus, puedes ponerlo en False)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # Muestra la consola para ver los logs. Pon en False para una app de ventana.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
