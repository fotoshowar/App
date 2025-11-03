# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Solo necesitamos añadir las carpetas de NUESTRO proyecto.
        ('static', 'static'),
        ('models', 'models'),
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
    hookspath=['hooks'], # <-- ¡LÍNEA CLAVE! Le decimos a PyInstaller que busque hooks en nuestra carpeta.
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
),

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
),

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
