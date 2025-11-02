#!/usr/bin/env python3
"""
Script de configuración e instalación para Face Recognition Desktop
Automatiza la instalación de dependencias y configuración inicial
"""

import os
import sys
import subprocess
import platform
import sqlite3
from pathlib import Path

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - OK")
    return True

def create_virtual_env():
    """Crear entorno virtual"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Entorno virtual ya existe")
        return True
    
    try:
        print("📦 Creando entorno virtual...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Entorno virtual creado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creando entorno virtual: {e}")
        return False

def get_pip_command():
    """Obtener comando pip correcto para el entorno virtual"""
    system = platform.system()
    if system == "Windows":
        return str(Path("venv") / "Scripts" / "pip.exe")
    else:
        return str(Path("venv") / "bin" / "pip")

def install_dependencies():
    """Instalar dependencias"""
    pip_cmd = get_pip_command()
    
    # Verificar que pip existe
    if not Path(pip_cmd).exists():
        print("❌ Error: No se encontró pip en el entorno virtual")
        return False
    
    try:
        print("📦 Actualizando pip...")
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        print("📦 Instalando dependencias principales...")
        subprocess.run([pip_cmd, "install", "--upgrade", "setuptools", "wheel"], check=True)
        
        # Instalar PyQt6 primero
        print("📦 Instalando PyQt6...")
        subprocess.run([pip_cmd, "install", "PyQt6==6.7.1"], check=True)
        
        # Instalar dependencias de computer vision
        print("📦 Instalando OpenCV y NumPy...")
        subprocess.run([pip_cmd, "install", "opencv-python==4.8.1.78", "numpy==1.24.3"], check=True)
        
        # Instalar dependencias de reconocimiento facial
        print("📦 Instalando bibliotecas de reconocimiento facial...")
        subprocess.run([pip_cmd, "install", "face-recognition==1.3.0"], check=True)
        
        print("📦 Instalando modelos de deep learning...")
        subprocess.run([pip_cmd, "install", "torch==2.0.1", "torchvision==0.15.2"], check=True)
        subprocess.run([pip_cmd, "install", "facenet-pytorch==2.5.3"], check=True)
        
        print("📦 Instalando utilidades científicas...")
        subprocess.run([pip_cmd, "install", "scipy==1.11.1", "scikit-learn==1.3.0"], check=True)
        
        # InsightFace puede ser problemático, instalarlo al final
        print("📦 Instalando InsightFace (opcional)...")
        try:
            subprocess.run([pip_cmd, "install", "insightface==0.7.3"], check=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("⚠️  InsightFace no se pudo instalar, continuando sin él...")
        
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_directories():
    """Crear directorios necesarios"""
    directories = ["data", "uploads", "uploads/photos", "uploads/faces", "models"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Directorio creado: {dir_name}")
    
    return True

def setup_database():
    """Configurar base de datos inicial"""
    db_path = Path("data") / "face_recognition.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        
        # Crear tabla de fotos
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                faces_count INTEGER DEFAULT 0,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                image_width INTEGER,
                image_height INTEGER
            )
        ''')
        
        # Crear tabla de caras
        conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                face_path TEXT NOT NULL,
                embeddings TEXT NOT NULL,
                bounding_box TEXT NOT NULL,
                confidence REAL NOT NULL,
                detection_method TEXT,
                landmarks TEXT,
                face_quality_score REAL DEFAULT 0.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
            )
        ''')
        
        # Crear índices
        conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces(photo_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_faces_confidence ON faces(confidence)')
        
        conn.commit()
        conn.close()
        
        print("✅ Base de datos configurada")
        return True
        
    except Exception as e:
        print(f"❌ Error configurando base de datos: {e}")
        return False

def create_run_script():
    """Crear script para ejecutar la aplicación"""
    system = platform.system()
    
    if system == "Windows":
        script_content = """@echo off
echo Iniciando Face Recognition Desktop...
venv\\Scripts\\python.exe face_recognition_desktop.py
pause
"""
        script_name = "run_face_recognition.bat"
    else:
        script_content = """#!/bin/bash
echo "Iniciando Face Recognition Desktop..."
./venv/bin/python face_recognition_desktop.py
"""
        script_name = "run_face_recognition.sh"
    
    script_path = Path(script_name)
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if system != "Windows":
        os.chmod(script_path, 0o755)
    
    print(f"✅ Script de ejecución creado: {script_name}")
    return True

def test_installation():
    """Probar que la instalación funciona"""
    print("🧪 Probando instalación...")
    
    system = platform.system()
    python_cmd = str(Path("venv") / ("Scripts" if system == "Windows" else "bin") / "python")
    
    test_script = """
import sys
try:
    import PyQt6
    print("✅ PyQt6 OK")
    
    import cv2
    print("✅ OpenCV OK")
    
    import numpy
    print("✅ NumPy OK")
    
    import face_recognition
    print("✅ face_recognition OK")
    
    import torch
    print("✅ PyTorch OK")
    
    try:
        import insightface
        print("✅ InsightFace OK")
    except:
        print("⚠️  InsightFace no disponible (opcional)")
    
    print("✅ Todas las dependencias principales están disponibles")
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ Error importando: {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run([python_cmd, "-c", test_script], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print("Advertencias:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error probando instalación: {e}")
        return False

def main():
    """Función principal de configuración"""
    print("🔧 Face Recognition Desktop - Configuración Automática")
    print("=" * 60)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Crear entorno virtual
    if not create_virtual_env():
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        return False
    
    # Configurar directorios
    if not setup_directories():
        return False
    
    # Configurar base de datos
    if not setup_database():
        return False
    
    # Crear script de ejecución
    if not create_run_script():
        return False
    
    # Probar instalación
    if not test_installation():
        print("⚠️  La instalación puede tener problemas")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ¡Configuración completada exitosamente!")
    print("\nPara ejecutar la aplicación:")
    
    if platform.system() == "Windows":
        print("  - Ejecuta: run_face_recognition.bat")
        print("  - O ejecuta: venv\\Scripts\\python face_recognition_desktop.py")
    else:
        print("  - Ejecuta: ./run_face_recognition.sh")
        print("  - O ejecuta: ./venv/bin/python face_recognition_desktop.py")
    
    print("\n📝 Notas importantes:")
    print("  - La primera ejecución puede ser lenta (descarga de modelos)")
    print("  - Se requiere cámara web para búsqueda en tiempo real")
    print("  - Los datos se almacenan localmente en la carpeta 'data'")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ La configuración falló. Revisa los errores anteriores.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Configuración cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
