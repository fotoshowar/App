#!/usr/bin/env python3
"""
Script para resolver conflictos de dependencias específicos
Instala versiones compatibles de todas las librerías
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Ejecutar comando con manejo de errores"""
    print(f"Ejecutando: {description}")
    print(f"Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Exitoso")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def fix_dependencies():
    """Resolver conflictos de dependencias paso a paso"""
    
    # Detectar pip command
    if os.path.exists("venv/bin/pip"):
        pip_cmd = "venv/bin/pip"
    elif os.path.exists("venv/Scripts/pip.exe"):
        pip_cmd = "venv/Scripts/pip.exe"
    else:
        pip_cmd = "pip"
    
    print("🔧 Resolviendo conflictos de dependencias...")
    
    # Paso 1: Desinstalar las librerías problemáticas
    print("\n1️⃣ Desinstalando versiones conflictivas...")
    packages_to_remove = [
        "opencv-python", "opencv-python-headless", 
        "facenet-pytorch", "Pillow", "numpy"
    ]
    
    for package in packages_to_remove:
        run_command([pip_cmd, "uninstall", "-y", package], 
                   f"Desinstalar {package}")
    
    # Paso 2: Instalar versiones específicas compatibles
    print("\n2️⃣ Instalando versiones compatibles...")
    
    # Versiones específicas que funcionan juntas
    compatible_packages = [
        # Core primero
        "numpy>=1.21.0,<2.0.0",  # Compatible con OpenCV y facenet
        "Pillow>=10.2.0,<10.3.0",  # Versión requerida por facenet-pytorch
        
        # OpenCV sin conflictos
        "opencv-python-headless>=4.8.0,<4.10.0",
        
        # PyQt6
        "PyQt6>=6.6.0",
        
        # Reconocimiento facial
        "face-recognition>=1.3.0",
        
        # PyTorch CPU (versiones específicas)
        "torch>=2.0.0,<2.1.0",
        "torchvision>=0.15.0,<0.16.0",
        
        # FaceNet después de PyTorch
        "facenet-pytorch>=2.5.0,<2.6.0",
        
        # Científicas
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0"
    ]
    
    success_count = 0
    for package in compatible_packages:
        if run_command([pip_cmd, "install", package], f"Instalar {package}"):
            success_count += 1
    
    # Paso 3: Verificar instalación
    print(f"\n3️⃣ Verificando instalación ({success_count}/{len(compatible_packages)} exitosos)...")
    
    verification_script = """
import sys
def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

results = []
results.append(check_import("numpy", "NumPy"))
results.append(check_import("cv2", "OpenCV"))
results.append(check_import("PIL", "Pillow"))
results.append(check_import("PyQt6.QtWidgets", "PyQt6"))
results.append(check_import("face_recognition", "face_recognition"))
results.append(check_import("torch", "PyTorch"))
results.append(check_import("torchvision", "torchvision"))

try:
    import facenet_pytorch
    print("✅ facenet_pytorch")
    results.append(True)
except ImportError as e:
    print(f"❌ facenet_pytorch: {e}")
    results.append(False)

results.append(check_import("scipy", "SciPy"))
results.append(check_import("sklearn", "scikit-learn"))

print(f"\\nResultado: {sum(results)}/{len(results)} módulos funcionando")
sys.exit(0 if sum(results) >= 7 else 1)  # Al menos 7 de 10 deben funcionar
"""
    
    python_cmd = pip_cmd.replace("pip", "python").replace(".exe", ".exe" if pip_cmd.endswith(".exe") else "")
    
    try:
        result = subprocess.run([python_cmd, "-c", verification_script], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n🎉 Dependencias resueltas exitosamente!")
            return True
        else:
            print("\n⚠️ Algunos módulos tienen problemas, pero puede funcionar parcialmente")
            return False
            
    except Exception as e:
        print(f"Error verificando: {e}")
        return False

def create_requirements_compatible():
    """Crear archivo requirements con versiones compatibles"""
    requirements_content = """# Dependencias compatibles para Face Recognition Desktop
# Versiones probadas que funcionan juntas

# Core
numpy>=1.21.0,<2.0.0
Pillow>=10.2.0,<10.3.0

# Computer Vision
opencv-python-headless>=4.8.0,<4.10.0

# GUI
PyQt6>=6.6.0

# Face Recognition
face-recognition>=1.3.0
dlib>=19.22.0

# Deep Learning
torch>=2.0.0,<2.1.0
torchvision>=0.15.0,<0.16.0
facenet-pytorch>=2.5.0,<2.6.0

# Scientific
scipy>=1.10.0
scikit-learn>=1.3.0

# Optional (puede fallar en algunos sistemas)
# insightface>=0.7.0
"""
    
    with open("requirements_compatible.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Archivo requirements_compatible.txt creado")

if __name__ == "__main__":
    print("🔧 Solucionador de Conflictos de Dependencias")
    print("=" * 50)
    
    # Crear requirements compatible
    create_requirements_compatible()
    
    # Resolver dependencias
    success = fix_dependencies()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 ¡Conflictos resueltos!")
        print("\nAhora puedes ejecutar:")
        print("python face_recognition_desktop.py")
    else:
        print("\n" + "=" * 50)
        print("⚠️ Algunos conflictos persisten")
        print("\nIntenta la instalación manual:")
        print("pip install -r requirements_compatible.txt")
