# build_exe.py (versión corregida)
import subprocess
import sys
import os
import platform # Importamos platform para detectar el SO

def build():
    """Construye el ejecutable con PyInstaller."""
    print("🔨 Iniciando construcción del ejecutable...")
    
    # Rutas
    project_dir = os.getcwd()
    icon_path = os.path.join(project_dir, 'icon.ico')
    
    # Comando PyInstaller base
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",  # Sin consola. En Linux, podrías usar "--console" para depurar si falla.
        "--name=FotoshowAgent",
    ]

    # --- CORRECCIÓN CLAVE ---
    # Determinamos el separador correcto para --add-data
    # En Linux/macOS es ':', en Windows es ';'
    data_separator = ':'
    if platform.system() == 'Windows':
        data_separator = ';'
    
    # Añadir los datos de la interfaz web con el separador correcto
    pyinstaller_cmd.append(f"--add-data=web_interface{data_separator}web_interface")

    # Añadir icono solo si existe
    if os.path.exists(icon_path):
        pyinstaller_cmd.append(f"--icon={icon_path}")
        print(f"✅ Icono encontrado en {icon_path}")
    else:
        print("⚠️ Icono 'icon.ico' no encontrado. Se usará el icono por defecto.")

    # Hidden imports son cruciales
    hidden_imports = [
        "uvicorn", "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
        "webview", "aiohttp", "watchdog", "cv2", "numpy",
        "main", "backend_logic" # Añade tus propios módulos
    ]
    for imp in hidden_imports:
        pyinstaller_cmd.append(f"--hidden-import={imp}")

    # El script principal
    pyinstaller_cmd.append("agent.py")

    try:
        print(f"🛠️ Ejecutando comando: {' '.join(pyinstaller_cmd)}")
        subprocess.run(pyinstaller_cmd, check=True)
        print("\n✅ ¡Construcción completada!")
        print("📦 El ejecutable se encuentra en la carpeta 'dist/FotoshowAgent.exe'")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error durante la construcción: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Error: 'pyinstaller' no se encontró. Asegúrate de instalarlo con 'pip install pyinstaller'")
        sys.exit(1)

if __name__ == "__main__":
    build()
