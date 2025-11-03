# main_db.py
import sys
import traceback
import uvicorn
from pathlib import Path

# Importamos la aplicación desde el otro archivo
from app import app

def get_application_path():
    """Obtiene la ruta base de la aplicación de forma robusta."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent.resolve()

if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition API v5.3.0-SQLite-DB-Only")
    print("=" * 60)
    print(f"Directorio de la aplicación: {get_application_path()}")
    print(f"Servidor corriendo en: http://localhost:8888")
    print("Presiona Ctrl+C para detener el servidor.")
    print("=" * 60)
    
    try:
        # Ejecuta la aplicación importada
        uvicorn.run(
            app,  # <-- Usamos el objeto 'app' importado
            host="0.0.0.0",
            port=8888,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print("\n" + "=" * 50)
        print("¡OCURRIÓ UN ERROR!")
        print("=" * 50)
        traceback.print_exc()
        print("=" * 50)
    finally:
        print("\nEl programa se ha detenido. Presiona Enter para salir...")
        input()
