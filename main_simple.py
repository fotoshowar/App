# main_simple.py
import sys
import traceback
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- CONFIGURACIÓN DE RUTAS ---
# Detecta si estamos en un ejecutable o en un script para encontrar la carpeta 'static'
if getattr(sys, 'frozen', False):
    # En modo ejecutable, la carpeta 'static' debe estar al lado del .exe
    APPLICATION_PATH = Path(sys.executable).parent
else:
    # En modo script, la carpeta 'static' está en el mismo directorio que el script
    APPLICATION_PATH = Path(__file__).parent

STATIC_DIR = APPLICATION_PATH / "static"

# --- INICIALIZACIÓN DE LA APLICACIÓN ---
app = FastAPI(title="Face Recognition Frontend", version="1.0.0-Simple")

# --- MIDDLEWARE ---
# Permitir peticiones desde cualquier origen (útil para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS PRINCIPALES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Sirve la página principal index.html."""
    try:
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse(
                "<h1>Error 404</h1><p>No se encontró index.html en la carpeta 'static'.</p>",
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error del servidor</h1><p>{e}</p>", status_code=500)

@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    """Sirve el panel de administración."""
    try:
        html_path = STATIC_DIR / "html_update_ngrok.html" # Asegúrate que este archivo exista
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse(
                "<h1>Error 404</h1><p>No se encontró el panel de admin en la carpeta 'static'.</p>",
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error del servidor</h1><p>{e}</p>", status_code=500)

# --- MONTAR ARCHIVOS ESTÁTICOS ---
# Sirve todo el contenido de la carpeta 'static' en la ruta /static
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"ADVERTENCIA: La carpeta 'static' no fue encontrada en {STATIC_DIR}")

# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 50)
    print("Iniciando Servidor Simple (Frontend Only)")
    print("=" * 50)
    print(f"Directorio de la aplicación: {APPLICATION_PATH}")
    print(f"Servidor corriendo en: http://localhost:8888")
    print("Presiona Ctrl+C para detener el servidor.")
    print("=" * 50)

    try:
        # Inicia el servidor de uvicorn
        uvicorn.run(
            "main_simple:app",
            host="0.0.0.0",  # Escucha en todas las interfaces de red
            port=8888,
            reload=False,     # No recargar automáticamente en modo ejecutable
            log_level="info"
        )
    except Exception as e:
        # Si ocurre cualquier error, lo imprime en pantalla
        print("\n" + "=" * 50)
        print("¡OCURRIÓ UN ERROR!")
        print("=" * 50)
        traceback.print_exc() # Imprime el error completo
        print("=" * 50)
    finally:
        # Este bloque se ejecuta siempre, haya error o no.
        # Evita que la consola se cierre inmediatamente.
        print("\nEl programa se ha detenido. Presiona Enter para salir...")
        input()
