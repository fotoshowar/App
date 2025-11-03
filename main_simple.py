# main_simple.py
import sys
import traceback
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- CONFIGURACIÓN DE RUTAS (MEJORADA) ---
def get_application_path():
    """Obtiene la ruta base de la aplicación de forma robusta."""
    if getattr(sys, 'frozen', False):
        # En modo ejecutable, la carpeta de recursos está en sys._MEIPASS
        # pero la carpeta 'static' debería estar al lado del .exe
        return Path(sys.executable).parent
    else:
        # En modo script, es el directorio del script
        return Path(__file__).parent.resolve()

APPLICATION_PATH = get_application_path()
STATIC_DIR = APPLICATION_PATH / "static"

# --- INICIALIZACIÓN DE LA APLICACIÓN ---
app = FastAPI(title="Face Recognition Frontend", version="1.0.0-Simple")

# --- MIDDLEWARE ---
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
        print(f"Buscando index.html en: {html_path}") # Línea de depuración
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse(
                f"<h1>Error 404</h1><p>No se encontró index.html en la carpeta 'static'.</p><p>Ruta buscada: {html_path}</p>",
                status_code=404
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error del servidor</h1><p>{e}</p>", status_code=500)

@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    """Sirve el panel de administración."""
    try:
        html_path = STATIC_DIR / "html_update_ngrok.html"
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
if STATIC_DIR.exists():
    print(f"Montando archivos estáticos desde: {STATIC_DIR}") # Línea de depuración
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"ADVERTENCIA: La carpeta 'static' no fue encontrada en {STATIC_DIR}")

# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 50)
    print("Iniciando Servidor Simple (Frontend Only)")
    print("=" * 50)
    print(f"Directorio de la aplicación: {APPLICATION_PATH}")
    print(f"Directorio de archivos estáticos: {STATIC_DIR}")
    print(f"Servidor corriendo en: http://localhost:8888")
    print("Presiona Ctrl+C para detener el servidor.")
    print("=" * 50)

    try:
        uvicorn.run(
            app,
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
