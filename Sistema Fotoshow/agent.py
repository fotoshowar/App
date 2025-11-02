# agent.py
import os
import sys
import threading
from pathlib import Path

import webview
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# Importar nuestra lógica
from backend_logic import DesktopAgent

# --- Configuración de la App ---
WEB_INTERFACE_PATH = Path(__file__).parent / "web_interface"
ICON_PATH = Path(__file__).parent / "icon.ico"

# --- Instancia Global del Agente ---
agent = DesktopAgent()

# --- API FastAPI para comunicación con la UI ---
api = FastAPI()

@api.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(WEB_INTERFACE_PATH / "index.html")

@api.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    return FileResponse(WEB_INTERFACE_PATH / file_path)

@api.get("/api/config")
async def get_config():
    return agent.config.__dict__

@api.post("/api/config")
async def update_config(request: Request):
    data = await request.json()
    agent.update_config(data)
    return {"success": True}

@api.get("/api/status")
async def get_status():
    return agent.get_status()

@api.get("/api/read-log")
async def read_log(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "Log file not found."

# --- Funciones expuestas al frontend de PyWebView ---
class ApiForJS:
    def select_folder(self):
        file_types = ('Folder (*.*)', '*.*')
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG, allow_multiple=False, file_types=file_types)
        return result[0] if result else None

    def get_log_file(self):
        return agent.get_status()['log_file']

def start_server():
    """Inicia el servidor FastAPI en un hilo separado."""
    # `log_level="critical"` para que uvicorn no ensucie la consola
    uvicorn.run(api, host="127.0.0.1", port=8765, log_level="critical")

def main():
    # Iniciar el backend del agente
    agent.start()

    # Iniciar el servidor web en un hilo daemon
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Darle tiempo al servidor para que arranque
    import time
    time.sleep(1.5)

    # Crear la ventana de la aplicación
    window = webview.create_window(
        'FotoShow Agent',
        'http://127.0.0.1:8765',
        js_api=ApiForJS(),
        width=900,
        height=700,
        resizable=True,
        min_size=(600, 400),
        icon=str(ICON_PATH) if ICON_PATH.exists() else None
    )
    
    # Iniciar el bucle de PyWebView (esto es bloqueante)
    webview.start(debug=False) # Poner debug=True en desarrollo
    
    # Cuando se cierra la ventana, detener el backend
    agent.stop()

if __name__ == "__main__":
    main()
