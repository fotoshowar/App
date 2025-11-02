# backend_logic.py
import os
import sys
import json
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import cv2
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Importa tu lógica existente ---
# PyInstaller necesita ayuda para encontrar los módulos en modo --onefile
try:
    from main import AdvancedFaceProcessorIntegration, safe_convert_for_json
except ImportError:
    # Si se ejecuta desde el .exe, sys._MEIPASS es la ruta temporal
    sys.path.append(os.path.join(sys._MEIPASS, ''))
    from main import AdvancedFaceProcessorIntegration, safe_convert_for_json

# --- Configuración ---
API_BASE_URL = "https://tu-servicio-en-la-nube.com" 
CONFIG_FILE = Path(os.environ['APPDATA']) / "FotoshowAgent" / "config.json"
LOG_FILE = Path(os.environ['LOCALAPPDATA']) / "FotoshowAgent" / "agent.log"

# Asegurar que los directorios existan
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AgentConfig:
    def __init__(self):
        self.api_key: str = ""
        self.photographer_id: str = ""
        self.monitored_folders: List[str] = []
        self.whatsapp_number: str = ""
        self.load()

    def load(self):
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                self.__dict__.update(json.load(f))
                logger.info("Configuración cargada.")

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info("Configuración guardada.")

class PhotoProcessor(FileSystemEventHandler):
    def __init__(self, agent: 'DesktopAgent'):
        self.agent = agent
        self.processor = AdvancedFaceProcessorIntegration()
        self.processing_queue = asyncio.Queue()
        self._is_running = True

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            logger.info(f"🖼️ Nuevo archivo detectado: {event.src_path}")
            asyncio.run_coroutine_threadsafe(self.processing_queue.put(event.src_path), self.agent.loop)

    async def process_queue(self):
        while self._is_running:
            try:
                photo_path = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                await self._process_photo(photo_path)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en el procesador de cola: {e}")

    async def _process_photo(self, photo_path: str):
        try:
            logger.info(f"⚙️ Procesando {photo_path}...")
            faces_data = self.processor.detect_and_encode_faces(photo_path, save_faces=False)
            
            if faces_data:
                photo_id = Path(photo_path).stem
                payload = {"photo_id": photo_id, "filepath": photo_path, "faces": []}
                for face in faces_data:
                    # Estandarizamos a un solo embedding para evitar problemas de dimensión
                    embedding_key = list(face['embeddings'].keys())[0]
                    payload["faces"].append({
                        "face_id": face['face_id'],
                        "bbox": face['bbox'],
                        "embedding": face['embeddings'][embedding_key],
                        "method": face['method'],
                        "confidence": face['confidence']
                    })
                
                success = await self.agent.api_client.upload_faces(payload)
                if success:
                    logger.info(f"✅ {len(faces_data)} caras de {photo_path} sincronizadas.")
                else:
                    logger.error(f"❌ Falló sincronización de {photo_path}.")
            else:
                logger.info(f"ℹ️ No se detectaron caras en {photo_path}.")

        except Exception as e:
            logger.error(f"Error procesando {photo_path}: {e}", exc_info=True)

class APIClient:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self):
        if self.session is None or self.session.closed:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(headers=headers, base_url=API_BASE_URL, timeout=timeout)

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def register(self, name: str, email: str, whatsapp: str) -> Dict:
        await self.ensure_session()
        payload = {"name": name, "email": email, "whatsapp_number": whatsapp}
        try:
            async with self.session.post("/api/agent/register", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.config.photographer_id = data["photographer_id"]
                    self.config.api_key = data["api_key"]
                    self.config.save()
                    logger.info(f"Agente registrado. ID: {self.config.photographer_id}")
                    return {"success": True, "photographer_id": self.config.photographer_id}
                else:
                    error_text = await resp.text()
                    logger.error(f"Error en registro: {resp.status} - {error_text}")
                    return {"success": False, "error": error_text}
        except Exception as e:
            logger.error(f"Error de conexión en registro: {e}")
            return {"success": False, "error": str(e)}

    async def ping(self) -> Dict:
        await self.ensure_session()
        try:
            async with self.session.get("/api/agent/ping") as resp:
                if resp.status == 200:
                    return {"status": "ok", "message": await resp.json()}
                return {"status": "error", "message": f"HTTP {resp.status}"}
        except Exception as e:
            logger.error(f"Error en ping: {e}")
            return {"status": "error", "message": str(e)}

    async def upload_faces(self, payload: Dict) -> bool:
        await self.ensure_session()
        try:
            async with self.session.post("/api/agent/update_faces", json=payload) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Error subiendo caras: {e}")
            return False

class DesktopAgent:
    def __init__(self):
        self.config = AgentConfig()
        self.api_client = APIClient(self.config)
        self.loop = asyncio.new_event_loop()
        self.observer = Observer()
        self.processor: Optional[PhotoProcessor] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._run_async_tasks, daemon=True)
        self._thread.start()
        logger.info("🚀 Backend del agente iniciado.")

    def stop(self):
        if not self._is_running:
            return
        logger.info("🛑 Deteniendo backend del agente...")
        self._is_running = False
        self.observer.stop()
        self.observer.join()
        if self.processor:
            self.processor._is_running = False
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("✅ Backend detenido.")

    def _run_async_tasks(self):
        asyncio.set_event_loop(self.loop)
        if self.config.api_key:
            self.loop.run_until_complete(self._main_loop())
        else:
            logger.info("⏳ Esperando configuración para iniciar el monitoreo.")
            while self._is_running and not self.config.api_key:
                asyncio.sleep(1)

    async def _main_loop(self):
        await self.api_client.ensure_session()
        self._start_folder_monitoring()
        if self.processor:
            asyncio.create_task(self.processor.process_queue())
        
        while self._is_running:
            try:
                ping_result = await self.api_client.ping()
                if ping_result['status'] == 'ok':
                    logger.debug("🏓 Ping exitoso al servidor.")
                else:
                    logger.warning(f"⚠️ Fallo en ping: {ping_result['message']}")
                await asyncio.wait_for(asyncio.sleep(30), timeout=35.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error en el bucle principal: {e}")
                await asyncio.sleep(60)

    def _start_folder_monitoring(self):
        if not self.processor:
            self.processor = PhotoProcessor(self)
        for folder_path in self.config.monitored_folders:
            if Path(folder_path).exists():
                self.observer.schedule(self.processor, folder_path, recursive=True)
                logger.info(f"📁 Monitoreando carpeta: {folder_path}")
            else:
                logger.warning(f"⚠️ La carpeta no existe: {folder_path}")
        self.observer.start()
        logger.info("👀 Monitor de carpetas iniciado.")

    def get_status(self):
        return {
            "is_running": self._is_running,
            "is_monitoring": self.observer.is_alive() if self.observer else False,
            "monitored_folders": self.config.monitored_folders,
            "photographer_name": getattr(self.config, 'name', 'No configurado'),
            "log_file": str(LOG_FILE)
        }

    def update_config(self, new_config_data: dict):
        self.config.__dict__.update(new_config_data)
        self.config.save()
        
        # Si se añadió una API key, reiniciar el monitoreo
        if new_config_data.get('api_key') and not self.observer.is_alive():
            logger.info("🔄 Configuración guardada. Reiniciando monitoreo...")
            self._start_folder_monitoring()
