# app.py
import io
import json
import logging
import os
import shutil
import sys
import time
import traceback
import uuid
import asyncio
import cv2  # Lo mantenemos para obtener las dimensiones de la imagen
import hashlib
import httpx
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- IMPORTS DE BASE DE DATOS ---
import aiosqlite

# --- RUTA BASE DE LA APLICACIÓN ---
if getattr(sys, 'frozen', False):
    APPLICATION_PATH = Path(sys.executable).parent
    base_path = sys._MEIPASS
else:
    APPLICATION_PATH = Path(__file__).parent
    base_path = APPLICATION_PATH

print(f"Aplicación corriendo desde: {APPLICATION_PATH}")

# Forzar la codificación a UTF-8 para la salida estándar (consola)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('photo_manager.log', encoding='utf-8') # Cambiado el nombre del log
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
BASE_URL = "https://besides-blue-klein-jungle.trycloudflare.com"
UPLOAD_DIR = APPLICATION_PATH / "uploads"
STATIC_DIR = APPLICATION_PATH / "static"
DB_PATH = APPLICATION_PATH / "photos.db" # Ruta más explícita para la BD
OLD_JSON_PATH = APPLICATION_PATH / "products_metadata.json"

# Asegurarse de que las carpetas existan
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# JSON serializer personalizado (sin cambios)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def safe_convert_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: safe_convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

app = FastAPI(title="Photo Manager API", version="6.0.0-SQLite-Only")

# CORS (sin cambios)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Middleware para ngrok (sin cambios)
@app.middleware("http")
async def ngrok_middleware(request: Request, call_next):
    response = await call_next(request)
    if "ngrok" in str(request.headers.get("host", "")):
        response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============================================
# CLASE DE BASE DE DATOS (SOLO SQLITE)
# ============================================

class PhotoDatabase:
    """Maneja todas las operaciones de la base de datos SQLite para fotos."""
    def __init__(self):
        self.db_path = DB_PATH
        logger.info("🗄️ Inicializando base de datos SQLite...")
        logger.info("⏳ La inicialización asíncrona de la BD se ejecutará en el startup event.")

    async def initialize(self):
        await self._setup_database()
        await self._migrate_from_json()
        logger.info("✅ Base de datos SQLite inicializada y lista.")

    async def _setup_database(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    upload_date TEXT NOT NULL,
                    image_width INTEGER,
                    image_height INTEGER,
                    file_size INTEGER
                )
            """)
            # Mantenemos la tabla de búsquedas, aunque no se use, por si se activa en el futuro
            await db.execute("""
                CREATE TABLE IF NOT EXISTS searched_clients (
                    id TEXT PRIMARY KEY,
                    phone_number TEXT,
                    search_date TEXT NOT NULL,
                    face_image_path TEXT,
                    best_match_photo_id TEXT,
                    best_match_similarity REAL,
                    FOREIGN KEY (best_match_photo_id) REFERENCES photos (id)
                )
            """)
            await db.commit()

    async def _migrate_from_json(self):
        if OLD_JSON_PATH.exists():
            logger.info("🔄 Iniciando migración desde products_metadata.json a SQLite...")
            try:
                with open(OLD_JSON_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "photos" in data and data["photos"]:
                    async with aiosqlite.connect(self.db_path) as db:
                        for photo_id, photo_data in data["photos"].items():
                            await db.execute("""
                                INSERT OR IGNORE INTO photos (id, filename, filepath, upload_date, image_width, image_height, file_size)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                photo_data.get('id'),
                                photo_data.get('filename'),
                                photo_data.get('filepath'),
                                photo_data.get('upload_date'),
                                photo_data.get('image_width'),
                                photo_data.get('image_height'),
                                photo_data.get('file_size')
                            ))
                        await db.commit()
                    logger.info(f"✅ Migración completada. Se migraron {len(data['photos'])} fotos.")
                    os.rename(OLD_JSON_PATH, f"{OLD_JSON_PATH}.migrated")
                else:
                    logger.info("📄 products_metadata.json está vacío o no tiene fotos. No se requiere migración.")
            except Exception as e:
                logger.error(f"❌ Error durante la migración desde JSON: {e}")
        else:
            logger.info("📄 products_metadata.json no encontrado. Iniciando con base de datos SQLite nueva.")

    async def add_photo(self, photo_id: str, filename: str, filepath: str):
        """Añade una nueva foto a la base de datos."""
        logger.info(f"🚀 Guardando metadatos para la foto: {photo_id}")
        try:
            img = cv2.imread(filepath)
            image_height, image_width = img.shape[:2] if img is not None else (0, 0)
            file_size = Path(filepath).stat().st_size

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO photos (id, filename, filepath, upload_date, image_width, image_height, file_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (photo_id, filename, filepath, datetime.now().isoformat(), image_width, image_height, file_size))
                await db.commit()
            logger.info(f"✅ Metadatos para la foto {photo_id} guardados en SQLite.")
            return True
        except Exception as e:
            logger.error(f"❌ ERROR FATAL durante el guardado de la foto {photo_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def get_all_photos(self) -> List[Dict]:
        """Obtiene todas las fotos de la base de datos."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos ORDER BY upload_date DESC")
            photos = [dict(row) for row in await cursor.fetchall()]
        return photos

    async def get_photo(self, photo_id: str) -> Optional[Dict]:
        """Obtiene una foto específica por su ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_photo(self, photo_id: str) -> Dict:
        """Elimina una foto de la base de datos y del sistema de archivos."""
        try:
            photo_data = await self.get_photo(photo_id)
            if not photo_data:
                return {'success': False, 'error': 'Foto no encontrada'}
            
            try:
                Path(photo_data['filepath']).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"No se pudo borrar archivo de producto: {e}")

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                await db.commit()
            
            logger.info(f"Foto {photo_id} eliminada de SQLite.")
            return {'success': True}
        except Exception as e:
            logger.error(f"Error eliminando foto: {e}")
            return {'success': False, 'error': str(e)}

    async def get_stats(self) -> Dict:
        """Obtiene estadísticas simples de la base de datos."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM photos")
            total_photos = (await cursor.fetchone())[0]
        
        return {'total_photos': total_photos}

    async def get_product_filepath(self, photo_id: str) -> Optional[str]:
        """Obtiene la ruta del archivo de una foto."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return row[0] if row else None

# ============================================
# VARIABLES GLOBALES E INSTANCIAS
# ============================================

database = PhotoDatabase()

# ============================================
# ENDPOINTS DE LA API
# ============================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Sirve el index.html desde la carpeta static."""
    try:
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("<h1>Error: No se encontró index.html en la carpeta 'static'</h1>", status_code=404)
    except Exception as e:
        logger.error(f"Error sirviendo index.html: {e}")
        return HTMLResponse("<h1>Error del servidor</h1>", status_code=500)

@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    """Sirve el html_update_ngrok.html desde la carpeta static."""
    try:
        html_path = STATIC_DIR / "html_update_ngrok.html"
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("<h1>Error: No se encontró html_update_ngrok.html en la carpeta 'static'</h1>", status_code=404)
    except Exception as e:
        logger.error(f"Error sirviendo el panel de admin: {e}")
        return HTMLResponse("<h1>Error del servidor</h1>", status_code=500)

@app.get("/api-status")
async def api_status():
    database_stats = await database.get_stats()
    return {
        "message": "Photo Manager API v6.0.0-SQLite-Only",
        "status": "running",
        "database": database_stats,
        "features": {
            "face_detection": False, # <-- ¡IMPORTANTE!
            "face_comparison": False,
            "photo_upload": True,
            "database": True
        }
    }

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    """Sube una foto, la guarda y registra sus metadatos en la BD."""
    try:
        # Validar tipo de archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
        
        # Generar un ID único y nombre de archivo
        photo_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        safe_filename = f"{photo_id}{file_extension}"
        filepath = UPLOAD_DIR / safe_filename
        
        # Guardar el archivo en el disco
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Guardar metadatos en la base de datos (SIN DETECCIÓN DE ROSTROS)
        success = await database.add_photo(photo_id, file.filename or "unknown.jpg", str(filepath))
        
        if not success:
            # Si falla la BD, borramos el archivo para no dejar huérfanos
            filepath.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Error al guardar en la base de datos.")
        
        return {"success": True, "message": "Foto subida y guardada", "photo_id": photo_id}
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photos")
async def get_photos():
    photos = await database.get_all_photos()
    return {"success": True, "photos": photos}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    result = await database.delete_photo(photo_id)
    if result['success']:
        return {"success": True, "message": "Foto eliminada correctamente", "photo_id": photo_id}
    else:
        raise HTTPException(status_code=404, detail=result.get('error', 'Foto no encontrada'))

@app.api_route("/api/image/photo/{photo_id}", methods=["GET", "HEAD"])
async def get_photo_image(photo_id: str, request: Request):
    logger.info(f"Solicitud de imagen para photo_id: {photo_id}")
    filepath_str = await database.get_product_filepath(photo_id)
    if not filepath_str:
        logger.warning(f"404: Foto no encontrada en la base de datos para photo_id: {photo_id}")
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    filepath = Path(filepath_str)
    if not filepath.exists():
        logger.warning(f"404: Archivo de foto no encontrado en el sistema de archivos para filepath: {filepath}")
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    if request.method == "HEAD":
        return Response(headers={"Content-Type": "image/jpeg", "Content-Length": str(filepath.stat().st_size)})
    
    return FileResponse(path=filepath, media_type="image/jpeg")

# Montar archivos estáticos
if STATIC_DIR.exists():
    print(f"Montando archivos estáticos desde: {STATIC_DIR}")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    print(f"ADVERTENCIA: La carpeta 'static' no fue encontrada en {STATIC_DIR}. El frontend no funcionará.")

# ============================================
# EJECUCIÓN Y STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Ejecutando startup event: Inicializando base de datos...")
    await database.initialize()
    logger.info("✅ Aplicación lista para recibir peticiones.")

# --- EL ARCHIVO TERMINA AQUÍ. SIN BLOQUE if __name__ == "__main__": ---
