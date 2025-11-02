# main_photo_manager.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles 
import uvicorn
import os
import hashlib
import shutil
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid
import json
from pathlib import Path
import asyncio
import logging
import traceback
import sys
import aiofiles
import aiohttp
from PIL import Image, ExifTags
import math
from concurrent.futures import ThreadPoolExecutor
import redis
from datetime import timedelta
import mimetypes
import stat
import base64
from pydantic import BaseModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('photo_manager.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Scalable Photo Manager API", version="2.0.0")

# Configuración
FACE_RECOGNITION_API = "https://athematic-carson-subtetanic.ngrok-free.dev"
MAX_FILE_SIZE = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
THUMBNAIL_SIZE = (300, 300)
REDIS_URL = "redis://localhost:6379"

# --- MODELOS PYDANTIC PARA EL CATÁLOGO ---
class PhotoMetadata(BaseModel):
    filename: str
    filepath: str
    file_size: int
    image_width: int
    image_height: int
    format: str
    client_id: str
    upload_date: str

class PhotoRegistration(BaseModel):
    metadata: PhotoMetadata
    thumbnail_base64: str

# CORS mejorado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Middleware para ngrok y optimización
@app.middleware("http")
async def optimization_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept, ngrok-skip-browser-warning"
    if request.url.path.startswith("/api/image/") or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'"{hash(str(request.url))}"'
    return response

# Directorios con estructura escalable
BASE_DIR = Path("photo_storage")
UPLOAD_DIR = BASE_DIR / "uploads"
THUMBNAILS_DIR = BASE_DIR / "thumbnails"
DB_FILE = BASE_DIR / "photo_database.json"
BACKUP_DIR = BASE_DIR / "backups"

for directory in [BASE_DIR, UPLOAD_DIR, THUMBNAILS_DIR, BACKUP_DIR]:
    directory.mkdir(exist_ok=True)

# Pool de hilos para procesamiento en paralelo
executor = ThreadPoolExecutor(max_workers=4)

# Redis para caché (opcional)
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    logger.info("Redis conectado")
    USE_REDIS = True
except:
    logger.warning("Redis no disponible, usando caché local")
    USE_REDIS = False
    redis_client = None

class ScalablePhotoDatabase:
    def __init__(self):
        self.db_file = DB_FILE
        self.data = self.load_database()
    
    def load_database(self) -> Dict:
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data.setdefault('photos', {})
                data.setdefault('collections', {})
                data.setdefault('tags', {})
                data.setdefault('metadata', {'version': '2.0', 'created': datetime.now().isoformat()})
                logger.info(f"BD cargada: {len(data['photos'])} fotos")
                return data
            except Exception as e:
                logger.error(f"Error cargando BD: {e}")
        return {
            'photos': {},
            'collections': {},
            'tags': {},
            'metadata': {'version': '2.0', 'created': datetime.now().isoformat()}
        }
    
    def save_database(self, backup=True):
        try:
            if backup and self.db_file.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = BACKUP_DIR / f"photo_db_backup_{timestamp}.json"
                shutil.copy2(self.db_file, backup_path)
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            if USE_REDIS and redis_client:
                redis_client.setex("photo_db", 3600, json.dumps(self.data))
        except Exception as e:
            logger.error(f"Error guardando BD: {e}")
            raise

# Instancia global de la base de datos
database = ScalablePhotoDatabase()

# --- ENDPOINTS PRINCIPALES ---
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content="<h1>Photo Manager API v2.0</h1><p>Catálogo activo. Ver endpoints en <a href='/docs'>/docs</a></p>")

@app.get("/api-status")
async def api_status():
    return {
        "message": "Scalable Photo Manager API v2.0",
        "status": "running",
        "database": database.get_stats(),
        "redis_available": USE_REDIS
    }

# main_photo_manager.py

# ... (dentro de tu archivo, reemplaza estas funciones)

@app.post("/api/register-photo")
async def register_photo(photo_data: PhotoRegistration):
    """
    Registra una foto en la base de datos central sin almacenar el archivo completo.
    """
    logger.info(f"Petición recibida para registrar foto: {photo_data.metadata.filename}")
    try:
        meta = photo_data.metadata
        thumb_b64 = photo_data.thumbnail_base64
        photo_id = str(uuid.uuid4())
        
        thumbnail_path = None
        try:
            header, encoded = thumb_b64.split(",", 1)
            thumbnail_data = base64.b64decode(encoded)
            thumbnail_filename = f"{photo_id}_thumb.jpg"
            thumbnail_path = THUMBNAILS_DIR / thumbnail_filename
            
            async with aiofiles.open(thumbnail_path, 'wb') as f:
                await f.write(thumbnail_data)
            logger.info(f"✅ Thumbnail guardado exitosamente en: {thumbnail_path}")
        except Exception as e:
            logger.error(f"❌ Error guardando thumbnail para {meta.filename}: {e}")
            thumbnail_path = None

        photo_data_to_save = {
            'id': photo_id,
            'original_name': meta.filename,
            'client_filepath': meta.filepath,
            'client_id': meta.client_id,
            'file_size': meta.file_size,
            'upload_date': meta.upload_date,
            'image_width': meta.image_width,
            'image_height': meta.image_height,
            'format': meta.format,
            'thumbnail_path': str(thumbnail_path.relative_to(BASE_DIR)) if thumbnail_path else None,
            'registered_at': datetime.now().isoformat()
        }
        
        database.data['photos'][photo_id] = photo_data_to_save
        database.save_database()
        
        logger.info(f"✅ Foto registrada en catálogo: {meta.filename} desde cliente {meta.client_id}")
        
        return {
            "success": True,
            "message": "Foto registrada correctamente en el catálogo",
            "catalog_id": photo_id
        }
    except Exception as e:
        logger.error(f"❌ Error registrando foto: {e}")
        raise HTTPException(status_code=500, detail=f"Error registrando foto: {str(e)}")


@app.get("/api/catalog/photos")
async def get_catalog_photos(limit: int = 100, offset: int = 0):
    """Obtiene todas las fotos registradas en el catálogo central."""
    try:
        all_photos = list(database.data['photos'].values())
        all_photos.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        paginated_photos = all_photos[offset:offset+limit]
        
        for photo in paginated_photos:
            if photo.get('thumbnail_path'):
                # Usamos la URL del nuevo endpoint dedicado
                photo['thumbnail_url'] = f"/thumbnails/{Path(photo['thumbnail_path']).name}"
            else:
                photo['thumbnail_url'] = None

        return {
            "success": True,
            "photos": paginated_photos,
            "total": len(all_photos)
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo catálogo de fotos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thumbnails/{thumbnail_name}")
async def get_thumbnail(thumbnail_name: str):
    """
    Sirve una miniatura específica desde la carpeta de thumbnails.
    """
    thumbnail_path = THUMBNAILS_DIR / thumbnail_name
    
    if not thumbnail_path.exists():
        logger.warning(f"⚠️ Thumbnail no encontrado: {thumbnail_path}")
        raise HTTPException(status_code=404, detail="Miniatura no encontrada")
    
    logger.info(f"✅ Sirviendo thumbnail: {thumbnail_path}")
    return FileResponse(
        path=thumbnail_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000"}
    )



    
if __name__ == "__main__":
    print("Scalable Photo Manager API v2.0 - Modo Catálogo")
    print("Servidor: http://localhost:8000")
    print("Para ngrok: ngrok http 8000 --host-header=localhost:8000")
    
    uvicorn.run(
        "main_photo_manager:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
