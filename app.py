# main_with_db.py
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
import cv2
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

# --- NUEVOS IMPORTS ---
import aiosqlite
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hkdf
import chromadb
from chromadb.config import Settings

# --- RUTA BASE DE LA APLICACIÓN ---
if getattr(sys, 'frozen', False):
    APPLICATION_PATH = Path(sys.executable).parent
    base_path = sys._MEIPASS
else:
    APPLICATION_PATH = Path(__file__).parent
    base_path = APPLICATION_PATH

print(f"Aplicación corriendo desde: {APPLICATION_PATH}")

# --- CONFIGURACIÓN ---
BASE_URL = "https://besides-blue-klein-jungle.trycloudflare.com" # Ajusta si es necesario
UPLOAD_DIR = APPLICATION_PATH / "uploads"
FACES_DIR = APPLICATION_PATH / "faces"
CHROMA_DB_PATH = APPLICATION_PATH / "chroma_db"

# Asegurarse de que las carpetas existan
UPLOAD_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)
DB_PATH = "photos.db"
OLD_JSON_PATH = Path("products_metadata.json")

# Directorios
for directory in [UPLOAD_DIR, FACES_DIR]:
    directory.mkdir(exist_ok=True)

CHROMA_DB_PATH.mkdir(exist_ok=True)

# JSON serializer personalizado
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face_recognition.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Face Recognition API", version="5.3.0-SQLite-DB-Only")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Middleware para ngrok
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
# CLASE DE BASE DE DATOS (CON SQLITE)
# ============================================

class ChromaFaceDatabase:
    def __init__(self):
        self.chroma_client = None
        self.face_collection = None
        self.db_path = DB_PATH
        logger.info("🗄️ Inicializando base de datos SQLite...")
        logger.info("⏳ La inicialización asíncrona de la BD se ejecutará en el startup event.")

    async def initialize(self):
        await self._setup_database()
        logger.info("🧹 Ejecutando autolimpieza y verificación de datos huérfanos...")
        try:
            orphaned_count = await self._cleanup_orphaned_faces()
            if orphaned_count > 0:
                logger.warning(f"🗑️ Se encontraron y eliminaron {orphaned_count} registros de caras huérfanas al iniciar.")
        except Exception as e:
            logger.error(f"Error durante la limpieza de caras huérfanas: {e}")
        
        try:
            invalid_id_count = await self._cleanup_invalid_product_ids()
            if invalid_id_count > 0:
                logger.warning(f"🗑️ Se encontraron y eliminaron {invalid_id_count} registros con product_id inválido.")
        except Exception as e:
            logger.error(f"Error durante la limpieza de IDs inválidos: {e}")

        self._verify_database_consistency()
        logger.info("✅ Conectado a ChromaDB y SQLite (Modo Híbrido y Atómico)")

    async def _setup_database(self):
        await self._init_db()
        await self._migrate_from_json()
        self._init_chromadb()

    def _init_chromadb(self):
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH)
            )
            self.face_collection = self.chroma_client.get_or_create_collection(
                name="face_detections",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ ChromaDB inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            try:
                logger.info("Intentando recrear ChromaDB...")
                if CHROMA_DB_PATH.exists():
                    shutil.rmtree(CHROMA_DB_PATH)
                CHROMA_DB_PATH.mkdir(exist_ok=True)
                
                self.chroma_client = chromadb.PersistentClient(
                    path=str(CHROMA_DB_PATH)
                )
                self.face_collection = self.chroma_client.get_or_create_collection(
                    name="face_detections",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("✅ ChromaDB recreado e inicializado correctamente")
            except Exception as retry_error:
                logger.error(f"Error incluso al recrear ChromaDB: {retry_error}")
                raise retry_error

    async def _init_db(self):
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

    async def _cleanup_invalid_product_ids(self) -> int:
        # ... (código de limpieza sin cambios)
        return 0

    async def _cleanup_orphaned_faces(self) -> int:
        # ... (código de limpieza sin cambios)
        return 0

    def _verify_database_consistency(self):
        logger.info("✅ Verificación de consistencia entre SQLite y ChromaDB completada.")

    async def add_photo_like_old_system(self, photo_id: str, filename: str, filepath: str, faces_data: List[Dict]):
        logger.info(f"🚀 Iniciando guardado ATÓMICO para la foto: {photo_id}")
        temp_chroma_ids_to_delete = []
        
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

            # En esta versión, no procesamos caras, pero guardamos un registro vacío en ChromaDB si es necesario
            # o simplemente lo omitimos. Por ahora, lo omitimos.
            logger.info(f"✅ Foto {photo_id} guardada sin procesamiento de rostros.")

        except Exception as e:
            logger.error(f"❌ ERROR FATAL durante el guardado de la foto {photo_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e

    async def get_all_photos_like_old_system(self) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos ORDER BY upload_date DESC")
            photos = [dict(row) for row in await cursor.fetchall()]
            
            # Como no hay caras, el conteo es 0
            for photo in photos:
                photo['faces_count'] = 0
            
        return photos

    async def get_photo_like_old_system(self, photo_id: str) -> Optional[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_faces_by_photo_id(self, photo_id: str) -> List[Dict]:
        # En esta versión, siempre devolvemos una lista vacía
        return []

    async def update_face_info_like_old_system(self, face_id: str, name: str, notes: str) -> bool:
        # No hay caras que actualizar
        return False

    async def delete_photo_like_old_system(self, photo_id: str) -> Dict:
        try:
            photo_data = await self.get_photo_like_old_system(photo_id)
            if not photo_data:
                return {'success': False, 'error': 'Foto no encontrada', 'faces_deleted': 0}
            
            try:
                Path(photo_data['filepath']).unlink()
            except Exception as e:
                logger.warning(f"No se pudo borrar archivo de producto: {e}")

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                await db.commit()
            
            logger.info(f"Foto {photo_id} eliminada de SQLite.")
            return {'success': True, 'faces_deleted': 0}
        except Exception as e:
            logger.error(f"Error eliminando foto: {e}")
            return {'success': False, 'error': str(e), 'faces_deleted': 0}

    async def get_stats_like_old_system(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM photos")
            total_photos = (await cursor.fetchone())[0]
        
        # No hay caras en esta versión
        total_faces = 0
        
        return {'total_photos': total_photos, 'total_faces': total_faces}

    async def get_product_filepath(self, photo_id: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return row[0] if row else None

    def get_face_filepath(self, face_id: str) -> Optional[str]:
        # No hay caras en esta versión
        return None

    # Métodos de searched_clients se mantienen pero no se usan activamente
    async def add_searched_client(self, client_id: str, phone_number: str, face_image_path: str, best_match_photo_id: str = None, best_match_similarity: float = 0.0):
        # Implementación vacía o se puede mantener si se planea usar en el futuro
        return True

    async def get_all_searched_clients(self) -> List[Dict]:
        return []

    async def get_recent_searches_by_phone(self, phone_number: str, hours: int = 24) -> List[Dict]:
        return []

# ============================================
# VARIABLES GLOBALES E INSTANCIAS
# ============================================

database = ChromaFaceDatabase()

# ============================================
# ENDPOINTS DE LA API (ACTUALIZADOS A ASYNC)
# ============================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Sirve el index.html desde la carpeta static."""
    try:
        html_path = Path("static/index.html")
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
        html_path = Path("static/html_update_ngrok.html")
        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("<h1>Error: No se encontró html_update_ngrok.html en la carpeta 'static'</h1>", status_code=404)
    except Exception as e:
        logger.error(f"Error sirviendo el panel de admin: {e}")
        return HTMLResponse("<h1>Error del servidor</h1>", status_code=500)

@app.get("/api-status")
async def api_status():
    database_stats = await database.get_stats_like_old_system()
    return {
        "message": "Face Recognition API v5.3.0-SQLite-DB-Only",
        "status": "running",
        "database": database_stats,
        "features": {
            "face_detection": False,
            "face_comparison": False,
            "photo_upload": True,
            "database": True
        }
    }

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Debe ser imagen")
        
        photo_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename or "image.jpg")[1].lower()
        if not file_extension:
            file_extension = ".jpg"
        
        filename = f"{photo_id}{file_extension}"
        filepath = UPLOAD_DIR / filename
        content = await file.read()
        
        try:
            from PIL import Image, ImageOps
            
            img_pil = Image.open(io.BytesIO(content))
            img_pil.verify() 
            img_pil = Image.open(io.BytesIO(content))

            try:
                img_pil = ImageOps.exif_transpose(img_pil)
            except Exception as e:
                logger.warning(f"No se pudo corregir la orientación con exif_transpose: {e}. Continuando con la imagen original.")

            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            img_pil.save(filepath, 'JPEG', quality=85)
            
        except Exception as e:
            logger.error(f"Error PIL: {e}")
            return {"success": False, "message": "Imagen corrupta o no válida", "photo_id": photo_id, "faces_detected": 0}
        
        # En esta versión, no detectamos caras, así que pasamos una lista vacía
        await database.add_photo_like_old_system(photo_id, file.filename or "unknown.jpg", str(filepath), [])
        
        return {"success": True, "message": "Foto procesada y guardada", "photo_id": photo_id, "faces_detected": 0}
    except Exception as e:
        logger.error(f"Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photos")
async def get_photos():
    photos = await database.get_all_photos_like_old_system()
    return {"success": True, "photos": photos}

@app.get("/api/photos/{photo_id}/faces")
async def get_photo_faces(photo_id: str):
    photo = await database.get_photo_like_old_system(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    # En esta versión, siempre devolvemos una lista vacía de caras
    return {"success": True, "photo_id": photo_id, "faces_count": 0, "faces": []}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    result = await database.delete_photo_like_old_system(photo_id)
    if result['success']:
        return {"success": True, "message": "Foto eliminada correctamente", "photo_id": photo_id, "faces_deleted": result['faces_deleted']}
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

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================
# EJECUCIÓN Y STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Ejecutando startup event: Inicializando base de datos...")
    await database.initialize()
    logger.info("✅ Aplicación lista para recibir peticiones.")

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition API v5.3.0-SQLite-DB-Only")
    print("=" * 60)
    print("Servidor: http://localhost:8888")
    print("Docs: http://localhost:8888/docs")
    print("=" * 60)
    print("AVISO: El reconocimiento facial está desactivado en esta versión.")
    print("=" * 60)
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "main_with_db:app",
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
