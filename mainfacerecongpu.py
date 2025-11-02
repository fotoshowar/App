# main.py - Versión Final Híbrica con Guardado Atómico y Depuración (Corregida)

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
import face_recognition
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

# --- CONFIGURACIÓN ---
BASE_URL = "https://athematic-carson-subtetanic.ngrok-free.dev"
UPLOAD_DIR = Path("uploads")
FACES_DIR = Path("faces")
CHROMA_DB_PATH = Path("chroma_db")
DB_PATH = "photos.db"
OLD_JSON_PATH = Path("products_metadata.json")

# Directorios
for directory in [UPLOAD_DIR, FACES_DIR]:
    directory.mkdir(exist_ok=True)

# CORRECCIÓN: Asegurarse de que el directorio de ChromaDB exista
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

app = FastAPI(title="Advanced Face Recognition API", version="5.3.0-SQLite")

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
# FUNCIÓN DE DESENCRIPTACIÓN DE WHATSAPP
# ============================================

def decrypt_whatsapp_media(encrypted_data: bytes, media_key_b64: str, 
                           file_enc_sha256_b64: str, media_type: str = 'image') -> bytes:
    """Desencripta medios de WhatsApp usando las claves proporcionadas."""
    try:
        logger.info("🔐 Iniciando desencriptación de medio de WhatsApp...")
        
        media_key = base64.b64decode(media_key_b64)
        file_enc_sha256 = base64.b64decode(file_enc_sha256_b64)
        
        calculated_sha256 = hashlib.sha256(encrypted_data).digest()
        if calculated_sha256 != file_enc_sha256:
            raise ValueError("❌ Hash SHA256 no coincide. Archivo corrupto.")
        logger.info("✅ Hash SHA256 verificado correctamente")
        
        media_type_info = {
            'image': b'WhatsApp Image Keys',
            'video': b'WhatsApp Video Keys',
            'audio': b'WhatsApp Audio Keys',
            'document': b'WhatsApp Document Keys'
        }
        info = media_type_info.get(media_type, b'WhatsApp Image Keys')
        
        derived = hkdf.hkdf_expand(
            hkdf.hkdf_extract(b'', media_key, hashlib.sha256),
            info,
            112,
            hashlib.sha256
        )
        
        iv = derived[:16]
        cipher_key = derived[16:48]
        mac_key = derived[48:80]
        
        logger.info("✅ Claves derivadas con HKDF")
        
        encrypted_body = encrypted_data[:-10]
        mac_from_file = encrypted_data[-10:]
        
        calculated_mac_v1 = hashlib.sha256(iv + encrypted_body + mac_key).digest()[:10]
        calculated_mac_v2 = hashlib.sha256(encrypted_body + mac_key).digest()[:10]
        calculated_mac_v3 = hashlib.sha256(mac_key + iv + encrypted_body).digest()[:10]
        
        mac_valid = False
        if calculated_mac_v1 == mac_from_file:
            logger.info("✅ MAC verificado (método v1)")
            mac_valid = True
        elif calculated_mac_v2 == mac_from_file:
            logger.info("✅ MAC verificado (método v2)")
            mac_valid = True
        elif calculated_mac_v3 == mac_from_file:
            logger.info("✅ MAC verificado (método v3)")
            mac_valid = True
        
        if not mac_valid:
            logger.warning("⚠️ MAC no coincide con ningún método conocido, intentando desencriptar de todos modos...")
        
        cipher = Cipher(
            algorithms.AES(cipher_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_body) + decryptor.finalize()
        
        padding_length = decrypted_data[-1]
        if isinstance(padding_length, str):
            padding_length = ord(padding_length)
        decrypted_data = decrypted_data[:-padding_length]
        
        logger.info(f"✅ Desencriptación exitosa ({len(decrypted_data)} bytes)")
        
        return decrypted_data
        
    except Exception as e:
        logger.error(f"❌ Error en desencriptación: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# ============================================
# CLASE DE PROCESAMIENTO FACIAL (SIMPLIFICADA Y RÁPIDA)
# ============================================

class AdvancedFaceProcessorIntegration:
    def __init__(self):
        logger.info("Inicializando Face Processor en modo rápido...")
        self.models_status = {
            'face_recognition': self._check_face_recognition(),
            'opencv': self._check_opencv()
        }
        logger.info(f"Estado de los modelos: {self.models_status}")

    def _check_face_recognition(self) -> bool:
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_img)
            logger.info("✅ face_recognition disponible y listo.")
            return True
        except Exception as e:
            logger.error(f"❌ face_recognition no disponible: {e}")
            return False

    def _check_opencv(self) -> bool:
        try:
            self.haar_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("✅ OpenCV disponible como fallback.")
            return True
        except Exception as e:
            logger.error(f"❌ OpenCV no disponible: {e}")
            return False

    def detect_and_encode_faces(self, image_path: str, save_faces: bool = True) -> List[Dict]:
        """
        Detecta y codifica caras usando el método más rápido y eficiente:
        face_recognition con embeddings de 128 dimensiones.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"No se pudo cargar imagen: {image_path}")
                return []
            
            logger.info("🏎️  Modo Rápido: Usando embeddings de 128 dimensiones para máxima velocidad.")
            
            # Usar el modelo 'cnn' para una detección más precisa (aún rápido en CPU)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img, model="cnn")
            
            if not face_locations:
                logger.info("No se detectaron caras en la imagen.")
                return []
            
            # Generar los embeddings para las caras detectadas
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            faces_data = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                face_img = img[top:bottom, left:right]
                
                if face_img.size > 0:
                    face_id = str(uuid.uuid4())
                    
                    face_filename = None
                    if save_faces:
                        face_filename = f"{face_id}.jpg"
                        face_path = FACES_DIR / face_filename
                        cv2.imwrite(str(face_path), face_img)
                    
                    face_data = {
                        'face_id': face_id,
                        'face_filename': face_filename,
                        'bbox': [int(left), int(top), int(right-left), int(bottom-top)],
                        'confidence': 0.90, # Confianza fija para este método
                        'embeddings': {'face_recognition': encoding.tolist()},
                        'method': 'face_recognition_fast',
                        'models_used': ['face_recognition'],
                        'processing_quality': 'fast_128d',
                        'face_image': face_img
                    }
                    faces_data.append(face_data)
            
            logger.info(f"✅ Procesamiento rápido completado. {len(faces_data)} caras detectadas y codificadas.")
            return faces_data

        except Exception as e:
            logger.error(f"Error en el procesamiento rápido: {e}")
            # Si falla face_recognition, usar el fallback de OpenCV
            return self._detect_with_opencv_fallback(img, save_faces)

    def _detect_with_opencv_fallback(self, img: np.ndarray, save_faces: bool = True) -> List[Dict]:
        """Fallback de OpenCV que también genera embeddings de 128 dims si es posible."""
        if not self.models_status.get('opencv', False):
            logger.error("Face_recognition y OpenCV fallaron. No se pueden detectar caras.")
            return []
            
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            faces_data = []
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                face_id = str(uuid.uuid4())
                face_filename = None
                if save_faces:
                    face_filename = f"{face_id}.jpg"
                    face_path = FACES_DIR / face_filename
                    cv2.imwrite(str(face_path), face_img)
                
                # Intentar generar un embedding de 128 dims con face_recognition
                embedding = None
                if self.models_status.get('face_recognition', False):
                    try:
                        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_face_img)
                        if face_encodings:
                            embedding = face_encodings[0].tolist()
                    except Exception as e:
                        logger.error(f"Error generando embedding fallback: {e}")

                if embedding:
                    face_data = {
                        'face_id': face_id,
                        'face_filename': face_filename,
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.75, # Confianza menor para el fallback
                        'embeddings': {'face_recognition': embedding},
                        'method': 'opencv_fallback',
                        'models_used': ['face_recognition'],
                        'processing_quality': 'fallback_128d',
                        'face_image': face_img
                    }
                    faces_data.append(face_data)
            
            logger.info(f"✅ Fallback de OpenCV completado. {len(faces_data)} caras procesadas.")
            return faces_data

        except Exception as e:
            logger.error(f"Error en el fallback de OpenCV: {e}")
            return []

    def compare_embeddings(self, embeddings1: Dict, embeddings2: Dict) -> float:
        """Compara dos embeddings de 128 dimensiones."""
        try:
            if 'face_recognition' in embeddings1 and 'face_recognition' in embeddings2:
                emb1 = np.array(embeddings1['face_recognition'])
                emb2 = np.array(embeddings2['face_recognition'])
                distance = np.linalg.norm(emb1 - emb2)
                return max(0, 1 - distance / 1.2) # Distancia euclidiana normalizada
            return 0.0
        except Exception as e:
            logger.error(f"Error en comparación: {e}")
            return 0.0
    
    def get_system_status(self) -> Dict:
        return {
            'advanced_processor_loaded': False, # Ya no se usa
            'models_status': self.models_status,
            'processing_mode': 'fast_128d_cpu' # Modo de procesamiento actual
        }




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
        try:
            logger.info("🔍 Iniciando limpieza de product_id inválidos...")
            all_results = self.face_collection.get(include=['metadatas'])
            invalid_ids = []
            
            if not all_results or not all_results['ids']:
                logger.info("🔍 No hay caras en la base de datos para limpiar.")
                return 0
            
            for i, face_id in enumerate(all_results['ids']):
                metadata = all_results['metadatas'][i]
                product_id = metadata.get('product_id')

                is_invalid = (
                    not product_id or 
                    str(product_id).strip().lower() in ["undefined", "null", "none", ""] or
                    product_id is None
                )

                if is_invalid:
                    logger.warning(f"👻 Cara con ID inválido encontrada y será eliminada: {face_id} (product_id: '{product_id}')")
                    invalid_ids.append(face_id)
            
            if invalid_ids:
                self.face_collection.delete(ids=invalid_ids)
                logger.info(f"🗑️ {len(invalid_ids)} caras con ID inválido eliminadas de ChromaDB.")
            else:
                logger.info("✅ No se encontraron caras con product_id inválido.")
            
            return len(invalid_ids)
        except Exception as e:
            logger.error(f"Error durante la limpieza de IDs inválidos: {e}")
            return 0

    async def _cleanup_orphaned_faces(self) -> int:
        try:
            all_results = self.face_collection.get(include=['metadatas'])
            orphaned_ids = []
            
            if not all_results or not all_results['ids']:
                return 0
            
            all_photo_ids = set()
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT id FROM photos")
                rows = await cursor.fetchall()
                all_photo_ids = {row[0] for row in rows}

            for i, face_id in enumerate(all_results['ids']):
                metadata = all_results['metadatas'][i]
                product_id = metadata.get('product_id')
                
                if not product_id or product_id not in all_photo_ids:
                    logger.warning(f"👻 Cara huérfana encontrada y será eliminada: {face_id} (product_id: {product_id})")
                    orphaned_ids.append(face_id)
            
            if orphaned_ids:
                self.face_collection.delete(ids=orphaned_ids)
                logger.info(f"🗑️ {len(orphaned_ids)} caras huérfanas eliminadas de ChromaDB.")
            
            return len(orphaned_ids)
        except Exception as e:
            logger.error(f"Error durante la autolimpieza de caras huérfanas: {e}")
            return 0

    def _verify_database_consistency(self):
        logger.info("✅ Verificación de consistencia entre SQLite y ChromaDB completada por la limpieza de huérfanos.")

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

            if not faces_data:
                logger.warning(f"⚠️ No hay caras para guardar para la foto {photo_id}.")
                return

            ids = []
            embeddings = []
            metadatas = []
            
            valid_faces_count = 0
            for face_data in faces_data:
                embedding = None
                if 'embeddings' in face_data and face_data['embeddings'] and 'face_recognition' in face_data['embeddings']:
                    embedding = face_data['embeddings']['face_recognition']
                
                if embedding is None:
                    logger.warning(f"⚠️ Omitiendo cara {face_data.get('face_id', 'N/A')} porque no tiene un embedding 'face_recognition' compatible.")
                    continue
                
                ids.append(face_data['face_id'])
                embeddings.append(embedding)
                metadatas.append({
                    "product_id": photo_id,
                    "photo_filename": filename,
                    "bbox": json.dumps(face_data['bbox']),
                    "confidence": face_data['confidence'],
                    "method": face_data.get('method', 'unknown'),
                    "face_filename": face_data.get('face_filename', ''),
                    "customer_name": "",
                    "customer_notes": ""
                })
                valid_faces_count += 1
            
            logger.info(f"📊 Se prepararon {valid_faces_count} caras válidas para ChromaDB.")

            # CORRECCIÓN: Verificación final de consistencia de dimensiones
            if embeddings:
                first_dim = len(embeddings[0])
                inconsistent_dims = [len(emb) for emb in embeddings if len(emb) != first_dim]
                if inconsistent_dims:
                    logger.error(f"❌ ERROR CRÍTICO: Inconsistencia de dimensiones encontrada. Dimensión esperada: {first_dim}, Dimensiones encontradas: {set(inconsistent_dims)}")
                    raise ValueError("Inconsistent dimensions in provided embeddings after final check.")
                
                logger.info(f"💾 Añadiendo {len(embeddings)} embeddings a ChromaDB (Dimensión: {first_dim})...")
                try:
                    self.face_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    temp_chroma_ids_to_delete = ids
                    logger.info(f"✅ {len(embeddings)} embeddings guardados en ChromaDB.")
                except Exception as e:
                    logger.error(f"Error al guardar en ChromaDB: {e}")
                    try:
                        logger.info("Intentando recrear la colección de ChromaDB...")
                        self.chroma_client.delete_collection("face_detections")
                        self.face_collection = self.chroma_client.create_collection(
                            name="face_detections",
                            metadata={"hnsw:space": "cosine"}
                        )
                        self.face_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                        temp_chroma_ids_to_delete = ids
                        logger.info(f"✅ {len(embeddings)} embeddings guardados en ChromaDB después de recrear la colección.")
                    except Exception as retry_error:
                        logger.error(f"Error incluso al recrear la colección: {retry_error}")
                        raise e
            else:
                logger.warning(f"⚠️ No se guardaron embeddings en ChromaDB.")

        except Exception as e:
            logger.error(f"❌ ERROR FATAL durante el guardado de la foto {photo_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.error("🔄 Iniciando ROLLBACK de ChromaDB...")
            if temp_chroma_ids_to_delete:
                try:
                    self.face_collection.delete(ids=temp_chroma_ids_to_delete)
                    logger.info(f"🗑️ Embeddings añadidos a ChromaDB fueron eliminados durante el rollback.")
                except Exception as delete_error:
                    logger.error(f"❌ ERROR FATAL durante la eliminación de embeddings de ChromaDB: {delete_error}")
            
            raise e

    async def search_similar_faces(self, search_embeddings: Dict, threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        logger.info(f"🔍 Iniciando búsqueda HÍBRIDA (iterativa) con threshold={threshold}")
        try:
            search_embedding = None
            if 'embeddings' in search_embeddings and 'face_recognition' in search_embeddings['embeddings']:
                search_embedding = search_embeddings['embeddings']['face_recognition']
                logger.info("✅ Usando embedding 'face_recognition' para la búsqueda (método compatible).")
            else:
                logger.warning("❌ Búsqueda fallida: La cara de búsqueda no tiene un embedding 'face_recognition' compatible con la base de datos.")
                return []
            
            if not search_embedding:
                logger.warning("Búsqueda fallida: No se encontró ningún embedding válido.")
                return []

            all_results = self.face_collection.get(include=['embeddings', 'metadatas'])
            matches = []
            
            if all_results['ids']:
                total_faces = len(all_results['ids'])
                logger.info(f"📊 Iterando sobre {total_faces} caras en la base de datos...")
                
                for i, face_id in enumerate(all_results['ids']):
                    current_embedding = all_results['embeddings'][i]
                    metadata = all_results['metadatas'][i]
                    product_id = metadata.get('product_id')

                    if not product_id or str(product_id).strip().lower() in ["undefined", "null", "none", ""]:
                        logger.error(f"🚨 Ignorando cara {face_id} con product_id inválido durante la búsqueda: '{product_id}'")
                        continue
                    
                    similarity = self._fallback_comparison({'face_recognition': search_embedding}, {'face_recognition': current_embedding})
                    
                    if similarity >= threshold:
                        photo_exists = await self.get_photo_like_old_system(product_id)
                        if photo_exists:
                            match_data = {
                                'face_id': face_id,
                                'photo_id': product_id,
                                'photo_filename': photo_exists.get('filename', 'unknown'),
                                'product_name': f"Photo Upload {product_id[:8]}",
                                'product_price': 0.0,
                                'customer_name': metadata.get('customer_name', ''),
                                'customer_email': '',
                                'similarity': float(similarity),
                                'confidence': float(metadata.get('confidence', 0.0)),
                                'detection_method': metadata.get('method', 'unknown'),
                                'bbox': json.loads(metadata.get('bbox', '[]'))
                            }
                            matches.append(match_data)

            matches.sort(key=lambda x: x['similarity'], reverse=True)
            logger.info(f"✅ Búsqueda híbrida completada. {len(matches)} coincidencias encontradas.")
            return matches[:limit]

        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            return []

    def _fallback_comparison(self, embeddings1: Dict, embeddings2: Dict) -> float:
        try:
            if 'face_recognition' in embeddings1 and 'face_recognition' in embeddings2:
                emb1 = np.array(embeddings1['face_recognition'])
                emb2 = np.array(embeddings2['face_recognition'])
                distance = np.linalg.norm(emb1 - emb2)
                return max(0, 1 - distance / 1.2)
            return 0.5
        except Exception as e:
            logger.error(f"Error en comparación fallback: {e}")
            return 0.0

    async def get_all_photos_like_old_system(self) -> List[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos ORDER BY upload_date DESC")
            photos = [dict(row) for row in await cursor.fetchall()]
            
            for photo in photos:
                try:
                    face_results = self.face_collection.get(where={"product_id": photo['id']})
                    photo['faces_count'] = len(face_results['ids']) if face_results and face_results['ids'] else 0
                except Exception as e:
                    logger.error(f"Error obteniendo conteo de caras para la foto {photo['id']}: {e}")
                    photo['faces_count'] = 0
            
        return photos

    async def get_photo_like_old_system(self, photo_id: str) -> Optional[Dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_faces_by_photo_id(self, photo_id: str) -> List[Dict]:
        all_results = self.face_collection.get(where={"product_id": photo_id}, include=['metadatas'])
        faces_list = []
        if all_results['ids']:
            for i, face_id in enumerate(all_results['ids']):
                metadata = all_results['metadatas'][i]
                faces_list.append({
                    'id': face_id,
                    'product_id': photo_id,
                    'bbox': json.loads(metadata.get('bbox', '[]')),
                    'confidence': metadata.get('confidence', 0.0),
                    'customer_name': metadata.get('customer_name', ''),
                    'customer_notes': metadata.get('customer_notes', ''),
                    'face_filename': metadata.get('face_filename', '')
                })
        return faces_list

    async def update_face_info_like_old_system(self, face_id: str, name: str, notes: str) -> bool:
        try:
            self.face_collection.update(
                ids=[face_id],
                metadatas=[{"customer_name": name, "customer_notes": notes}]
            )
            logger.info(f"✅ Info de cara {face_id} actualizada.")
            return True
        except Exception as e:
            logger.error(f"❌ Error actualizando info de cara: {e}")
            return False

    async def delete_photo_like_old_system(self, photo_id: str) -> Dict:
        try:
            photo_data = await self.get_photo_like_old_system(photo_id)
            if not photo_data:
                return {'success': False, 'error': 'Foto no encontrada', 'faces_deleted': 0}

            face_ids = [face['id'] for face in await self.get_faces_by_photo_id(photo_id)]
            faces_deleted = 0
            if face_ids:
                self.face_collection.delete(ids=face_ids)
                faces_deleted = len(face_ids)
            
            try:
                Path(photo_data['filepath']).unlink()
            except Exception as e:
                logger.warning(f"No se pudo borrar archivo de producto: {e}")

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                await db.commit()
            
            logger.info(f"Foto {photo_id} eliminada de SQLite y ChromaDB.")
            return {'success': True, 'faces_deleted': faces_deleted}
        except Exception as e:
            logger.error(f"Error eliminando foto: {e}")
            return {'success': False, 'error': str(e), 'faces_deleted': 0}

    async def get_stats_like_old_system(self) -> Dict:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM photos")
            total_photos = (await cursor.fetchone())[0]
        
        all_results = self.face_collection.get()
        total_faces = len(all_results['ids']) if all_results and all_results['ids'] else 0
        
        return {'total_photos': total_photos, 'total_faces': total_faces}

    async def get_product_filepath(self, photo_id: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,))
            row = await cursor.fetchone()
            return row[0] if row else None

    def get_face_filepath(self, face_id: str) -> Optional[str]:
        try:
            results = self.face_collection.get(ids=[face_id], include=['metadatas'])
            if results['metadatas'] and results['metadatas'][0]:
                return results['metadatas'][0].get('face_filename')
            return None
        except Exception as e:
            logger.error(f"Error obteniendo filepath de cara: {e}")
            return None

    async def add_searched_client(self, client_id: str, phone_number: str, face_image_path: str, best_match_photo_id: str = None, best_match_similarity: float = 0.0):
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO searched_clients (id, phone_number, search_date, face_image_path, best_match_photo_id, best_match_similarity)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    client_id,
                    phone_number,
                    datetime.now().isoformat(),
                    face_image_path,
                    best_match_photo_id,
                    best_match_similarity
                ))
                await db.commit()
            logger.info(f"✅ Cliente buscado guardado: {phone_number}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando cliente buscado: {e}")
            return False

    async def get_all_searched_clients(self) -> List[Dict]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("""
                    SELECT sc.*, p.filename as best_match_filename 
                    FROM searched_clients sc
                    LEFT JOIN photos p ON sc.best_match_photo_id = p.id
                    ORDER BY sc.search_date DESC
                """)
                clients = [dict(row) for row in await cursor.fetchall()]
            return clients
        except Exception as e:
            logger.error(f"❌ Error obteniendo clientes buscados: {e}")
            return []

    async def get_recent_searches_by_phone(self, phone_number: str, hours: int = 24) -> List[Dict]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                limit_date = (datetime.now() - timedelta(hours=hours)).isoformat()
                
                cursor = await db.execute("""
                    SELECT sc.*, p.filename as best_match_filename 
                    FROM searched_clients sc
                    LEFT JOIN photos p ON sc.best_match_photo_id = p.id
                    WHERE sc.phone_number = ? AND sc.search_date > ?
                    ORDER BY sc.search_date DESC
                """, (phone_number, limit_date))
                
                searches = [dict(row) for row in await cursor.fetchall()]
                return searches
        except Exception as e:
            logger.error(f"Error obteniendo búsquedas recientes: {e}")
            return []

# ============================================
# VARIABLES GLOBALES E INSTANCIAS
# ============================================

processor = AdvancedFaceProcessorIntegration()
database = ChromaFaceDatabase()

whatsapp_client = httpx.AsyncClient()
WHATSAPP_API_TOKEN = "73614cf4168c1ddf324f588da212f0d46a97d819444e1ad9fb4edab2b0eea64b"
WHATSAPP_API_URL = "https://wasenderapi.com/api/send-message"
WHATSAPP_WEBHOOK_SECRET = "542909770b6912825b8ea84a49a6f708"

message_queue = deque()
processing_queue = False

# ============================================
# FUNCIONES AUXILIARES DE WHATSAPP
# ============================================

async def send_whatsapp_message_with_retry(to_number, text, max_retries=3):
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"to": to_number, "text": text}
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt
                logger.info(f"Reintentando enviar mensaje en {delay} segundos")
                await asyncio.sleep(delay)
            
            response = await whatsapp_client.post(WHATSAPP_API_URL, headers=headers, json=payload)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 65))
                logger.warning(f"Límite de velocidad alcanzado. Esperando {retry_after} segundos...")
                await asyncio.sleep(retry_after)
                continue
                
            response.raise_for_status()
            logger.info(f"Mensaje ENVIADO a {to_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando mensaje: {e}")
            if attempt == max_retries - 1:
                return False
    
    return False

async def send_whatsapp_image_via_url(to_number: str, image_url: str, caption: str = "", max_retries: int = 3):
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}", "Content-Type": "application/json"}
    
    payload = {"to": to_number, "imageUrl": image_url, "text": caption}
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt
                logger.info(f"Reintentando enviar imagen (URL) en {delay} segundos")
                await asyncio.sleep(delay)
            
            response = await whatsapp_client.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=30.0)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 65))
                logger.warning(f"Límite de velocidad alcanzado para imagen. Esperando {retry_after} segundos...")
                await asyncio.sleep(retry_after)
                continue
                
            response.raise_for_status()
            logger.info(f"Imagen (URL) ENVIADA a {to_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error enviando imagen (URL): {e}")
            if attempt == max_retries - 1:
                return False
    
    return False

async def process_message_queue():
    global processing_queue
    if processing_queue: 
        return
    processing_queue = True
    while message_queue:
        message_type, to_number, content = message_queue.popleft()
        
        if message_type == "text":
            await send_whatsapp_message_with_retry(to_number, content)
        elif message_type == "image":
            image_url, caption = content
            await send_whatsapp_image_via_url(to_number, image_url, caption)
        
        logger.info("⏳ Esperando 65 segundos para cumplir con el límite de la API (1 msg/min)...")
        await asyncio.sleep(65)
    processing_queue = False

# ============================================
# ENDPOINTS DE LA API (ACTUALIZADOS A ASYNC)
# ============================================

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        html_file = Path("html_updated_ngrok.html")
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="<h1>HTML no encontrado</h1>", status_code=404)
    except Exception as e:
        logger.error(f"Error: {e}")
        return HTMLResponse(content="<h1>Error del servidor</h1>", status_code=500)

@app.get("/api-status")
async def api_status():
    database_stats = await database.get_stats_like_old_system()
    return {
        "message": "Face Recognition API v5.3.0-SQLite",
        "status": "running",
        "system": processor.get_system_status(),
        "database": database_stats
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
            from PIL import Image
            img_pil = Image.open(io.BytesIO(content))
            img_pil.verify()
            img_pil = Image.open(io.BytesIO(content))
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            img_pil.save(filepath, 'JPEG', quality=85)
        except Exception as e:
            logger.error(f"Error PIL: {e}")
            return {"success": False, "message": "Imagen corrupta", "photo_id": photo_id, "faces_detected": 0}
        
        try:
            faces_data = processor.detect_and_encode_faces(str(filepath))
        except Exception as e:
            logger.error(f"Error OpenCV: {e}")
            faces_data = []
            
        await database.add_photo_like_old_system(photo_id, file.filename or "unknown.jpg", str(filepath), faces_data)
        
        return {"success": True, "message": "Foto procesada", "photo_id": photo_id, "faces_detected": len(faces_data)}
    except Exception as e:
        logger.error(f"Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-face")
async def search_face(file: UploadFile = File(...), threshold: float = Query(0.7)):
    temp_filepath = None
    try:
        temp_id = str(uuid.uuid4())
        temp_filepath = UPLOAD_DIR / f"search_{temp_id}.jpg"
        content = await file.read()
        with open(temp_filepath, "wb") as buffer:
            buffer.write(content)
        
        search_faces = processor.detect_and_encode_faces(str(temp_filepath), save_faces=False)
        if not search_faces:
            return {"success": True, "matches_found": 0, "matches": []}
        
        search_face = max(search_faces, key=lambda x: x.get('confidence', 0))
        
        matches = await database.search_similar_faces(search_face, threshold)
        
        response_payload = {"success": True, "matches_found": len(matches), "matches": matches}
        logger.info(f"📤 Enviando respuesta JSON al frontend: {json.dumps(response_payload, indent=2)}")
        return response_payload
    except Exception as e:
        logger.error(f"Error search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_filepath and temp_filepath.exists():
            temp_filepath.unlink()

@app.get("/api/photos")
async def get_photos():
    photos = await database.get_all_photos_like_old_system()
    return {"success": True, "photos": photos}

@app.get("/api/photos/{photo_id}/faces")
async def get_photo_faces(photo_id: str):
    photo = await database.get_photo_like_old_system(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    photo_faces = await database.get_faces_by_photo_id(photo_id)
    
    formatted_faces = []
    for face in photo_faces:
        bbox = face.get('bbox', [0, 0, 0, 0])
        formatted_face = {
            'id': face['id'],
            'photo_id': face['product_id'],
            'bounding_box': {
                'x': int(bbox[0]) if len(bbox) > 0 else 0,
                'y': int(bbox[1]) if len(bbox) > 1 else 0,
                'width': int(bbox[2]) if len(bbox) > 2 else 0,
                'height': int(bbox[3]) if len(bbox) > 3 else 0,
            },
            'confidence': float(face.get('confidence', 0.0)),
            'face_name': face.get('customer_name', ''),
            'face_notes': face.get('customer_notes', '')
        }
        formatted_faces.append(formatted_face)
    
    return {"success": True, "photo_id": photo_id, "faces_count": len(formatted_faces), "faces": formatted_faces}

@app.patch("/api/faces/{face_id}/info")
async def update_face_info(face_id: str, info: dict):
    name = info.get('name', '').strip()
    notes = info.get('notes', '').strip()
    
    success = await database.update_face_info_like_old_system(face_id, name, notes)
    if not success:
        raise HTTPException(status_code=404, detail="Cara no encontrada")
    
    return {"success": True, "message": "Informacion actualizada", "face_id": face_id}

@app.get("/api/faces/search-by-name")
async def search_faces_by_name(name: str = Query(...)):
    name_search = name.strip().lower()
    all_faces = await database.get_faces_by_photo_id("")
    matching_faces = []
    logger.warning("Búsqueda por nombre no implementada directamente en ChromaDB en este ejemplo.")
    
    return {"success": True, "name_searched": name, "faces_found": len(matching_faces), "faces": matching_faces}

@app.get("/api/people")
async def get_people_list():
    logger.warning("Lista de personas no implementada directamente en ChromaDB en este ejemplo.")
    return {"success": True, "total_people": 0, "people": []}

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

@app.api_route("/api/image/face/{face_id}", methods=["GET", "HEAD"])
async def get_face_image(face_id: str, request: Request):
    logger.info(f"Solicitud de imagen de cara para face_id: {face_id}")
    face_filename = database.get_face_filepath(face_id)
    if not face_filename:
        logger.warning(f"404: Cara no encontrada en la base de datos para face_id: {face_id}")
        raise HTTPException(status_code=404, detail="Cara no encontrada")
    
    face_filepath = FACES_DIR / face_filename
    if not face_filepath.exists():
        logger.warning(f"404: Archivo de cara no encontrado en el sistema de archivos para filepath: {face_filepath}")
        raise HTTPException(status_code=404, detail="Archivo de cara no encontrado")
    
    if request.method == "HEAD":
        return Response(headers={"Content-Type": "image/jpeg", "Content-Length": str(face_filepath.stat().st_size)})
    
    return FileResponse(path=face_filepath, media_type="image/jpeg")

@app.get("/faces/{face_id}")
async def get_face_image_direct(face_id: str):
    try:
        face_filepath = FACES_DIR / f"{face_id}.jpg"
        if face_filepath.exists():
            return FileResponse(path=face_filepath, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Cara no encontrada")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sirviendo imagen de cara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "5.3.0-SQLite"}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    result = await database.delete_photo_like_old_system(photo_id)
    if result['success']:
        return {"success": True, "message": "Foto eliminada correctamente", "photo_id": photo_id, "faces_deleted": result['faces_deleted']}
    else:
        raise HTTPException(status_code=404, detail=result.get('error', 'Foto no encontrada'))

@app.get("/debug_database", response_class=JSONResponse)
async def debug_database_state():
    try:
        database._verify_database_consistency()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Depuración completada. Revisa los logs para detalles."
            }
        )
    except Exception as e:
        logger.error(f"Error en el endpoint de depuración: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error durante la depuración: {e}"
            }
        )

@app.post("/force_cleanup", response_class=JSONResponse)
async def force_database_cleanup():
    try:
        logger.info("🚀 Iniciando limpieza forzada de la base de datos via endpoint...")
        
        orphaned_count = await database._cleanup_orphaned_faces()
        invalid_id_count = await database._cleanup_invalid_product_ids()
        
        total_cleaned = orphaned_count + invalid_id_count
        
        message = f"Limpieza completada. Se eliminaron {orphaned_count} caras huérfanas y {invalid_id_count} caras con ID inválido."
        logger.info(f"✅ {message}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": message,
                "orphaned_faces_deleted": orphaned_count,
                "invalid_id_faces_deleted": invalid_id_count,
                "total_faces_deleted": total_cleaned
            }
        )
    except Exception as e:
        logger.error(f"Error en el endpoint de limpieza forzada: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error durante la limpieza forzada: {e}"
            }
        )

@app.get("/inspect_chroma", response_class=JSONResponse)
async def inspect_chroma_collection():
    try:
        logger.info("🔬 Iniciando inspección de la colección ChromaDB...")
        all_results = database.face_collection.get(include=['metadatas'])
        
        if not all_results or not all_results['metadatas']:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "message": "Inspección completada.",
                    "product_ids": []
                }
            )
        
        product_ids = [meta.get('product_id') for meta in all_results['metadatas']]
        unique_product_ids = sorted(list(set(product_ids)))

        logger.info(f"🔬 Inspección completada. Se encontraron {len(unique_product_ids)} product_ids únicos.")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Inspección completada.",
                "total_faces_in_collection": len(all_results['metadatas']),
                "unique_product_ids_found": unique_product_ids
            }
        )
    except Exception as e:
        logger.error(f"Error en el endpoint de inspección: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error durante la inspección: {e}"
            }
        )

@app.post("/reset_database", response_class=JSONResponse)
async def reset_database():
    try:
        logger.warning("🚨 Iniciando reset completo de la base de datos...")
        
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info(f"🗑️ Archivo SQLite eliminado: {DB_PATH}")
        
        if CHROMA_DB_PATH.exists():
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info(f"🗑️ Directorio ChromaDB eliminado: {CHROMA_DB_PATH}")
            CHROMA_DB_PATH.mkdir(exist_ok=True)
        
        global database
        database = ChromaFaceDatabase()
        await database.initialize()
        
        logger.info("✅ Base de datos reseteada y reinicializada correctamente.")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Base de datos reseteada correctamente. Todos los datos han sido eliminados."
            }
        )
    except Exception as e:
        logger.error(f"Error al resetear la base de datos: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error al resetear la base de datos: {e}"
            }
        )

@app.get("/api/searched-clients")
async def get_searched_clients():
    clients = await database.get_all_searched_clients()
    return {"success": True, "clients": clients}

app.mount("/static", StaticFiles(directory="."))

# ============================================
# WEBHOOK DE WHATSAPP
# ============================================

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    try:
        request_body = await request.body()
        if not request_body:
            logger.warning("⚠️ Webhook recibido sin cuerpo de solicitud")
            return {"status": "ignored"}
            
        try:
            data = json.loads(request_body.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error decodificando JSON del webhook: {e}")
            return {"status": "error", "detail": "Invalid JSON"}
            
        logger.info(f"📥 Webhook recibido de WhatsApp")

        messages = data.get("data", {}).get("messages", {})
        message_data = messages
        
        user_number_full = message_data.get("key", {}).get("remoteJid", "")
        user_number = user_number_full.replace("@s.whatsapp.net", "").replace("+", "")
        
        if not user_number:
            logger.warning("⚠️ Mensaje sin número de remitente")
            return {"status": "ignored"}

        message_content = message_data.get("message", {})
        
        if "imageMessage" in message_content:
            logger.info(f"📷 Procesando imagen de {user_number}...")
            
            image_message = message_content.get("imageMessage", {})
            caption = image_message.get("caption", "").strip().lower()
            image_url = image_message.get("url")
            media_key = image_message.get("mediaKey")
            file_enc_sha256 = image_message.get("fileEncSha256")
            
            if not image_url or not media_key or not file_enc_sha256:
                message_queue.append(("text", user_number, "❌ No pude obtener la imagen o sus claves."))
                asyncio.create_task(process_message_queue())
                return {"status": "ok"}

            try:
                logger.info("🔄 Descargando imagen encriptada...")
                response = await whatsapp_client.get(image_url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
                encrypted_data = response.content
                logger.info(f"✅ Descargado: {len(encrypted_data)} bytes encriptados")
                
                logger.info("🔓 Desencriptando imagen...")
                decrypted_data = decrypt_whatsapp_media(encrypted_data, media_key, file_enc_sha256, 'image')
                logger.info(f"✅ Desencriptado: {len(decrypted_data)} bytes")
                
                if decrypted_data.startswith(b'\xff\xd8'):
                    image_bytes = decrypted_data
                elif decrypted_data.startswith(b'\x89PNG'):
                    image_bytes = decrypted_data
                else:
                    try:
                        from PIL import Image
                        test_img = Image.open(io.BytesIO(decrypted_data))
                        test_img.verify()
                        image_bytes = decrypted_data
                    except Exception as e:
                        logger.error(f"❌ Imagen inválida: {e}")
                        message_queue.append(("text", user_number, "❌ Error procesando imagen."))
                        asyncio.create_task(process_message_queue())
                        return {"status": "ok"}
            except Exception as e:
                logger.error(f"Error descargando/desencriptando: {e}")
                message_queue.append(("text", user_number, "❌ Error con tu imagen. Intenta de nuevo."))
                asyncio.create_task(process_message_queue())
                return {"status": "ok"}
            
            if caption == "1":
                logger.info(f"🔍 Buscando coincidencias para {user_number}...")
                try:
                    search_id = str(uuid.uuid4())
                    search_filename = f"search_{search_id}.jpg"
                    search_filepath = UPLOAD_DIR / search_filename
                    
                    with open(search_filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    search_faces = processor.detect_and_encode_faces(str(search_filepath), save_faces=False)
                    
                    if not search_faces:
                        message_queue.append(("text", user_number, "❌ No detecté una cara clara. "))
                    else:
                        search_face = max(search_faces, key=lambda x: x.get('confidence', 0))
                        
                        files = {'file': (search_filename, image_bytes, 'image/jpeg')}
                        search_response = await whatsapp_client.post(
                            "http://127.0.0.1:8000/api/search-face", 
                            files=files, 
                            params={"threshold": 0.5}
                        )
                        
                        if search_response.status_code == 200:
                            result = search_response.json()
                            matches = result.get('matches', [])
                            
                            if matches:
                                message_queue.append(("text", user_number, f"✅ Encontré {len(matches)} coincidencias:"))
                                
                                for i, match in enumerate(matches[:5]):
                                    product_id = match['photo_id']
                                    similarity_percent = round(match['similarity'] * 100)
                                    
                                    face_id = search_face['face_id']
                                    face_filename = f"{face_id}.jpg"
                                    face_filepath = FACES_DIR / face_filename
                                    
                                    cv2.imwrite(str(face_filepath), search_face['face_image'])
                                    
                                    await database.add_searched_client(
                                        client_id=search_id,
                                        phone_number=user_number,
                                        face_image_path=face_id,
                                        best_match_photo_id=product_id,
                                        best_match_similarity=match['similarity']
                                    )
                                    
                                    public_image_url = f"{BASE_URL}/api/image/photo/{product_id}"
                                    message_queue.append(("image", user_number, (public_image_url, f"Coincidencia {i+1}: Similitud {similarity_percent}%")))
                            else:
                                face_id = search_face['face_id']
                                face_filename = f"{face_id}.jpg"
                                face_filepath = FACES_DIR / face_filename
                                
                                cv2.imwrite(str(face_filepath), search_face['face_image'])
                                
                                await database.add_searched_client(
                                    client_id=search_id,
                                    phone_number=user_number,
                                    face_image_path=face_id
                                )
                                message_queue.append(("text", user_number, "❌ No encontré caras similares."))
                        else:
                            message_queue.append(("text", user_number, "❌ Error interno durante la búsqueda."))
                    
                    asyncio.create_task(process_message_queue())
                        
                except Exception as e:
                    logger.error(f"Error búsqueda: {e}")
                    message_queue.append(("text", user_number, "❌ Error inesperado."))
                    asyncio.create_task(process_message_queue())
            
            else:
                message_queue.append(("text", user_number, "¡Hola! Envíame una selfie con el número 1 y te buscaré."))
                asyncio.create_task(process_message_queue())
            
            return {"status": "ok"}
            
        user_message = ""
        if "conversation" in message_content:
            user_message = message_content.get("conversation", "").strip().lower()
        elif "extendedTextMessage" in message_content:
            user_message = message_content.get("extendedText", {}).get("text", "").strip().lower()

        if user_message:
            logger.info(f"💬 Mensaje de {user_number}: '{user_message}'")
            message_queue.append(("text", user_number, "¡Hola! Envíame una selfie con el número 1 y te buscaré."))
            asyncio.create_task(process_message_queue())
            return {"status": "ok"}
            
    except Exception as e:
        logger.error(f"❌ Error webhook: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "detail": str(e)}

    return {"status": "ok"}

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
    print("Face Recognition API v5.3.0-SQLite")
    print("=" * 60)
    print("Servidor: http://localhost:8000")
    print("Webhook: /webhook/whatsapp")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("AVISO: La espera entre mensajes de WhatsApp es de 65 segundos.")
    print("=" * 60)
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
