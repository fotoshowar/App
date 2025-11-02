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

# --- IMPORTS PARA PIL ---
from PIL import Image, ImageOps, ExifTags

# --- CONFIGURACIÓN ---
# CORRECCIÓN: Usar IP pública en lugar de ngrok
BASE_URL = "http://191.97.117.104:8888"
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

app = FastAPI(title="Advanced Face Recognition API", version="5.3.1-SQLite-PIL-Fixed")

# CORS
# CORRECCIÓN: Permitir acceso desde la IP pública y localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# CORRECCIÓN: Eliminar middleware específico para ngrok
# Mantener un middleware genérico para headers CORS
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============================================
# FUNCIÓN PARA CORREGIR ORIENTACIÓN CON PIL
# ============================================

def fix_image_orientation(image_path: str) -> str:
    """
    Corrige la orientación de una imagen usando los datos EXIF con PIL.
    Devuelve la ruta de la imagen corregida.
    """
    try:
        logger.info(f"🔄 Corrigiendo orientación de imagen: {image_path}")
        
        # Abrir imagen con PIL
        image = Image.open(image_path)
        
        # Obtener orientación EXIF
        try:
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        orientation = value
                        logger.info(f"📐 Orientación EXIF detectada: {orientation}")
                        
                        # Aplicar rotación según la orientación
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
                        break
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer la orientación EXIF: {e}")
        
        # Guardar imagen corregida
        corrected_path = image_path.replace(".", "_corrected.")
        image.save(corrected_path, 'JPEG', quality=95)
        
        # Reemplazar el archivo original
        os.replace(corrected_path, image_path)
        
        logger.info(f"✅ Orientación corregida para: {image_path}")
        return image_path
        
    except Exception as e:
        logger.error(f"❌ Error corrigiendo orientación: {e}")
        return image_path

def fix_image_orientation_from_bytes(image_bytes: bytes) -> bytes:
    """
    Corrige la orientación de una imagen desde bytes usando los datos EXIF con PIL.
    Devuelve los bytes de la imagen corregida.
    """
    try:
        logger.info("🔄 Corrigiendo orientación de imagen desde bytes")
        
        # Abrir imagen con PIL desde bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Obtener orientación EXIF
        try:
            exif = image._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'Orientation':
                        orientation = value
                        logger.info(f"📐 Orientación EXIF detectada: {orientation}")
                        
                        # Aplicar rotación según la orientación
                        if orientation == 3:
                            image = image.rotate(180, expand=True)
                        elif orientation == 6:
                            image = image.rotate(270, expand=True)
                        elif orientation == 8:
                            image = image.rotate(90, expand=True)
                        break
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer la orientación EXIF: {e}")
        
        # Convertir a bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        corrected_bytes = img_byte_arr.getvalue()
        
        logger.info(f"✅ Orientación corregida desde bytes ({len(corrected_bytes)} bytes)")
        return corrected_bytes
        
    except Exception as e:
        logger.error(f"❌ Error corrigiendo orientación desde bytes: {e}")
        return image_bytes

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
# CLASE DE PROCESAMIENTO FACIAL
# ============================================

class AdvancedFaceProcessorIntegration:
    def __init__(self):
        logger.info("Inicializando Face Processor...")
        self.processor = None
        self.models_status = {}
        self.init_advanced_processor()
    
    def init_advanced_processor(self):
        try:
            logger.info("Intentando cargar AdvancedFaceProcessor...")
            from advanced_face_processor import AdvancedFaceProcessor
            device = 'cuda' if self._check_gpu_availability() else 'cpu'
            self.processor = AdvancedFaceProcessor(device=device)
            self.models_status['advanced_processor'] = True
            logger.info("AdvancedFaceProcessor cargado exitosamente")
            return
        except ImportError:
            logger.error("advanced_face_processor.py no encontrado")
            self.models_status['advanced_processor'] = False
        except Exception as e:
            logger.error(f"Error cargando AdvancedFaceProcessor: {e}")
            self.models_status['advanced_processor'] = False
        
        logger.warning("Usando procesador fallback")
        self.init_fallback_processor()
    
    def _check_gpu_availability(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def init_fallback_processor(self):
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_img)
            self.models_status['face_recognition'] = True
            logger.info("face_recognition disponible")
        except Exception as e:
            logger.error(f"face_recognition no disponible: {e}")
            self.models_status['face_recognition'] = False
        
        try:
            self.haar_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.models_status['opencv'] = True
            logger.info("OpenCV disponible")
        except Exception as e:
            logger.error(f"OpenCV no disponible: {e}")
            self.models_status['opencv'] = False
    
    def detect_and_encode_faces(self, image_path: str, save_faces: bool = True) -> List[Dict]:
        """
        Detecta y codifica caras en una imagen, estandarizando todos los embeddings a face_recognition.
        """
        try:
            # CORRECCIÓN: Primero corregir la orientación con PIL
            fixed_image_path = fix_image_orientation(image_path)
            
            img = cv2.imread(str(fixed_image_path))
            if img is None:
                logger.error(f"No se pudo cargar imagen: {fixed_image_path}")
                return []
            
            processed_faces = []
            
            if self.processor and self.models_status.get('advanced_processor', False):
                logger.info("Usando AdvancedFaceProcessor")
                detected_faces = self.processor.detect_and_encode_faces(img)
                
                if detected_faces:
                    for i, face_data in enumerate(detected_faces):
                        face_id = str(uuid.uuid4())
                        bbox = face_data.get('bbox', {})
                        
                        if isinstance(bbox, dict):
                            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                        else:
                            x, y, w, h = bbox
                        
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        x = max(0, min(x, img.shape[1] - 1))
                        y = max(0, min(y, img.shape[0] - 1))
                        w = max(1, min(w, img.shape[1] - x))
                        h = max(1, min(h, img.shape[0] - y))
                        
                        face_img = img[y:y+h, x:x+w]
                        
                        if face_img.size > 0:
                            face_filename = None
                            if save_faces:
                                face_filename = f"{face_id}.jpg"
                                face_path = FACES_DIR / face_filename
                                cv2.imwrite(str(face_path), face_img)
                            
                            # CORRECCIÓN CLAVE: Forzar la generación de un embedding face_recognition (128-dim)
                            standardized_embedding = None
                            try:
                                rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_encodings = face_recognition.face_encodings(rgb_face_img)
                                if face_encodings:
                                    standardized_embedding = face_encodings[0].tolist()
                            except Exception as e:
                                logger.error(f"Error generando embedding face_recognition fallback para cara {face_id}: {e}")

                            if standardized_embedding:
                                processed_face = {
                                    'face_id': face_id,
                                    'face_filename': face_filename,
                                    'bbox': [x, y, w, h],
                                    'confidence': float(face_data.get('confidence', 0.95)),
                                    'embeddings': {'face_recognition': standardized_embedding},
                                    'method': 'advanced_standardized',
                                    'landmarks': safe_convert_for_json(face_data.get('landmarks', [])),
                                    'face_image': face_img,
                                    'models_used': ['face_recognition'],
                                    'processing_quality': 'advanced'
                                }
                                processed_faces.append(processed_face)
                            else:
                                logger.warning(f"⚠️ Omitiendo cara {face_id} porque no se pudo generar un embedding estándar.")
                    
                    logger.info(f"AdvancedFaceProcessor: {len(processed_faces)} caras procesadas y estandarizadas.")
                    return processed_faces
            
            # Si el advanced processor falla o no está disponible, usar el fallback
            return self.fallback_detection(img, save_faces)
            
        except Exception as e:
            logger.error(f"Error en deteccion: {e}")
            return self.fallback_detection(cv2.imread(str(image_path)), save_faces)
    
    def detect_and_encode_faces_from_bytes(self, image_bytes: bytes, save_faces: bool = True) -> List[Dict]:
        """
        Detecta y codifica caras desde bytes de imagen, estandarizando todos los embeddings a face_recognition.
        """
        try:
            # CORRECCIÓN: Primero corregir la orientación con PIL
            fixed_image_bytes = fix_image_orientation_from_bytes(image_bytes)
            
            # Convertir bytes a numpy array para OpenCV
            nparr = np.frombuffer(fixed_image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("No se pudo decodificar imagen desde bytes")
                return []
            
            processed_faces = []
            
            if self.processor and self.models_status.get('advanced_processor', False):
                logger.info("Usando AdvancedFaceProcessor desde bytes")
                detected_faces = self.processor.detect_and_encode_faces(img)
                
                if detected_faces:
                    for i, face_data in enumerate(detected_faces):
                        face_id = str(uuid.uuid4())
                        bbox = face_data.get('bbox', {})
                        
                        if isinstance(bbox, dict):
                            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                        else:
                            x, y, w, h = bbox
                        
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        x = max(0, min(x, img.shape[1] - 1))
                        y = max(0, min(y, img.shape[0] - 1))
                        w = max(1, min(w, img.shape[1] - x))
                        h = max(1, min(h, img.shape[0] - y))
                        
                        face_img = img[y:y+h, x:x+w]
                        
                        if face_img.size > 0:
                            face_filename = None
                            if save_faces:
                                face_filename = f"{face_id}.jpg"
                                face_path = FACES_DIR / face_filename
                                cv2.imwrite(str(face_path), face_img)
                            
                            # CORRECCIÓN CLAVE: Forzar la generación de un embedding face_recognition (128-dim)
                            standardized_embedding = None
                            try:
                                rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_encodings = face_recognition.face_encodings(rgb_face_img)
                                if face_encodings:
                                    standardized_embedding = face_encodings[0].tolist()
                            except Exception as e:
                                logger.error(f"Error generando embedding face_recognition fallback para cara {face_id}: {e}")

                            if standardized_embedding:
                                processed_face = {
                                    'face_id': face_id,
                                    'face_filename': face_filename,
                                    'bbox': [x, y, w, h],
                                    'confidence': float(face_data.get('confidence', 0.95)),
                                    'embeddings': {'face_recognition': standardized_embedding},
                                    'method': 'advanced_standardized',
                                    'landmarks': safe_convert_for_json(face_data.get('landmarks', [])),
                                    'face_image': face_img,
                                    'models_used': ['face_recognition'],
                                    'processing_quality': 'advanced'
                                }
                                processed_faces.append(processed_face)
                            else:
                                logger.warning(f"⚠️ Omitiendo cara {face_id} porque no se pudo generar un embedding estándar.")
                    
                    logger.info(f"AdvancedFaceProcessor desde bytes: {len(processed_faces)} caras procesadas y estandarizadas.")
                    return processed_faces
            
            # Si el advanced processor falla o no está disponible, usar el fallback
            return self.fallback_detection(img, save_faces)
            
        except Exception as e:
            logger.error(f"Error en deteccion desde bytes: {e}")
            return []
    
    def fallback_detection(self, img: np.ndarray, save_faces: bool = True) -> List[Dict]:
        if self.models_status.get('face_recognition', False):
            return self._detect_with_face_recognition(img, save_faces)
        if self.models_status.get('opencv', False):
            return self._detect_with_opencv(img, save_faces)
        return []
    
    def _detect_with_face_recognition(self, img: np.ndarray, save_faces: bool = True) -> List[Dict]:
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            
            if not face_locations:
                return []
            
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
                        'confidence': 0.90,
                        'embeddings': {'face_recognition': encoding.tolist()},
                        'method': 'face_recognition_fallback',
                        'models_used': ['face_recognition'],
                        'processing_quality': 'standard',
                        'face_image': face_img
                    }
                    faces_data.append(face_data)
            
            return faces_data
        except Exception as e:
            logger.error(f"Error en face_recognition: {e}")
            return []
    
    def _detect_with_opencv(self, img: np.ndarray, save_faces: bool = True) -> List[Dict]:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            faces_data = []
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_id = str(uuid.uuid4())
                
                face_filename = None
                if save_faces:
                    face_filename = f"{face_id}.jpg"
                    face_path = FACES_DIR / face_filename
                    cv2.imwrite(str(face_path), face_img)
                
                # CORRECCIÓN: Generar embedding con face_recognition si está disponible
                embeddings = {}
                if self.models_status.get('face_recognition', False):
                    try:
                        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_face_img)
                        if face_encodings:
                            embeddings['face_recognition'] = face_encodings[0].tolist()
                    except Exception as e:
                        logger.error(f"Error generando embedding con face_recognition: {e}")
                
                if not embeddings:
                    # Si no se puede generar un embedding real, omitimos esta cara para mantener consistencia
                    logger.warning(f"⚠️ Omitiendo cara detectada por OpenCV {face_id} porque no se pudo generar un embedding compatible.")
                    continue
                
                face_data = {
                    'face_id': face_id,
                    'face_filename': face_filename,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.75,
                    'embeddings': embeddings,
                    'method': 'opencv_fallback',
                    'models_used': list(embeddings.keys()),
                    'processing_quality': 'basic',
                    'face_image': face_img
                }
                faces_data.append(face_data)
            
            return faces_data
        except Exception as e:
            logger.error(f"Error en OpenCV: {e}")
            return []

    def compare_embeddings(self, embeddings1: Dict, embeddings2: Dict) -> float:
        if self.processor and self.models_status.get('advanced_processor', False):
            try:
                similarity = self.processor.compare_multi_embeddings(embeddings1, embeddings2)
                return float(similarity)
            except Exception as e:
                logger.error(f"Error en comparacion avanzada: {e}")
        
        return self._fallback_comparison(embeddings1, embeddings2)

    def _fallback_comparison(self, embeddings1: Dict, embeddings2: Dict) -> float:
        try:
            if 'face_recognition' in embeddings1 and 'face_recognition' in embeddings2:
                emb1 = np.array(embeddings1['face_recognition'])
                emb2 = np.array(embeddings2['face_recognition'])
                distance = np.linalg.norm(emb1 - emb2)
                return max(0, 1 - distance / 1.2)
            return 0.5
        except Exception as e:
            logger.error(f"Error en comparacion: {e}")
            return 0.0
    
    def get_system_status(self) -> Dict:
        return {
            'advanced_processor_loaded': self.models_status.get('advanced_processor', False),
            'models_status': self.models_status,
            'processing_mode': 'advanced' if self.processor else 'fallback'
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
                logger.info(f"🗑️ {faces_deleted} caras eliminadas de ChromaDB.")

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
                await db.commit()

            # Eliminar archivo de imagen
            try:
                if os.path.exists(photo_data['filepath']):
                    os.remove(photo_data['filepath'])
                    logger.info(f"🗑️ Archivo de imagen eliminado: {photo_data['filepath']}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar el archivo de imagen: {e}")

            return {'success': True, 'faces_deleted': faces_deleted}
        except Exception as e:
            logger.error(f"Error eliminando foto: {e}")
            return {'success': False, 'error': str(e), 'faces_deleted': 0}

    async def add_searched_client(self, client_id: str, phone_number: str, face_image_path: str, 
                                  best_match_photo_id: str = None, best_match_similarity: float = None) -> bool:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO searched_clients 
                    (id, phone_number, search_date, face_image_path, best_match_photo_id, best_match_similarity)
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

    async def get_searched_clients(self) -> List[Dict]:
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

# ============================================
# INICIALIZACIÓN Y ENDPOINTS
# ============================================

# Inicializar procesador facial y base de datos
face_processor = AdvancedFaceProcessorIntegration()
db = ChromaFaceDatabase()

@app.on_event("startup")
async def startup_event():
    await db.initialize()
    logger.info("✅ Aplicación iniciada correctamente")

# ============================================
# ENDPOINTS DE LA API
# ============================================

# CORRECCIÓN: Endpoint para servir el archivo HTML específico
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        # Intentar leer y devolver el archivo html_update_ngrok.html
        html_file = Path("html_update_ngrok.html")
        if html_file.exists():
            with open(html_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
        else:
            # Si el archivo no existe, devolver una página de error
            return HTMLResponse(content="""
            <html>
                <head>
                    <title>FOTOSHOW - Error</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
                        h1 { color: #e53e3e; }
                    </style>
                </head>
                <body>
                    <h1>Error: Archivo HTML no encontrado</h1>
                    <p>No se encontró el archivo html_update_ngrok.html en el directorio actual.</p>
                    <p>Por favor, asegúrate de que el archivo existe en el mismo directorio que main.py</p>
                </body>
            </html>
            """, status_code=404)
    except Exception as e:
        logger.error(f"Error sirviendo el archivo HTML: {e}")
        return HTMLResponse(content=f"""
        <html>
            <head>
                <title>FOTOSHOW - Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; }}
                    h1 {{ color: #e53e3e; }}
                </style>
            </head>
            <body>
                <h1>Error al cargar la página</h1>
                <p>Ocurrió un error al intentar cargar el archivo HTML: {str(e)}</p>
            </body>
        </html>
        """, status_code=500)

@app.get("/api-status")
async def get_api_status():
    return {
        "status": "online",
        "version": "5.3.1-SQLite-PIL-Fixed",
        "database": {
            "total_photos": len(await db.get_all_photos_like_old_system()),
            "total_faces": len(db.face_collection.get()['ids']) if db.face_collection else 0
        },
        "processor": face_processor.get_system_status()
    }

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        # Generar ID único para la foto
        photo_id = str(uuid.uuid4())
        
        # Guardar archivo temporalmente
        temp_path = UPLOAD_DIR / f"{photo_id}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Procesar imagen para detectar caras
        faces_data = face_processor.detect_and_encode_faces(str(temp_path))
        
        # Guardar en base de datos
        await db.add_photo_like_old_system(
            photo_id=photo_id,
            filename=file.filename,
            filepath=str(temp_path),
            faces_data=faces_data
        )
        
        return {
            "success": True,
            "photo_id": photo_id,
            "faces_detected": len(faces_data)
        }
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        return {"success": False, "detail": str(e)}

@app.post("/api/search-face")
async def search_face(
    file: UploadFile = File(...),
    threshold: float = Query(0.7, ge=0.5, le=0.95)
):
    try:
        # Leer archivo
        contents = await file.read()
        
        # Detectar caras en la imagen de búsqueda
        search_faces = face_processor.detect_and_encode_faces_from_bytes(contents, save_faces=False)
        
        if not search_faces:
            return {"success": False, "detail": "No se detectaron caras en la imagen de búsqueda"}
        
        # Buscar coincidencias para cada cara detectada
        all_matches = []
        for face_data in search_faces:
            matches = await db.search_similar_faces(face_data, threshold=threshold)
            all_matches.extend(matches)
        
        # Eliminar duplicados y ordenar por similitud
        unique_matches = []
        seen_photo_ids = set()
        
        for match in sorted(all_matches, key=lambda x: x['similarity'], reverse=True):
            if match['photo_id'] not in seen_photo_ids:
                unique_matches.append(match)
                seen_photo_ids.add(match['photo_id'])
        
        # Guardar cliente buscado (simulado con número de teléfono)
        client_id = str(uuid.uuid4())
        phone_number = f"+{1000000000 + int(hashlib.md5(contents).hexdigest(), 16) % 9000000000}"
        face_image_path = f"{client_id}.jpg"
        
        # Guardar imagen de búsqueda
        face_path = FACES_DIR / face_image_path
        with open(face_path, "wb") as f:
            f.write(contents)
        
        # Guardar en base de datos
        best_match = unique_matches[0] if unique_matches else None
        await db.add_searched_client(
            client_id=client_id,
            phone_number=phone_number,
            face_image_path=face_image_path,
            best_match_photo_id=best_match['photo_id'] if best_match else None,
            best_match_similarity=best_match['similarity'] if best_match else None
        )
        
        return {
            "success": True,
            "matches_found": len(unique_matches),
            "matches": unique_matches[:10]  # Limitar a 10 resultados
        }
    except Exception as e:
        logger.error(f"Error searching face: {e}")
        return {"success": False, "detail": str(e)}

@app.get("/api/photos")
async def get_photos():
    try:
        photos = await db.get_all_photos_like_old_system()
        return {"success": True, "photos": photos}
    except Exception as e:
        logger.error(f"Error getting photos: {e}")
        return {"success": False, "detail": str(e)}

@app.get("/api/photos/{photo_id}")
async def get_photo(photo_id: str):
    try:
        photo = await db.get_photo_like_old_system(photo_id)
        if not photo:
            return {"success": False, "detail": "Foto no encontrada"}
        
        faces = await db.get_faces_by_photo_id(photo_id)
        return {"success": True, "photo": photo, "faces": faces}
    except Exception as e:
        logger.error(f"Error getting photo: {e}")
        return {"success": False, "detail": str(e)}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    try:
        result = await db.delete_photo_like_old_system(photo_id)
        return result
    except Exception as e:
        logger.error(f"Error deleting photo: {e}")
        return {"success": False, "detail": str(e)}

@app.get("/api/searched-clients")
async def get_searched_clients():
    try:
        clients = await db.get_searched_clients()
        return {"success": True, "clients": clients}
    except Exception as e:
        logger.error(f"Error getting searched clients: {e}")
        return {"success": False, "detail": str(e)}

@app.get("/api/image/photo/{photo_id}")
async def get_photo_image(photo_id: str):
    try:
        photo = await db.get_photo_like_old_system(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Foto no encontrada")
        
        if not os.path.exists(photo['filepath']):
            raise HTTPException(status_code=404, detail="Archivo de imagen no encontrado")
        
        return FileResponse(photo['filepath'])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting photo image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faces/{face_filename}")
async def get_face_image(face_filename: str):
    try:
        face_path = FACES_DIR / face_filename
        if not face_path.exists():
            raise HTTPException(status_code=404, detail="Imagen de cara no encontrada")
        
        return FileResponse(face_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# CORRECCIÓN: Configurar para ejecutar en 0.0.0.0:8888
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
