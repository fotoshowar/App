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
import codecs
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

import tempfile
import traceback
from fastapi import File, Form, HTTPException





# --- CONFIGURACIÓN CRÍTICA PARA PYINSTALLER ---
# ESTE BLOQUE DEBE IR ANTES DE IMPORTAR face_recognition
try:
    if getattr(sys, 'frozen', False):
        # Estamos en un ejecutable de PyInstaller
        base_path = Path(sys._MEIPASS)
    else:
        # Estamos en modo desarrollo
        base_path = Path(__file__).parent

    # Construir la ruta completa a la carpeta de modelos
    model_path = base_path / 'face_recognition_models' / 'models'
    
    # PASO 1: Importar la librería de soporte PRIMERO
    import face_recognition_models

    # PASO 2: Parchear su variable de ruta ANTES de que la use
    face_recognition_models.model_path = str(model_path)
    # --- FIN DE LA CONFIGURACIÓN CRÍTICA ---

except Exception as e:
    # Si esto falla, no podemos continuar
    print(f"ERROR FATAL: No se pudieron configurar las rutas de los modelos: {e}")
    sys.exit(1)



# Añade esto después de los imports iniciales
try:
    import multipart
except ImportError:
    print("ERROR: La librería 'python-multipart' es necesaria pero no está instalada.")
    print("Por favor, instálala con: pip install python-multipart")
    sys.exit(1)



# --- CONFIGURACIÓN DE CODIFICACIÓN PARA WINDOWS ---
def configure_encoding():
    """Configura la codificación para manejar correctamente Unicode en Windows"""
    if sys.platform == 'win32':
        # Forzar UTF-8 para la salida estándar
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
        
        # Configurar la variable de entorno para UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'

# Llamar a la función de configuración de codificación
configure_encoding()

# --- CONFIGURACIÓN DE RUTAS BASE ---
def get_application_paths():
    """Obtiene las rutas base de la aplicación para desarrollo y ejecutable"""
    if getattr(sys, 'frozen', False):
        # Modo ejecutable compilado
        application_path = Path(sys.executable).parent.resolve()
        base_path = Path(sys._MEIPASS)  # Temporal donde PyInstaller extrae archivos
        print(f"Modo ejecutable detectado")
        print(f"Aplicación corriendo desde: {application_path}")
        print(f"Ruta de recursos temporales: {base_path}")
    else:
        # Modo desarrollo
        application_path = Path(__file__).parent.resolve()
        base_path = application_path
        print(f"Modo desarrollo detectado")
        print(f"Aplicación corriendo desde: {application_path}")
    
    return application_path, base_path

# Obtener rutas al inicio
APPLICATION_PATH, base_path = get_application_paths()

# --- CONFIGURACIÓN DE LOGGING CON UTF-8 ---
def setup_logging():
    """Configura el logging con soporte para UTF-8"""
    log_file = APPLICATION_PATH / 'face_recognition.log'
    
    # Crear un formateador personalizado que maneje Unicode
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            # Asegurar que el mensaje sea Unicode
            if isinstance(record.msg, bytes):
                record.msg = record.msg.decode('utf-8', errors='replace')
            return super().format(record)
    
    # Configurar handlers
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Forzar reconfiguración
    )
    
    # Aplicar formateador Unicode a todos los handlers
    formatter = UnicodeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
    
    return logging.getLogger(__name__)

# Configurar logging al inicio
logger = setup_logging()
logger.info("Sistema de logging inicializado con soporte UTF-8")

# --- CONFIGURACIÓN DE MODELOS (MEJORADA Y SIMPLIFICADA) ---
def setup_model_paths():
    """Configura las rutas de los modelos de forma robusta, priorizando la carpeta local."""
    global model_data_path, dlib_data_path
    
    model_data_path = None
    dlib_data_path = None
    
    # Determinar la ruta base de la aplicación
    if getattr(sys, 'frozen', False):
        # Modo ejecutable: la carpeta donde está el .exe
        application_base_path = Path(sys.executable).parent
    else:
        # Modo desarrollo: la carpeta del script
        application_base_path = Path(__file__).parent
    
    # Priorizar la carpeta local al ejecutable
    possible_model_paths = [
        application_base_path / 'face_recognition_models' / 'models',  # Ruta prioritaria
        application_base_path / 'models',                             # Ruta alternativa
        application_base_path / 'dlib_data',                           # Ruta para dlib
        # Las rutas antiguas se mantienen como fallback
        base_path / 'models',
        base_path / 'face_recognition_models' / 'models',
        base_path / 'dlib_data',
        APPLICATION_PATH / 'models',
        APPLICATION_PATH / 'face_recognition_models' / 'models',
        APPLICATION_PATH / 'dlib_data',
    ]
    
    # Buscar modelos de face_recognition
    for path in possible_model_paths:
        if (path / 'shape_predictor_68_face_landmarks.dat').exists():
            model_data_path = str(path)
            logger.info(f"Modelos de face_recognition encontrados en: {model_data_path}")
            break
    
    # Buscar modelos de Dlib
    for path in possible_model_paths:
        if (path / 'mmod_human_face_detector.dat').exists():
            dlib_data_path = str(path)
            logger.info(f"Modelos de Dlib encontrados en: {dlib_data_path}")
            break
    
    # Configurar rutas por defecto si no se encuentran
    if model_data_path is None:
        model_data_path = str(application_base_path / 'face_recognition_models' / 'models')
        logger.warning(f"ADVERTENCIA: No se encontraron los modelos de face_recognition. Usando ruta por defecto: {model_data_path}")
    
    if dlib_data_path is None:
        dlib_data_path = str(application_base_path / 'dlib_data')
        logger.warning(f"ADVERTENCIA: No se encontraron los modelos de Dlib. Usando ruta por defecto: {dlib_data_path}")
    
    # Configurar face_recognition_models si está disponible
    try:
        import face_recognition_models
        face_recognition_models.model_path = model_data_path
        logger.info("face_recognition_models configurado correctamente")
    except ImportError:
        logger.warning("ADVERTENCIA: face_recognition_models no está disponible")
    
    # Verificar dlib
    try:
        import dlib
        logger.info("dlib disponible")
    except ImportError:
        logger.warning("ADVERTENCIA: dlib no está disponible")
    
    logger.info(f"Ruta final de modelos de face_recognition: {model_data_path}")
    logger.info(f"Ruta final de modelos de Dlib: {dlib_data_path}")


# --- CONFIGURACIÓN DE CARPETAS DE DATOS (MEJORADA) ---
def setup_data_directories():
    """Configura las carpetas de datos de forma robusta"""
    global UPLOAD_DIR, FACES_DIR, CHROMA_DB_PATH, DB_PATH, OLD_JSON_PATH
    
    # Carpetas principales
    UPLOAD_DIR = APPLICATION_PATH / "uploads"
    FACES_DIR = APPLICATION_PATH / "faces"
    CHROMA_DB_PATH = APPLICATION_PATH / "chroma_db"
    DB_PATH = APPLICATION_PATH / "photos.db"
    OLD_JSON_PATH = APPLICATION_PATH / "products_metadata.json"
    
    # Función para crear carpetas con fallback
    def ensure_directory(path, folder_name):
        try:
            path.mkdir(exist_ok=True)
            logger.info(f"Carpeta {folder_name}: {path}")
            return path
        except Exception as e:
            logger.error(f"Error creando carpeta {folder_name}: {e}")
            # Usar carpeta temporal como fallback
            import tempfile
            fallback_path = Path(tempfile.gettempdir()) / "FotoShow" / folder_name
            fallback_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Usando carpeta alternativa para {folder_name}: {fallback_path}")
            return fallback_path
    
    # Crear todas las carpetas necesarias
    UPLOAD_DIR = ensure_directory(UPLOAD_DIR, "uploads")
    FACES_DIR = ensure_directory(FACES_DIR, "faces")
    CHROMA_DB_PATH = ensure_directory(CHROMA_DB_PATH, "chroma_db")

# --- FUNCIÓN DE INICIALIZACIÓN COMPLETA ---
def initialize_application():
    """Inicializa toda la configuración de la aplicación"""
    print("\n" + "="*60)
    print("INICIANDO CONFIGURACIÓN DE FOTOSHOW")
    print("="*60)
    
    try:
        # Configurar modelos
        print("\nCONFIGURANDO MODELOS...")
        setup_model_paths()
        
        # Configurar carpetas de datos
        print("\nCONFIGURANDO CARPETAS DE DATOS...")
        setup_data_directories()
        
        # Configurar URL base
        print("\nCONFIGURANDO URL BASE...")
        if getattr(sys, 'frozen', False):
            BASE_URL = "http://127.0.0.1:8888"
            print("Modo ejecutable detectado, usando localhost")
        else:
            BASE_URL = "https://communication-honolulu-ensures-difficulty.trycloudflare.com"
            print("Modo desarrollo detectado, usando Cloudflare tunnel")
        
        print(f"URL base configurada: {BASE_URL}")
        print("="*60)
        print("CONFIGURACIÓN COMPLETADA EXITOSAMENTE")
        print("="*60 + "\n")
        
        return BASE_URL
        
    except Exception as e:
        print("="*60)
        print("ERROR CRÍTICO EN LA INICIALIZACIÓN")
        print(f"Error: {e}")
        print("="*60)
        raise e

# Ejecutar la inicialización
BASE_URL = initialize_application()

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
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"No se pudo cargar imagen: {image_path}")
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
                                logger.warning(f"Omitiendo cara {face_id} porque no se pudo generar un embedding estándar.")
                    
                    logger.info(f"AdvancedFaceProcessor: {len(processed_faces)} caras procesadas y estandarizadas.")
                    return processed_faces
            
            # Si el advanced processor falla o no está disponible, usar el fallback
            return self.fallback_detection(img, save_faces)
            
        except Exception as e:
            logger.error(f"Error en deteccion: {e}")
            return self.fallback_detection(cv2.imread(str(image_path)), save_faces)
    
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
                    logger.warning(f"Omitiendo cara detectada por OpenCV {face_id} porque no se pudo generar un embedding compatible.")
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
        logger.info("Inicializando base de datos SQLite...")
        logger.info("La inicialización asíncrona de la BD se ejecutará en el startup event.")

    async def initialize(self):
        await self._setup_database()
        logger.info("Ejecutando autolimpieza y verificación de datos huérfanos...")
        try:
            orphaned_count = await self._cleanup_orphaned_faces()
            if orphaned_count > 0:
                logger.warning(f"Se encontraron y eliminaron {orphaned_count} registros de caras huérfanas al iniciar.")
        except Exception as e:
            logger.error(f"Error durante la limpieza de caras huérfanas: {e}")
        
        try:
            invalid_id_count = await self._cleanup_invalid_product_ids()
            if invalid_id_count > 0:
                logger.warning(f"Se encontraron y eliminaron {invalid_id_count} registros con product_id inválido.")
        except Exception as e:
            logger.error(f"Error durante la limpieza de IDs inválidos: {e}")

        self._verify_database_consistency()
        logger.info("Conectado a ChromaDB y SQLite (Modo Híbrido y Atómico)")

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
            logger.info("ChromaDB inicializado correctamente")
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
                logger.info("ChromaDB recreado e inicializado correctamente")
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
            logger.info("Iniciando migración desde products_metadata.json a SQLite...")
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
                    logger.info(f"Migración completada. Se migraron {len(data['photos'])} fotos.")
                    os.rename(OLD_JSON_PATH, f"{OLD_JSON_PATH}.migrated")
                else:
                    logger.info("products_metadata.json está vacío o no tiene fotos. No se requiere migración.")
            except Exception as e:
                logger.error(f"Error durante la migración desde JSON: {e}")
        else:
            logger.info("products_metadata.json no encontrado. Iniciando con base de datos SQLite nueva.")

    async def _cleanup_invalid_product_ids(self) -> int:
        try:
            logger.info("Iniciando limpieza de product_id inválidos...")
            all_results = self.face_collection.get(include=['metadatas'])
            invalid_ids = []
            
            if not all_results or not all_results['ids']:
                logger.info("No hay caras en la base de datos para limpiar.")
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
                    logger.warning(f"Cara con ID inválido encontrada y será eliminada: {face_id} (product_id: '{product_id}')")
                    invalid_ids.append(face_id)
            
            if invalid_ids:
                self.face_collection.delete(ids=invalid_ids)
                logger.info(f"{len(invalid_ids)} caras con ID inválido eliminadas de ChromaDB.")
            else:
                logger.info("No se encontraron caras con product_id inválido.")
            
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
                    logger.warning(f"Cara huérfana encontrada y será eliminada: {face_id} (product_id: {product_id})")
                    orphaned_ids.append(face_id)
            
            if orphaned_ids:
                self.face_collection.delete(ids=orphaned_ids)
                logger.info(f"{len(orphaned_ids)} caras huérfanas eliminadas de ChromaDB.")
            
            return len(orphaned_ids)
        except Exception as e:
            logger.error(f"Error durante la autolimpieza de caras huérfanas: {e}")
            return 0

    def _verify_database_consistency(self):
        logger.info("Verificación de consistencia entre SQLite y ChromaDB completada por la limpieza de huérfanos.")

    async def add_photo_like_old_system(self, photo_id: str, filename: str, filepath: str, faces_data: List[Dict]):
        logger.info(f"Iniciando guardado ATÓMICO para la foto: {photo_id}")
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
            logger.info(f"Metadatos para la foto {photo_id} guardados en SQLite.")

            if not faces_data:
                logger.warning(f"No hay caras para guardar para la foto {photo_id}.")
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
                    logger.warning(f"Omitiendo cara {face_data.get('face_id', 'N/A')} porque no tiene un embedding 'face_recognition' compatible.")
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
            
            logger.info(f"Se prepararon {valid_faces_count} caras válidas para ChromaDB.")

            # CORRECCIÓN: Verificación final de consistencia de dimensiones
            if embeddings:
                first_dim = len(embeddings[0])
                inconsistent_dims = [len(emb) for emb in embeddings if len(emb) != first_dim]
                if inconsistent_dims:
                    logger.error(f"ERROR CRÍTICO: Inconsistencia de dimensiones encontrada. Dimensión esperada: {first_dim}, Dimensiones encontradas: {set(inconsistent_dims)}")
                    raise ValueError("Inconsistent dimensions in provided embeddings after final check.")
                
                logger.info(f"Añadiendo {len(embeddings)} embeddings a ChromaDB (Dimensión: {first_dim})...")
                try:
                    self.face_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    temp_chroma_ids_to_delete = ids
                    logger.info(f"{len(embeddings)} embeddings guardados en ChromaDB.")
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
                        logger.info(f"{len(embeddings)} embeddings guardados en ChromaDB después de recrear la colección.")
                    except Exception as retry_error:
                        logger.error(f"Error incluso al recrear la colección: {retry_error}")
                        raise e
            else:
                logger.warning(f"No se guardaron embeddings en ChromaDB.")

        except Exception as e:
            logger.error(f"ERROR FATAL durante el guardado de la foto {photo_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.error("Iniciando ROLLBACK de ChromaDB...")
            if temp_chroma_ids_to_delete:
                try:
                    self.face_collection.delete(ids=temp_chroma_ids_to_delete)
                    logger.info(f"Embeddings añadidos a ChromaDB fueron eliminados durante el rollback.")
                except Exception as delete_error:
                    logger.error(f"ERROR FATAL durante la eliminación de embeddings de ChromaDB: {delete_error}")
            
            raise e

    async def search_similar_faces(self, search_embeddings: Dict, threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        logger.info(f"Iniciando búsqueda HÍBRIDA (iterativa) con threshold={threshold}")
        try:
            search_embedding = None
            if 'embeddings' in search_embeddings and 'face_recognition' in search_embeddings['embeddings']:
                search_embedding = search_embeddings['embeddings']['face_recognition']
                logger.info("Usando embedding 'face_recognition' para la búsqueda (método compatible).")
            else:
                logger.warning("Búsqueda fallida: La cara de búsqueda no tiene un embedding 'face_recognition' compatible con la base de datos.")
                return []
            
            if not search_embedding:
                logger.warning("Búsqueda fallida: No se encontró ningún embedding válido.")
                return []

            all_results = self.face_collection.get(include=['embeddings', 'metadatas'])
            matches = []
            
            if all_results['ids']:
                total_faces = len(all_results['ids'])
                logger.info(f"Iterando sobre {total_faces} caras en la base de datos...")
                
                for i, face_id in enumerate(all_results['ids']):
                    current_embedding = all_results['embeddings'][i]
                    metadata = all_results['metadatas'][i]
                    product_id = metadata.get('product_id')

                    if not product_id or str(product_id).strip().lower() in ["undefined", "null", "none", ""]:
                        logger.error(f"Ignorando cara {face_id} con product_id inválido durante la búsqueda: '{product_id}'")
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
            logger.info(f"Búsqueda híbrida completada. {len(matches)} coincidencias encontradas.")
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
            logger.info(f"Info de cara {face_id} actualizada.")
            return True
        except Exception as e:
            logger.error(f"Error actualizando info de cara: {e}")
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
            logger.info(f"Cliente buscado guardado: {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error guardando cliente buscado: {e}")
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
            logger.error(f"Error obteniendo clientes buscados: {e}")
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
            from PIL import Image, ImageOps  # <-- Asegúrate de importar ImageOps
            
            # Abrir la imagen desde los bytes en memoria
            img_pil = Image.open(io.BytesIO(content))
            
            # Verificar que la imagen no esté corrupta ANTES de cualquier operación
            img_pil.verify() 
            # Necesitamos re-abrir la imagen después de verify()
            img_pil = Image.open(io.BytesIO(content))

            # --- LA SOLUCIÓN CLAVE ---
            # Usar ImageOps.exif_transpose para corregir la orientación automáticamente.
            # Esta función es mucho más robusta. Si no hay datos de orientación, no hace nada.
            try:
                img_pil = ImageOps.exif_transpose(img_pil)
            except Exception as e:
                # Si por alguna razón rara falla, continuamos con la imagen original.
                # Es mejor tener la imagen mal rotada a que no se cargue nada.
                logger.warning(f"No se pudo corregir la orientación con exif_transpose: {e}. Continuando con la imagen original.")

            # Convertir a RGB si es necesario (antes de guardar)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            # Guardar la imagen ya orientada correctamente
            img_pil.save(filepath, 'JPEG', quality=85)
            
        except Exception as e:
            logger.error(f"Error PIL: {e}")
            return {"success": False, "message": "Imagen corrupta o no válida", "photo_id": photo_id, "faces_detected": 0}
        
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

@app.get("/debug-paths")
async def debug_paths():
    """Endpoint de depuración para verificar rutas clave."""
    import sys
    import os
    import face_recognition_models
    
    debug_info = {
        "sys.frozen": getattr(sys, 'frozen', False),
        "sys.executable": sys.executable if getattr(sys, 'frozen', False) else "N/A (script mode)",
        "sys._MEIPASS": sys._MEIPASS if getattr(sys, 'frozen', False) else "N/A (script mode)",
        "os.getcwd()": os.getcwd(),
        "APPLICATION_PATH (from main.py)": str(APPLICATION_PATH),
    }
    
    # Comprobar la ruta del parche de face_recognition_models
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        model_data_path = os.path.join(base_path, 'face_recognition_models', 'models')
        debug_info["face_recognition_models.model_path"] = model_data_path
        debug_info["face_recognition_models.models exists"] = os.path.exists(model_data_path)
    else:
        debug_info["face_recognition_models.model_path"] = "N/A (script mode)"
        debug_info["face_recognition_models.models exists"] = "N/A (script mode)"

    return debug_info

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
        logger.info("Iniciando limpieza forzada de la base de datos via endpoint...")
        
        orphaned_count = await database._cleanup_orphaned_faces()
        invalid_id_count = await database._cleanup_invalid_product_ids()
        
        total_cleaned = orphaned_count + invalid_id_count
        
        message = f"Limpieza completada. Se eliminaron {orphaned_count} caras huérfanas y {invalid_id_count} caras con ID inválido."
        logger.info(f"{message}")
        
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
        logger.info("Iniciando inspección de la colección ChromaDB...")
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

        logger.info(f"Inspección completada. Se encontraron {len(unique_product_ids)} product_ids únicos.")
        
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
        logger.warning("Iniciando reset completo de la base de datos...")
        
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            logger.info(f"Archivo SQLite eliminado: {DB_PATH}")
        
        if CHROMA_DB_PATH.exists():
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info(f"Directorio ChromaDB eliminado: {CHROMA_DB_PATH}")
            CHROMA_DB_PATH.mkdir(exist_ok=True)
        
        global database
        database = ChromaFaceDatabase()
        await database.initialize()
        
        logger.info("Base de datos reseteada y reinicializada correctamente.")
        
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

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================
# ENDPOINT DE BÚSQUEDA DE CARAS (FALTANTE)
# ============================================

@app.post("/api/search-face")
async def search_face(
    file: UploadFile = File(...), 
    threshold: float = Form(0.7)
):
    """
    Endpoint para buscar caras similares en la base de datos
    """
    try:
        logger.info(f"Iniciando búsqueda de cara con threshold: {threshold}")
        
        # Validar que sea una imagen
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Guardar temporalmente la imagen
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        logger.info(f"Imagen temporal guardada en: {temp_path}")
        
        # Procesar la imagen para detectar caras
        faces_data = processor.detect_and_encode_faces(temp_path, save_faces=False)
        
        if not faces_data:
            logger.warning("No se detectaron caras en la imagen de búsqueda")
            os.unlink(temp_path)
            return {
                "success": True, 
                "matches_found": 0, 
                "matches": [],
                "message": "No se detectaron caras en la imagen"
            }
        
        logger.info(f"Se detectaron {len(faces_data)} caras en la imagen")
        
        # Buscar coincidencias en la base de datos
        all_matches = []
        for i, face_data in enumerate(faces_data):
            logger.info(f"Buscando coincidencias para la cara {i+1}")
            
            search_results = await database.search_similar_faces(
                {"embeddings": face_data["embeddings"]}, 
                threshold=threshold,
                limit=50
            )
            
            logger.info(f"Cara {i+1}: {len(search_results)} coincidencias encontradas")
            all_matches.extend(search_results)
        
        # Eliminar duplicados (mismas fotos de diferentes caras)
        unique_matches = []
        seen_photo_ids = set()
        
        for match in all_matches:
            if match["photo_id"] not in seen_photo_ids:
                unique_matches.append(match)
                seen_photo_ids.add(match["photo_id"])
        
        # Ordenar por similitud descendente
        unique_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limitar resultados a los mejores 20
        final_matches = unique_matches[:20]
        
        # Limpiar archivo temporal
        os.unlink(temp_path)
        
        logger.info(f"Búsqueda completada: {len(final_matches)} coincidencias únicas")
        
        return {
            "success": True, 
            "matches_found": len(final_matches), 
            "matches": final_matches,
            "total_faces_detected": len(faces_data),
            "threshold_used": threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en búsqueda de cara: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda: {str(e)}")

# ============================================
# EJECUCIÓN Y STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("Ejecutando startup event: Inicializando base de datos...")
    await database.initialize()
    logger.info("Aplicación lista para recibir peticiones.")

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition API v5.3.0-SQLite")
    print("=" * 60)
    print("Servidor: http://localhost:8888")
    print("Docs: http://localhost:8888/docs")
    print("=" * 60)
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8888,
        reload=False,
        log_level="info"
    )
