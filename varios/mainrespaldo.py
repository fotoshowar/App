from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response
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
import logging
import face_recognition
from scipy.spatial.distance import cosine
import traceback
import sys
import httpx
import io
import time
import asyncio
import re
from collections import deque

# --- NUEVOS IMPORTS PARA DESENCRIPTACIÓN ---
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hkdf

# --- CONFIGURACIÓN ---
BASE_URL = "https://athematic-carson-subtetanic.ngrok-free.dev" 

# JSON serializer personalizado para manejar NumPy types
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

app = FastAPI(title="Advanced Face Recognition API", version="3.0.0")

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

# Directorios
UPLOAD_DIR = Path("uploads")
FACES_DIR = Path("faces") 
DB_FILE = Path("face_database.json")
MODELS_DIR = Path("models")

for directory in [UPLOAD_DIR, FACES_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

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
            logger.info(f"   MAC del archivo: {mac_from_file.hex()}")
            logger.info(f"   MAC calculado v1: {calculated_mac_v1.hex()}")
            logger.info(f"   MAC calculado v2: {calculated_mac_v2.hex()}")
            logger.info(f"   MAC calculado v3: {calculated_mac_v3.hex()}")
        
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
# CLASES EXISTENTES (SIN CAMBIOS)
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
    
    def detect_and_encode_faces(self, image_path: str) -> List[Dict]:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"No se pudo cargar imagen: {image_path}")
                return []
            
            if self.processor and self.models_status.get('advanced_processor', False):
                logger.info("Usando AdvancedFaceProcessor")
                detected_faces = self.processor.detect_and_encode_faces(img)
                
                if detected_faces:
                    processed_faces = []
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
                            face_filename = f"{face_id}.jpg"
                            face_path = FACES_DIR / face_filename
                            cv2.imwrite(str(face_path), face_img)
                            
                            processed_face = {
                                'face_id': face_id,
                                'face_filename': face_filename,
                                'bbox': [x, y, w, h],
                                'confidence': float(face_data.get('confidence', 0.95)),
                                'embeddings': safe_convert_for_json(face_data.get('embeddings', {})),
                                'method': str(face_data.get('method', 'advanced_multi_model')),
                                'landmarks': safe_convert_for_json(face_data.get('landmarks', [])),
                                'face_image': face_img,
                                'models_used': list(face_data.get('embeddings', {}).keys()),
                                'processing_quality': 'advanced'
                            }
                            processed_faces.append(processed_face)
                    
                    logger.info(f"AdvancedFaceProcessor: {len(processed_faces)} caras detectadas")
                    return processed_faces
            
            return self.fallback_detection(img)
            
        except Exception as e:
            logger.error(f"Error en deteccion: {e}")
            return self.fallback_detection(cv2.imread(str(image_path)))
    
    def fallback_detection(self, img: np.ndarray) -> List[Dict]:
        if self.models_status.get('face_recognition', False):
            return self._detect_with_face_recognition(img)
        if self.models_status.get('opencv', False):
            return self._detect_with_opencv(img)
        return []
    
    def _detect_with_face_recognition(self, img: np.ndarray) -> List[Dict]:
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
    
    def _detect_with_opencv(self, img: np.ndarray) -> List[Dict]:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            faces_data = []
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_id = str(uuid.uuid4())
                face_filename = f"{face_id}.jpg"
                face_path = FACES_DIR / face_filename
                cv2.imwrite(str(face_path), face_img)
                
                face_data = {
                    'face_id': face_id,
                    'face_filename': face_filename,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.75,
                    'embeddings': {'opencv_basic': [float(i) for i in range(128)]},
                    'method': 'opencv_fallback',
                    'models_used': ['opencv'],
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

class FaceDatabase:
    def __init__(self):
        self.db_file = DB_FILE
        self.data = self.load_database()
    
    def load_database(self) -> Dict:
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data.setdefault('photos', {})
                data.setdefault('faces', {})
                logger.info(f"BD cargada: {len(data['photos'])} fotos, {len(data['faces'])} caras")
                return data
            except Exception as e:
                logger.error(f"Error cargando BD: {e}")
        return {'photos': {}, 'faces': {}}
    
    def save_database(self):
        try:
            if self.db_file.exists():
                backup_path = self.db_file.with_suffix('.json.backup')
                shutil.copy2(self.db_file, backup_path)
            
            safe_data = safe_convert_for_json(self.data)
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(safe_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        except Exception as e:
            logger.error(f"Error guardando BD: {e}")
            raise
    
    def add_photo(self, photo_id: str, filename: str, filepath: str, faces_data: List[Dict]):
        try:
            img = cv2.imread(filepath)
            image_height, image_width = img.shape[:2] if img is not None else (0, 0)
            file_size = Path(filepath).stat().st_size
            
            photo_data = {
                'id': photo_id,
                'filename': filename,
                'filepath': filepath,
                'upload_date': datetime.now().isoformat(),
                'faces_count': len(faces_data),
                'faces': [],
                'image_width': image_width,
                'image_height': image_height,
                'file_size': file_size
            }
            
            for face_data in faces_data:
                face_id = face_data['face_id']
                photo_data['faces'].append(face_id)
                
                face_data_copy = face_data.copy()
                if 'face_image' in face_data_copy:
                    del face_data_copy['face_image']
                
                face_data_copy.update({
                    'photo_id': photo_id,
                    'photo_filename': filename,
                    'face_name': '',
                    'face_notes': ''
                })
                
                self.data['faces'][face_id] = face_data_copy
            
            self.data['photos'][photo_id] = photo_data
            self.save_database()
            logger.info(f"Foto guardada: {filename}")
        except Exception as e:
            logger.error(f"Error agregando foto: {e}")
            raise
    
    def get_all_photos(self) -> List[Dict]:
        return list(self.data['photos'].values())
    
    def get_photo(self, photo_id: str) -> Optional[Dict]:
        return self.data['photos'].get(photo_id)
    
    def get_all_faces(self) -> List[Dict]:
        return list(self.data['faces'].values())
    
    def update_face_info(self, face_id: str, name: str, notes: str) -> bool:
        try:
            if face_id in self.data['faces']:
                self.data['faces'][face_id]['face_name'] = name.strip()
                self.data['faces'][face_id]['face_notes'] = notes.strip()
                self.save_database()
                return True
            return False
        except Exception as e:
            logger.error(f"Error actualizando cara: {e}")
            return False
    
    def get_stats(self) -> Dict:
        return {
            'total_photos': len(self.data['photos']),
            'total_faces': len(self.data['faces'])
        }
    
    def search_similar_faces(self, search_embeddings: Dict, threshold: float, processor) -> List[Dict]:
        all_faces = self.get_all_faces()
        matches = []
        
        for face_data in all_faces:
            try:
                similarity = processor.compare_embeddings(search_embeddings, face_data['embeddings'])
                if similarity >= threshold:
                    match_data = {
                        'face_id': face_data['face_id'],
                        'photo_id': face_data['photo_id'],
                        'photo_filename': face_data['photo_filename'],
                        'similarity': float(similarity),
                        'confidence': float(face_data.get('confidence', 0.0)),
                        'method': str(face_data.get('method', 'unknown'))
                    }
                    matches.append(match_data)
            except Exception as e:
                continue
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:50]

    def delete_photo(self, photo_id: str) -> Dict:
        try:
            if photo_id not in self.data['photos']:
                return {
                    'success': False, 
                    'error': 'Foto no encontrada',
                    'faces_deleted': 0
                }
            
            photo_data = self.data['photos'][photo_id]
            face_ids = photo_data.get('faces', [])
            
            faces_deleted = 0
            for face_id in face_ids:
                if face_id in self.data['faces']:
                    del self.data['faces'][face_id]
                    faces_deleted += 1
            
            del self.data['photos'][photo_id]
            self.save_database()
            
            logger.info(f"Foto {photo_id} eliminada: {faces_deleted} caras")
            
            return {
                'success': True,
                'photo_data': photo_data,
                'faces_deleted': faces_deleted
            }
            
        except Exception as e:
            logger.error(f"Error eliminando foto de BD {photo_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_deleted': 0
            }

# Instancias globales
processor = AdvancedFaceProcessorIntegration()
database = FaceDatabase()

# Variables globales para WhatsApp
whatsapp_client = httpx.AsyncClient()
WHATSAPP_API_TOKEN = "2175af4678f3b398b8dc9d2762e92772992984df03a59126f2f67f60bd094c00"
WHATSAPP_API_URL = "https://wasenderapi.com/api/send-message"
WHATSAPP_WEBHOOK_SECRET = "de9e287cdf4b088a5a228d1848c70997"

# Sistema de cola para mensajes
message_queue = deque()
processing_queue = False

async def send_whatsapp_message_with_retry(to_number, text, max_retries=3):
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
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
    headers = {
        "Authorization": f"Bearer {WHATSAPP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "to": to_number,
        "imageUrl": image_url,
        "text": caption
    }
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = 2 ** attempt
                logger.info(f"Reintentando enviar imagen (URL) en {delay} segundos")
                await asyncio.sleep(delay)
            
            response = await whatsapp_client.post(
                WHATSAPP_API_URL, 
                headers=headers, 
                json=payload,
                timeout=30.0
            )
            
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

# ENDPOINTS
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
    return {
        "message": "Face Recognition API v3.0",
        "status": "running",
        "system": processor.get_system_status(),
        "database": database.get_stats()
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
            
        database.add_photo(photo_id, file.filename or "unknown.jpg", str(filepath), faces_data)
        
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
        
        search_faces = processor.detect_and_encode_faces(str(temp_filepath))
        if not search_faces:
            return {"success": True, "matches_found": 0, "matches": []}
        
        search_face = max(search_faces, key=lambda x: x.get('confidence', 0))
        matches = database.search_similar_faces(search_face['embeddings'], threshold, processor)
        
        return {"success": True, "matches_found": len(matches), "matches": matches}
    except Exception as e:
        logger.error(f"Error search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_filepath and temp_filepath.exists():
            temp_filepath.unlink()

@app.get("/api/photos")
async def get_photos():
    return {"success": True, "photos": database.get_all_photos()}

@app.get("/api/photos/{photo_id}/faces")
async def get_photo_faces(photo_id: str):
    photo = database.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    all_faces = database.get_all_faces()
    photo_faces = [face for face in all_faces if face['photo_id'] == photo_id]
    
    formatted_faces = []
    for face in photo_faces:
        bbox = face.get('bbox', [0, 0, 0, 0])
        formatted_face = {
            'id': face['face_id'],
            'photo_id': face['photo_id'],
            'bounding_box': {
                'x': int(bbox[0]) if len(bbox) > 0 else 0,
                'y': int(bbox[1]) if len(bbox) > 1 else 0,
                'width': int(bbox[2]) if len(bbox) > 2 else 0,
                'height': int(bbox[3]) if len(bbox) > 3 else 0,
            },
            'confidence': float(face.get('confidence', 0.0)),
            'face_name': face.get('face_name', ''),
            'face_notes': face.get('face_notes', '')
        }
        formatted_faces.append(formatted_face)
    
    return {"success": True, "photo_id": photo_id, "faces_count": len(formatted_faces), "faces": formatted_faces}

@app.patch("/api/faces/{face_id}/info")
async def update_face_info(face_id: str, info: dict):
    name = info.get('name', '').strip()
    notes = info.get('notes', '').strip()
    
    success = database.update_face_info(face_id, name, notes)
    if not success:
        raise HTTPException(status_code=404, detail="Cara no encontrada")
    
    return {"success": True, "message": "Informacion actualizada", "face_id": face_id}

@app.get("/api/faces/search-by-name")
async def search_faces_by_name(name: str = Query(...)):
    name_search = name.strip().lower()
    all_faces = database.get_all_faces()
    
    matching_faces = []
    for face in all_faces:
        face_name = face.get('face_name', '').strip().lower()
        if name_search in face_name and face_name:
            photo = database.get_photo(face['photo_id'])
            formatted_face = {
                'face_id': face['face_id'],
                'photo_id': face['photo_id'],
                'photo_filename': photo['filename'] if photo else 'Desconocido',
                'face_name': face.get('face_name', ''),
                'confidence': float(face.get('confidence', 0.0))
            }
            matching_faces.append(formatted_face)
    
    return {"success": True, "name_searched": name, "faces_found": len(matching_faces), "faces": matching_faces}

@app.get("/api/people")
async def get_people_list():
    all_faces = database.get_all_faces()
    people_stats = {}
    
    for face in all_faces:
        face_name = face.get('face_name', '').strip()
        if not face_name:
            continue
        
        if face_name not in people_stats:
            people_stats[face_name] = {
                'name': face_name,
                'face_count': 0,
                'photo_ids': set(),
                'total_quality': 0.0
            }
        
        people_stats[face_name]['face_count'] += 1
        people_stats[face_name]['photo_ids'].add(face['photo_id'])
        people_stats[face_name]['total_quality'] += float(face.get('confidence', 0.0))
    
    people_list = []
    for person_data in people_stats.values():
        avg_quality = person_data['total_quality'] / person_data['face_count'] if person_data['face_count'] > 0 else 0.0
        people_list.append({
            'name': person_data['name'],
            'face_count': person_data['face_count'],
            'photo_count': len(person_data['photo_ids']),
            'avg_quality': avg_quality
        })
    
    people_list.sort(key=lambda x: x['face_count'], reverse=True)
    
    return {"success": True, "total_people": len(people_list), "people": people_list}

@app.api_route("/api/image/photo/{photo_id}", methods=["GET", "HEAD"])
async def get_photo_image(photo_id: str, request: Request):
    photo = database.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    filepath = Path(photo['filepath'])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    if request.method == "HEAD":
        return Response(
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(filepath.stat().st_size)
            }
        )
    
    return FileResponse(path=filepath, media_type="image/jpeg")

@app.api_route("/api/image/face/{face_id}", methods=["GET", "HEAD"])
async def get_face_image(face_id: str, request: Request):
    all_faces = database.get_all_faces()
    face_data = next((f for f in all_faces if f['face_id'] == face_id), None)
    
    if not face_data:
        raise HTTPException(status_code=404, detail="Cara no encontrada")
    
    face_filepath = FACES_DIR / face_data['face_filename']
    if not face_filepath.exists():
        raise HTTPException(status_code=404, detail="Archivo de cara no encontrado")
    
    if request.method == "HEAD":
        return Response(
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(face_filepath.stat().st_size)
            }
        )

    return FileResponse(path=face_filepath, media_type="image/jpeg")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "3.0.0"}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    try:
        photo = database.get_photo(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Foto no encontrada")
        
        all_faces = database.get_all_faces()
        photo_faces = [face for face in all_faces if face['photo_id'] == photo_id]
        
        for face in photo_faces:
            try:
                face_filepath = FACES_DIR / face['face_filename']
                if face_filepath.exists():
                    face_filepath.unlink()
            except Exception as e:
                logger.warning(f"No se pudo eliminar cara: {e}")
        
        try:
            photo_filepath = Path(photo['filepath'])
            if photo_filepath.exists():
                photo_filepath.unlink()
        except Exception as e:
            logger.warning(f"No se pudo eliminar foto: {e}")
        
        for face in photo_faces:
            if face['face_id'] in database.data['faces']:
                del database.data['faces'][face['face_id']]
        
        if photo_id in database.data['photos']:
            del database.data['photos'][photo_id]
        
        database.save_database()
        
        return {"success": True, "message": "Foto eliminada correctamente", "photo_id": photo_id, "faces_deleted": len(photo_faces)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando foto: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/static", StaticFiles(directory="."), name="static")

# ============================================
# WEBHOOK DE WHATSAPP (RESPUESTA DIRECTA)
# ============================================

@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """Webhook con soporte para medios encriptados de WhatsApp y envío de la mejor imagen."""
    try:
        request_body = await request.body()
        data = json.loads(request_body.decode('utf-8'))
        logger.info(f"📥 Webhook recibido de WhatsApp")

        messages = data.get("data", {}).get("messages", {})
        message_data = messages
        
        user_number_full = message_data.get("key", {}).get("remoteJid", "")
        user_number = user_number_full.replace("@s.whatsapp.net", "").replace("+", "")
        
        if not user_number:
            logger.warning("⚠️ Mensaje sin número de remitente")
            return {"status": "ignored"}

        message_content = message_data.get("message", {})
        
        # MANEJAR MENSAJES CON IMAGEN
        if "imageMessage" in message_content:
            logger.info(f"📷 Procesando imagen de {user_number}...")
            
            image_message = message_content.get("imageMessage", {})
            caption = image_message.get("caption", "").strip().lower()
            image_url = image_message.get("url")
            media_key = image_message.get("mediaKey")
            file_enc_sha256 = image_message.get("fileEncSha256")
            
            if not image_url or not media_key or not file_enc_sha256:
                message_queue.append(("text", f"+{user_number}", "❌ No pude obtener la imagen o sus claves."))
                asyncio.create_task(process_message_queue())
                return {"status": "ok"}

            # Descargar y desencriptar
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
                        message_queue.append(("text", f"+{user_number}", "❌ Error procesando imagen."))
                        asyncio.create_task(process_message_queue())
                        return {"status": "ok"}

            except Exception as e:
                logger.error(f"Error descargando/desencriptando: {e}")
                message_queue.append(("text", f"+{user_number}", "❌ Error con tu imagen. Intenta de nuevo."))
                asyncio.create_task(process_message_queue())
                return {"status": "ok"}

            # COMANDO 1: SUBIR FOTO
            if caption == "1":
                logger.info(f"📤 Subiendo foto de {user_number}...")
                try:
                    files = {'file': (f'whatsapp_{user_number}.jpg', image_bytes, 'image/jpeg')}
                    internal_response = await whatsapp_client.post("http://127.0.0.1:8000/api/upload-photo", files=files)
                    
                    if internal_response.status_code == 200:
                        result = internal_response.json()
                        
                        if result.get('success', False):
                            faces_detected = result.get('faces_detected', 0)
                            message_queue.append(("text", f"+{user_number}", f"✅ ¡Foto agregada!\n\n*{faces_detected} cara(s)* detectadas."))
                        else:
                            message_queue.append(("text", f"+{user_number}", f"❌ No pude procesar tu foto."))
                    else:
                        message_queue.append(("text", f"+{user_number}", "❌ Error en el servidor."))
                    
                    asyncio.create_task(process_message_queue())
                except Exception as e:
                    logger.error(f"Error interno: {e}")
                    message_queue.append(("text", f"+{user_number}", "❌ Error de conexión."))
                    asyncio.create_task(process_message_queue())

            # COMANDO 2: BUSCAR CARAS (CAMBIO: RESPUESTA DIRECTA)
            elif caption == "2":
                logger.info(f"🔍 Buscando cara de {user_number}...")
                try:
                    temp_filename = f"temp_search_{uuid.uuid4()}.jpg"
                    temp_filepath = UPLOAD_DIR / temp_filename
                    
                    with open(temp_filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    search_faces = processor.detect_and_encode_faces(str(temp_filepath))
                    
                    if not search_faces:
                        message_queue.append(("text", f"+{user_number}", "❌ No detecté una cara clara."))
                    else:
                        search_face = max(search_faces, key=lambda x: x.get('confidence', 0))
                        matches = database.search_similar_faces(search_face['embeddings'], 0.5, processor)
                        
                        if matches:
                            best_match = matches[0]
                            photo_id = best_match['photo_id']
                            similarity_percent = round(best_match['similarity'] * 100)
                            
                            public_image_url = f"{BASE_URL}/api/image/photo/{photo_id}"
                            logger.info(f"🔗 Añadida a la cola la mejor coincidencia: {public_image_url}")
                            
                            # --- CAMBIO: SE ELIMINA EL MENSAJE INTERMEDIO ---
                            # Se añade directamente la imagen a la cola.
                            message_queue.append(("image", f"+{user_number}", (public_image_url, f"Similitud: {similarity_percent}%")))
                            
                        else:
                            message_queue.append(("text", f"+{user_number}", "❌ No encontré caras similares."))
                    
                    if temp_filepath.exists():
                        temp_filepath.unlink()
                    
                    asyncio.create_task(process_message_queue())
                        
                except Exception as e:
                    logger.error(f"Error búsqueda: {e}")
                    message_queue.append(("text", f"+{user_number}", "❌ Error inesperado."))
                    asyncio.create_task(process_message_queue())
            
            # SIN COMANDO
            else:
                message_queue.append(("text", f"+{user_number}", "Envía la foto con:\n**1** = Subir a galería\n**2** = Buscar la mejor coincidencia"))
                asyncio.create_task(process_message_queue())
            
            return {"status": "ok"}

        # MANEJAR MENSAJES DE TEXTO
        user_message = ""
        if "conversation" in message_content:
            user_message = message_content.get("conversation", "").strip().lower()
        elif "extendedTextMessage" in message_content:
            user_message = message_content.get("extendedTextMessage", {}).get("text", "").strip().lower()

        if user_message:
            logger.info(f"💬 Mensaje de {user_number}: '{user_message}'")
            message_queue.append(("text", f"+{user_number}", "¡Hola! Para usar el bot:\n\n📷 **Subir foto:** Envía imagen con **1**\n🔍 **Buscar cara:** Envía imagen con **2**"))
            asyncio.create_task(process_message_queue())
            return {"status": "ok"}

    except Exception as e:
        logger.error(f"❌ Error webhook: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "detail": str(e)}

    return {"status": "ok"}

if __name__ == "__main__":
    print("=" * 60)
    print("Face Recognition API v3.0 - Respuesta Directa")
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
