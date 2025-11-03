# face_detector.py
import logging
import uuid
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Importamos el procesador avanzado si está disponible
try:
    from advanced_face_processor import AdvancedFaceProcessor
    ADVANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSOR_AVAILABLE = False

# Usa las mismas rutas que en el archivo principal
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    APPLICATION_PATH = Path(sys.executable).parent
else:
    APPLICATION_PATH = Path(__file__).parent

FACES_DIR = APPLICATION_PATH / "faces"
FACES_DIR.mkdir(exist_ok=True)

# Configuración del logger
logger = logging.getLogger(__name__)

class AdvancedFaceProcessorIntegration:
    def __init__(self):
        logger.info("Inicializando Face Processor...")
        self.processor = None
        self.models_status = {}
        self.init_advanced_processor()
    
    def init_advanced_processor(self):
        if ADVANCED_PROCESSOR_AVAILABLE:
            try:
                logger.info("Intentando cargar AdvancedFaceProcessor...")
                device = 'cuda' if self._check_gpu_availability() else 'cpu'
                self.processor = AdvancedFaceProcessor(device=device)
                self.models_status['advanced_processor'] = True
                logger.info("AdvancedFaceProcessor cargado exitosamente")
                return
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
                            
                            # Forzar la generación de un embedding face_recognition (128-dim)
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
                
                # Generar embedding con face_recognition si está disponible
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

# --- Instancia global del procesador que usaremos en main ---
processor = AdvancedFaceProcessorIntegration()

# --- Función pública y sencilla para que la llame main ---
def detect_faces_in_image(image_path: str) -> List[Dict]:
    """
    Función principal que expone el módulo.
    Detecta caras en una imagen y devuelve una lista de datos.
    """
    logger.info(f"Solicitando detección de caras para: {image_path}")
    return processor.detect_and_encode_faces(image_path)
