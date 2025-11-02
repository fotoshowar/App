"""
Sistema de Reconocimiento Facial de Alta Precisión
Implementa múltiples métodos de reconocimiento facial de última generación
"""

import cv2
import numpy as np
import face_recognition
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import dlib
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from typing import List, Dict, Tuple, Optional
import logging
import os
import sys
import urllib.request
import bz2
from pathlib import Path

# --- Configuración del logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFaceProcessor:
    """
    Procesador de caras avanzado que combina múltiples modelos de estado del arte:
    1. FaceNet (Google) - Precisión muy alta
    2. ArcFace/InsightFace - Estado del arte actual
    3. face_recognition (dlib) - Rápido y confiable
    4. MTCNN - Detección precisa de caras
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando dispositivo: {self.device}")
        
        # --- ¡CAMBIO CLAVE PARA PYINSTALLER! ---
        # Determinar la ruta base de la aplicación, tanto si es un script como un ejecutable.
        if getattr(sys, 'frozen', False):
            # Si estamos corriendo como un .exe de PyInstaller
            base_path = os.path.dirname(sys.executable)
        else:
            # Si estamos corriendo el script directamente (para desarrollo)
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Crear la ruta a la carpeta 'models' de forma segura
        self.models_dir = os.path.join(base_path, 'models')
        
        # Crear el directorio de modelos si no existe
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Directorio de modelos configurado en: {self.models_dir}")
        
        # Inicializar modelos
        self.init_models()

    def download_dlib_model(self, model_name: str, url: str) -> str:
        """Descarga un modelo de dlib si no existe."""
        # --- ¡CAMBIO CLAVE! ---
        # Usar la ruta absoluta que calculamos en __init__
        model_path = os.path.join(self.models_dir, model_name)
        
        if os.path.exists(model_path):
            logger.info(f"Modelo {model_name} ya existe, saltando descarga...")
            return str(model_path)
        
        logger.info(f"Descargando {model_name}...")
        compressed_path = str(model_path) + ".bz2"
        
        try:
            # Descargar archivo comprimido
            urllib.request.urlretrieve(url, compressed_path)
            
            # Descomprimir
            with bz2.BZ2File(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Eliminar archivo comprimido
            os.remove(compressed_path)
            logger.info(f"Modelo {model_name} descargado y descomprimido exitosamente")
            
        except Exception as e:
            logger.error(f"Error descargando {model_name}: {e}")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            raise
        
        return str(model_path)
        
    def init_models(self):
        """Inicializa todos los modelos de reconocimiento facial"""
        try:
            # 1. MTCNN para detección precisa de caras
            logger.info("Cargando MTCNN...")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            
            # 2. FaceNet (InceptionResnetV1) pre-entrenado
            logger.info("Cargando FaceNet...")
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            # 3. Transformaciones para FaceNet
            self.facenet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # 4. InsightFace (ArcFace)
            logger.info("Cargando InsightFace...")
            try:
                self.insightface_app = insightface.app.FaceAnalysis()
                self.insightface_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, 
                                           det_size=(640, 640))
                self.use_insightface = True
            except Exception as e:
                logger.warning(f"InsightFace no disponible: {e}")
                self.use_insightface = False
            
            # 5. dlib face recognition
            logger.info("Inicializando dlib...")
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                
                # Descargar modelos de dlib solo si no existen
                landmarks_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
                landmarks_path = self.download_dlib_model("shape_predictor_68_face_landmarks.dat", landmarks_url)
                self.shape_predictor = dlib.shape_predictor(landmarks_path)
                
                face_rec_url = "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2"
                face_rec_path = self.download_dlib_model("dlib_face_recognition_resnet_model_v1.dat", face_rec_url)
                self.face_encoder = dlib.face_recognition_model_v1(face_rec_path)
                
                self.use_dlib = True
                logger.info("dlib inicializado correctamente")
                
            except Exception as e:
                logger.error(f"Error inicializando dlib: {e}")
                self.use_dlib = False
            
            logger.info("Todos los modelos disponibles cargados exitosamente!")
            
        except Exception as e:
            logger.error(f"Error inicializando modelos: {e}")
            raise
    
    def detect_faces_mtcnn(self, image: np.ndarray) -> List[Dict]:
        """Detecta caras usando MTCNN (más preciso)"""
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar caras y puntos de referencia
        boxes, probs, landmarks = self.mtcnn.detect(rgb_image, landmarks=True)
        
        faces = []
        if boxes is not None:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob > 0.9:  # Filtro de confianza alto
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Asegurar que las coordenadas estén dentro de la imagen
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    face_region = image[y1:y2, x1:x2]
                    
                    if face_region.size > 0:
                        faces.append({
                            'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                            'confidence': float(prob),
                            'landmarks': landmark,
                            'face_image': face_region,
                            'method': 'mtcnn'
                        })
        
        return faces
    
    def generate_facenet_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Genera embedding usando FaceNet"""
        try:
            # Preparar imagen
            if len(face_image.shape) == 3:
                face_tensor = self.facenet_transform(face_image).unsqueeze(0).to(self.device)
            else:
                face_tensor = face_image.unsqueeze(0).to(self.device)
            
            # Generar embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
                
            # Normalizar
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error en FaceNet embedding: {e}")
            return np.zeros(512)
    
    def generate_insightface_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Genera embedding usando InsightFace (ArcFace)"""
        if not self.use_insightface:
            return None
            
        try:
            # InsightFace espera imagen completa, no solo la cara
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.insightface_app.get(rgb_image)
            
            if len(faces) > 0:
                # Tomar la cara con mayor confianza
                best_face = max(faces, key=lambda x: x.det_score)
                embedding = best_face.embedding
                
                # Normalizar
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error en InsightFace embedding: {e}")
            return None
    
    def generate_dlib_embedding(self, image: np.ndarray, face_bbox: Dict) -> Optional[np.ndarray]:
        """Genera embedding usando dlib - FIXED VERSION"""
        if not self.use_dlib:
            return None
            
        try:
            # Ensure we have RGB format for dlib
            if len(image.shape) == 3:
                # If BGR, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Create dlib rectangle
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
            rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Get landmarks using grayscale image
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            landmarks = self.shape_predictor(gray, rect)
            
            # Generate embedding using RGB image and landmarks
            face_descriptor = self.face_encoder.compute_face_descriptor(
                rgb_image, 
                landmarks
            )
            
            # Convert to numpy array
            embedding = np.array(face_descriptor)
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error en dlib embedding: {e}")
            return None
    
    def generate_face_recognition_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Genera embedding usando face_recognition"""
        try:
            # Convertir a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Obtener encodings
            encodings = face_recognition.face_encodings(rgb_image)
            
            if len(encodings) > 0:
                embedding = encodings[0]
                # Normalizar
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error en face_recognition embedding: {e}")
            return None
    
    def detect_faces_advanced(self, image: np.ndarray) -> List[Dict]:
        """Detección avanzada de caras combinando múltiples métodos"""
        all_faces = []
        
        # 1. MTCNN (más preciso)
        mtcnn_faces = self.detect_faces_mtcnn(image)
        all_faces.extend(mtcnn_faces)
        
        # 2. Face_recognition como backup
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            for (top, right, bottom, left) in face_locations:
                # Verificar si no es duplicado con MTCNN
                is_duplicate = False
                for existing_face in mtcnn_faces:
                    existing_bbox = existing_face['bbox']
                    overlap = self.calculate_bbox_overlap(
                        {'x': left, 'y': top, 'width': right-left, 'height': bottom-top},
                        existing_bbox
                    )
                    if overlap > 0.3:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    face_region = image[top:bottom, left:right]
                    all_faces.append({
                        'bbox': {'x': left, 'y': top, 'width': right-left, 'height': bottom-top},
                        'confidence': 0.8,
                        'face_image': face_region,
                        'method': 'face_recognition'
                    })
        except Exception as e:
            logger.error(f"Error en face_recognition detection: {e}")
        
        logger.info(f"Detectadas {len(all_faces)} caras usando métodos avanzados")
        return all_faces
    
    def generate_multi_model_embedding(self, face_data: Dict, original_image: np.ndarray) -> Dict:
        """Genera embeddings usando múltiples modelos para máxima precisión"""
        embeddings = {}
        face_image = face_data['face_image']
        face_bbox = face_data['bbox']
        
        # 1. FaceNet
        facenet_emb = self.generate_facenet_embedding(face_image)
        if facenet_emb is not None:
            embeddings['facenet'] = facenet_emb.tolist()
        
        # 2. InsightFace (usando imagen completa)
        insightface_emb = self.generate_insightface_embedding(original_image)
        if insightface_emb is not None:
            embeddings['insightface'] = insightface_emb.tolist()
        
        # 3. dlib
        dlib_emb = self.generate_dlib_embedding(original_image, face_bbox)
        if dlib_emb is not None:
            embeddings['dlib'] = dlib_emb.tolist()
        
        # 4. face_recognition
        face_rec_emb = self.generate_face_recognition_embedding(face_image)
        if face_rec_emb is not None:
            embeddings['face_recognition'] = face_rec_emb.tolist()
        
        return embeddings
    
    def compare_multi_embeddings(self, embeddings1: Dict, embeddings2: Dict) -> float:
        """Compara embeddings de múltiples modelos y retorna similitud promedio ponderada"""
        similarities = []
        weights = {
            'insightface': 0.4,    # Mejor modelo actual
            'facenet': 0.3,        # Muy bueno y estable
            'dlib': 0.2,           # Confiable
            'face_recognition': 0.1 # Backup
        }
        
        total_weight = 0
        weighted_similarity = 0
        
        for model_name in ['insightface', 'facenet', 'dlib', 'face_recognition']:
            if model_name in embeddings1 and model_name in embeddings2:
                emb1 = np.array(embeddings1[model_name])
                emb2 = np.array(embeddings2[model_name])
                
                # Calcular similitud coseno
                similarity = 1 - cosine(emb1, emb2)
                similarity = max(0, min(1, similarity))  # Clamp entre 0 y 1
                
                weight = weights[model_name]
                weighted_similarity += similarity * weight
                total_weight += weight
                
                logger.info(f"Similitud {model_name}: {similarity:.3f}")
        
        if total_weight > 0:
            final_similarity = weighted_similarity / total_weight
        else:
            final_similarity = 0.0
        
        logger.info(f"Similitud final ponderada: {final_similarity:.3f}")
        return final_similarity
    
    def calculate_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calcula el solapamiento entre dos bounding boxes"""
        x1, y1, w1, h1 = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
        x2, y2, w2, h2 = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
        
        # Coordenadas de intersección
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def detect_and_encode_faces(self, image: np.ndarray) -> List[Dict]:
        """Método principal: detecta caras y genera embeddings multi-modelo"""
        # Detectar caras
        detected_faces = self.detect_faces_advanced(image)
        
        # Generar embeddings para cada cara
        processed_faces = []
        for face_data in detected_faces:
            # Generar embeddings multi-modelo
            embeddings = self.generate_multi_model_embedding(face_data, image)
            
            if embeddings:  # Solo agregar si al menos un modelo funcionó
                face_data['embeddings'] = embeddings
                processed_faces.append(face_data)
        
        return processed_faces
