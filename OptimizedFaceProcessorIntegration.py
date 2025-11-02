"""
Sistema de Reconocimiento Facial Optimizado
Implementa detección y reconocimiento facial eficiente con face_recognition
"""

import cv2
import numpy as np
import face_recognition
import dlib
from typing import List, Dict, Tuple, Optional
import logging
import os
import urllib.request
import bz2
from pathlib import Path
import hashlib
from scipy.spatial.distance import cosine

class OptimizedFaceProcessor:
    """
    Procesador de caras optimizado que utiliza solo face_recognition para máxima eficiencia
    """
    
    def __init__(self):
        self.device = 'cpu'  # Simplificado - solo usamos CPU para face_recognition
        print(f"Usando dispositivo: {self.device}")
        
        # Crear directorio para modelos
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Inicializar modelos
        self.init_models()
        
        # Parámetros de deduplicación
        self.face_overlap_threshold = 0.3  # Umbral para considerar caras duplicadas
        self.min_face_size = 50  # Tamaño mínimo de cara en píxeles
        
    def download_dlib_model(self, model_name: str, url: str) -> str:
        """Descarga un modelo de dlib si no existe"""
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            print(f"Modelo {model_name} ya existe, saltando descarga...")
            return str(model_path)
        
        print(f"Descargando {model_name}...")
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
            print(f"Modelo {model_name} descargado y descomprimido exitosamente")
            
        except Exception as e:
            print(f"Error descargando {model_name}: {e}")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            raise
        
        return str(model_path)
        
    def init_models(self):
        """Inicializa los modelos necesarios"""
        try:
            print("Inicializando face_recognition...")
            
            # Verificar que face_recognition funciona
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_img)
            
            self.use_face_recognition = True
            print("face_recognition inicializado correctamente")
            
        except Exception as e:
            print(f"Error inicializando face_recognition: {e}")
            self.use_face_recognition = False
    
    def validate_face_quality(self, face_image: np.ndarray) -> Dict:
        """Valida la calidad de una cara detectada"""
        quality_score = 0
        issues = []
        
        # Verificar tamaño
        height, width = face_image.shape[:2]
        if min(height, width) < self.min_face_size:
            issues.append(f"Cara demasiado pequeña ({min(height, width)}px < {self.min_face_size}px)")
        else:
            quality_score += 0.3
        
        # Verificar desenfoque (usando Laplacian variance)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            issues.append(f"Cara borrosa (blur score: {blur_score:.2f})")
        else:
            quality_score += 0.3
        
        # Verificar iluminación
        brightness = np.mean(gray)
        if brightness < 50 or brightness > 200:
            issues.append(f"Iluminación pobre (brillo: {brightness:.2f})")
        else:
            quality_score += 0.2
        
        # Verificar contraste
        contrast = np.std(gray)
        if contrast < 30:
            issues.append(f"Bajo contraste (std: {contrast:.2f})")
        else:
            quality_score += 0.2
        
        return {
            'valid': quality_score > 0.6,
            'score': quality_score,
            'issues': issues,
            'metrics': {
                'size': min(height, width),
                'blur': blur_score,
                'brightness': brightness,
                'contrast': contrast
            }
        }
    
    def calculate_face_hash(self, face_image: np.ndarray) -> str:
        """Calcula un hash único para una cara para detectar duplicados"""
        # Redimensionar a tamaño pequeño para hash consistente
        small_face = cv2.resize(face_image, (32, 32))
        gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
        
        # Calcular hash perceptual
        hash_bytes = hashlib.md5(gray.tobytes()).digest()
        return hash_bytes.hex()
    
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
    
    def detect_faces_optimized(self, image: np.ndarray) -> List[Dict]:
        """Detecta caras usando face_recognition con deduplicación"""
        if not self.use_face_recognition:
            return []
        
        try:
            # Convertir BGR a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar caras con diferentes modelos para mayor robustez
            face_locations = []
            
            # Intentar con modelo CNN primero (más preciso)
            try:
                face_locations = face_recognition.face_locations(rgb_image, model="cnn")
            except:
                # Si falla, usar HOG (más rápido)
                face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return []
            
            # Generar embeddings para todas las caras detectadas
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            detected_faces = []
            face_hashes = set()  # Para detectar duplicados por hash
            
            for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                top, right, bottom, left = location
                
                # Extraer cara
                face_img = image[top:bottom, left:right]
                
                # Validar calidad de la cara
                quality = self.validate_face_quality(face_img)
                if not quality['valid']:
                    print(f"Cara {i} descartada por baja calidad: {quality['issues']}")
                    continue
                
                # Calcular hash para detectar duplicados
                face_hash = self.calculate_face_hash(face_img)
                if face_hash in face_hashes:
                    print(f"Cara {i} descartada por ser duplicada (hash: {face_hash[:8]}...)")
                    continue
                
                face_hashes.add(face_hash)
                
                # Verificar solapamiento con caras ya detectadas
                is_duplicate = False
                new_bbox = {'x': left, 'y': top, 'width': right-left, 'height': bottom-top}
                
                for existing_face in detected_faces:
                    overlap = self.calculate_bbox_overlap(new_bbox, existing_face['bbox'])
                    if overlap > self.face_overlap_threshold:
                        print(f"Cara {i} descartada por solapamiento ({overlap:.2f} > {self.face_overlap_threshold})")
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                # Crear ID único para la cara
                face_id = f"face_{hashlib.md5(f'{face_hash}_{i}'.encode()).hexdigest()[:8]}"
                
                face_data = {
                    'face_id': face_id,
                    'bbox': new_bbox,
                    'confidence': 0.95,  # face_recognition no da confianza, usamos un valor alto
                    'face_image': face_img,
                    'method': 'face_recognition_optimized',
                    'quality_score': quality['score'],
                    'quality_metrics': quality['metrics'],
                    'face_hash': face_hash,
                    'embeddings': {
                        'face_recognition': encoding.tolist()
                    }
                }
                
                detected_faces.append(face_data)
            
            print(f"Detectadas {len(detected_faces)} caras únicas y válidas")
            return detected_faces
            
        except Exception as e:
            print(f"Error en detección de caras: {e}")
            return []
    
    def compare_faces(self, face1_encoding: np.ndarray, face2_encoding: np.ndarray, tolerance: float = 0.6) -> bool:
        """Compara dos caras usando face_recognition"""
        try:
            distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
            return distance < tolerance
        except Exception as e:
            print(f"Error comparando caras: {e}")
            return False
    
    def calculate_similarity(self, face1_encoding: np.ndarray, face2_encoding: np.ndarray) -> float:
        """Calcula similitud entre dos caras (0-1)"""
        try:
            distance = face_recognition.face_distance([face1_encoding], face2_encoding)[0]
            # Convertir distancia a similitud (0-1)
            similarity = max(0, 1 - distance)
            return similarity
        except Exception as e:
            print(f"Error calculando similitud: {e}")
            return 0.0
    
    def detect_and_encode_faces(self, image: np.ndarray, save_faces: bool = True) -> List[Dict]:
        """Método principal: detecta caras y genera embeddings optimizados"""
        # Detectar caras con deduplicación
        detected_faces = self.detect_faces_optimized(image)
        
        # Procesar cada cara detectada
        processed_faces = []
        for face_data in detected_faces:
            # Ya tenemos el embedding de face_recognition
            if face_data['embeddings'] and 'face_recognition' in face_data['embeddings']:
                processed_faces.append(face_data)
        
        return processed_faces
