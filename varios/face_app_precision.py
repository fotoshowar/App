#!/usr/bin/env python3
"""
Aplicación de reconocimiento facial con precisión mejorada
Usa múltiples técnicas para aumentar precisión sin librerías pesadas
"""

import sys
import os
import json
import sqlite3
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QProgressBar, QTextEdit,
        QFileDialog, QScrollArea, QGridLayout, QSlider, 
        QGroupBox, QFrame, QMessageBox, QStatusBar, QCheckBox
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False

class EnhancedFaceProcessor:
    """Procesador mejorado con múltiples técnicas de precisión"""
    
    def __init__(self):
        print("Inicializando procesador de precisión mejorada...")
        self.face_cascade = self.load_opencv_cascade()
        self.quality_threshold = 0.3
        
    def load_opencv_cascade(self):
        """Cargar clasificador de OpenCV como respaldo"""
        try:
            cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    return cv2.CascadeClassifier(path)
            return None
        except:
            return None
    
    def enhance_image_quality(self, image):
        """Mejorar calidad de imagen para mejor detección"""
        # Convertir a escala de grises para procesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtros para mejorar calidad
        # 1. Ecualización de histograma adaptativa
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        
        # 2. Reducción de ruido
        denoised = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
        
        # 3. Sharpening suave
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convertir de vuelta a color
        enhanced_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return enhanced_color, sharpened
    
    def calculate_face_quality(self, face_image, bbox):
        """Calcular score de calidad de la cara detectada"""
        if face_image.size == 0:
            return 0.0
        
        score = 0.0
        
        # 1. Tamaño de la cara (caras más grandes = mejor)
        area = bbox['width'] * bbox['height']
        size_score = min(1.0, area / 10000)  # Normalizar a 100x100 píxeles
        score += size_score * 0.3
        
        # 2. Nitidez (detectar blur)
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)
        score += sharpness_score * 0.25
        
        # 3. Contraste
        contrast = gray_face.std()
        contrast_score = min(1.0, contrast / 50)
        score += contrast_score * 0.2
        
        # 4. Relación de aspecto (caras más cuadradas = mejor)
        aspect_ratio = bbox['width'] / max(bbox['height'], 1)
        aspect_score = 1.0 - abs(1.0 - aspect_ratio)
        score += aspect_score * 0.15
        
        # 5. Posición central (caras centradas suelen ser mejores)
        center_score = 1.0  # Simplificado, asumimos buena posición
        score += center_score * 0.1
        
        return min(1.0, score)
    
    def detect_faces_multiple_methods(self, image):
        """Detectar caras usando múltiples métodos y combinar resultados"""
        all_faces = []
        
        # Mejorar imagen primero
        enhanced_image, enhanced_gray = self.enhance_image_quality(image)
        
        # Método 1: face_recognition con imagen mejorada (más preciso)
        if FACE_REC_AVAILABLE:
            faces_fr = self.detect_with_face_recognition(enhanced_image)
            all_faces.extend(faces_fr)
        
        # Método 2: face_recognition con imagen original (respaldo)
        if FACE_REC_AVAILABLE and len(all_faces) == 0:
            faces_fr_orig = self.detect_with_face_recognition(image)
            all_faces.extend(faces_fr_orig)
        
        # Método 3: OpenCV Haar Cascades (respaldo adicional)
        if self.face_cascade and len(all_faces) == 0:
            faces_opencv = self.detect_with_opencv(enhanced_gray, image)
            all_faces.extend(faces_opencv)
        
        # Filtrar por calidad
        quality_faces = []
        for face in all_faces:
            quality = self.calculate_face_quality(face['face_image'], face['bbox'])
            if quality >= self.quality_threshold:
                face['quality_score'] = quality
                quality_faces.append(face)
        
        # Remover duplicados (caras muy cercanas)
        unique_faces = self.remove_duplicate_faces(quality_faces)
        
        # Generar múltiples encodings por cara
        enhanced_faces = []
        for face in unique_faces:
            enhanced_face = self.generate_multiple_encodings(face, image)
            if enhanced_face:
                enhanced_faces.append(enhanced_face)
        
        return enhanced_faces
    
    def detect_with_face_recognition(self, image):
        """Detección usando face_recognition"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Probar con diferentes modelos
            faces = []
            
            # Modelo HOG (más rápido)
            locations_hog = face_recognition.face_locations(rgb_image, model="hog")
            if locations_hog:
                encodings_hog = face_recognition.face_encodings(rgb_image, locations_hog)
                for (top, right, bottom, left), encoding in zip(locations_hog, encodings_hog):
                    face_image = image[top:bottom, left:right]
                    faces.append({
                        'bbox': {'x': left, 'y': top, 'width': right-left, 'height': bottom-top},
                        'confidence': 0.85,
                        'face_image': face_image,
                        'method': 'face_recognition_hog',
                        'encoding': encoding
                    })
            
            # Si no encuentra caras con HOG, probar CNN (más preciso pero lento)
            if not faces:
                try:
                    locations_cnn = face_recognition.face_locations(rgb_image, model="cnn")
                    if locations_cnn:
                        encodings_cnn = face_recognition.face_encodings(rgb_image, locations_cnn)
                        for (top, right, bottom, left), encoding in zip(locations_cnn, encodings_cnn):
                            face_image = image[top:bottom, left:right]
                            faces.append({
                                'bbox': {'x': left, 'y': top, 'width': right-left, 'height': bottom-top},
                                'confidence': 0.95,
                                'face_image': face_image,
                                'method': 'face_recognition_cnn',
                                'encoding': encoding
                            })
                except:
                    pass  # CNN puede fallar en algunos sistemas
            
            return faces
            
        except Exception as e:
            print(f"Error en face_recognition: {e}")
            return []
    
    def detect_with_opencv(self, gray_image, color_image):
        """Detección usando OpenCV como respaldo"""
        try:
            faces_opencv = self.face_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces = []
            for (x, y, w, h) in faces_opencv:
                face_image = color_image[y:y+h, x:x+w]
                # Crear encoding básico usando OpenCV
                face_descriptor = self.create_opencv_descriptor(face_image)
                
                faces.append({
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': 0.7,
                    'face_image': face_image,
                    'method': 'opencv_haar',
                    'encoding': face_descriptor
                })
            
            return faces
            
        except Exception as e:
            print(f"Error en OpenCV: {e}")
            return []
    
    def create_opencv_descriptor(self, face_image):
        """Crear descriptor usando OpenCV para respaldo"""
        try:
            # Redimensionar a tamaño fijo
            resized = cv2.resize(face_image, (128, 128))
            
            # Crear múltiples descriptores
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # 1. Histograma LBP (Local Binary Pattern)
            lbp = self.calculate_lbp(gray)
            
            # 2. Histograma de gradientes
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # 3. Histogramas de color
            hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
            
            # Combinar todos los descriptores
            descriptor = np.concatenate([
                lbp.flatten(),
                magnitude.flatten()[::4],  # Submuestrear para reducir tamaño
                hist_b.flatten(),
                hist_g.flatten(),
                hist_r.flatten()
            ])
            
            # Normalizar
            descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-8)
            
            return descriptor.tolist()
            
        except Exception as e:
            print(f"Error creando descriptor OpenCV: {e}")
            return []
    
    def calculate_lbp(self, gray_image):
        """Calcular Local Binary Pattern"""
        try:
            height, width = gray_image.shape
            lbp = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = gray_image[i, j]
                    binary_string = ''
                    
                    # 8 vecinos
                    neighbors = [
                        gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                        gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                        gray_image[i+1, j-1], gray_image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor > center else '0'
                    
                    lbp[i, j] = int(binary_string, 2)
            
            # Calcular histograma
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            return hist
            
        except:
            return np.zeros(256)
    
    def remove_duplicate_faces(self, faces):
        """Remover caras duplicadas basado en solapamiento"""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        for face in faces:
            is_duplicate = False
            for unique_face in unique_faces:
                overlap = self.calculate_overlap(face['bbox'], unique_face['bbox'])
                if overlap > 0.3:  # 30% de solapamiento = duplicado
                    # Mantener la de mayor calidad/confianza
                    if face.get('quality_score', 0) > unique_face.get('quality_score', 0):
                        unique_faces.remove(unique_face)
                        unique_faces.append(face)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def calculate_overlap(self, bbox1, bbox2):
        """Calcular solapamiento entre dos bounding boxes"""
        x1, y1, w1, h1 = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
        x2, y2, w2, h2 = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
        
        # Intersección
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_multiple_encodings(self, face, original_image):
        """Generar múltiples encodings de una cara para mayor robustez"""
        encodings = {}
        
        # Encoding principal
        if 'encoding' in face:
            if face['method'].startswith('face_recognition'):
                encodings['face_recognition'] = face['encoding'].tolist()
            else:
                encodings['opencv_descriptor'] = face['encoding']
        
        # Encoding con transformaciones para robustez
        if FACE_REC_AVAILABLE and 'face_recognition' in encodings:
            try:
                face_image = face['face_image']
                
                # 1. Versión con mejor contraste
                enhanced = cv2.convertScaleAbs(face_image, alpha=1.2, beta=10)
                rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb_enhanced)
                if locations:
                    enhanced_encodings = face_recognition.face_encodings(rgb_enhanced, locations)
                    if enhanced_encodings:
                        encodings['face_recognition_enhanced'] = enhanced_encodings[0].tolist()
                
                # 2. Versión con diferente resolución
                if face_image.shape[0] > 100 and face_image.shape[1] > 100:
                    resized = cv2.resize(face_image, (150, 150))
                    rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    locations_resized = face_recognition.face_locations(rgb_resized)
                    if locations_resized:
                        resized_encodings = face_recognition.face_encodings(rgb_resized, locations_resized)
                        if resized_encodings:
                            encodings['face_recognition_resized'] = resized_encodings[0].tolist()
            
            except Exception as e:
                print(f"Error generando encodings adicionales: {e}")
        
        # Actualizar face con múltiples encodings
        face['embeddings'] = encodings
        return face
    
    def compare_multiple_embeddings(self, embeddings1, embeddings2):
        """Comparar múltiples embeddings y retornar mejor coincidencia"""
        max_similarity = 0.0
        comparisons = []
        
        # Comparar face_recognition encodings
        fr_types = ['face_recognition', 'face_recognition_enhanced', 'face_recognition_resized']
        for type1 in fr_types:
            if type1 in embeddings1:
                for type2 in fr_types:
                    if type2 in embeddings2:
                        try:
                            emb1 = np.array(embeddings1[type1])
                            emb2 = np.array(embeddings2[type2])
                            distance = np.linalg.norm(emb1 - emb2)
                            similarity = max(0, 1.0 - distance / 2.0)  # Normalizar
                            comparisons.append(similarity)
                            max_similarity = max(max_similarity, similarity)
                        except:
                            pass
        
        # Comparar descriptores OpenCV
        if 'opencv_descriptor' in embeddings1 and 'opencv_descriptor' in embeddings2:
            try:
                desc1 = np.array(embeddings1['opencv_descriptor'])
                desc2 = np.array(embeddings2['opencv_descriptor'])
                
                # Similitud coseno
                cosine_sim = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2) + 1e-8)
                cosine_sim = max(0, cosine_sim)
                
                comparisons.append(cosine_sim)
                max_similarity = max(max_similarity, cosine_sim)
                
            except:
                pass
        
        # Retornar promedio ponderado de las mejores comparaciones
        if comparisons:
            # Tomar las 3 mejores comparaciones
            top_comparisons = sorted(comparisons, reverse=True)[:3]
            weighted_similarity = np.average(top_comparisons, weights=range(len(top_comparisons), 0, -1))
            return weighted_similarity
        
        return max_similarity

class EnhancedFaceApp(QMainWindow):
    """Aplicación con precisión mejorada"""
    
    def __init__(self):
        super().__init__()
        self.processor = EnhancedFaceProcessor()
        self.init_ui()
        self.setup_database()
        
    def init_ui(self):
        self.setWindowTitle("Face Recognition - Precisión Mejorada")
        self.setGeometry(100, 100, 1200, 800)
        
        # Estilo mejorado
        self.setStyleSheet("""
            QMainWindow { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #34495e);
                color: white; 
            }
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white; border: none; 
                padding: 12px; border-radius: 6px; 
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }
            QTabWidget::pane { 
                border: 2px solid #3498db; 
                border-radius: 8px;
                background: #2c3e50;
            }
            QTabBar::tab { 
                background: #34495e; color: white; 
                padding: 10px 20px; margin: 2px;
                border-radius: 6px;
            }
            QTabBar::tab:selected { 
                background: #3498db; 
            }
            QTextEdit { 
                background: #1e1e1e; color: #ecf0f1; 
                border: 2px solid #34495e; border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
            }
            QProgressBar { 
                border: 2px solid #34495e; 
                border-radius: 6px;
                background: #2c3e50;
            }
            QProgressBar::chunk { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 8px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header mejorado
        header = QLabel("Face Recognition - Precisión Mejorada")
        header.setStyleSheet("""
            font-size: 28px; font-weight: bold; 
            padding: 25px; color: #ecf0f1;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3498db, stop:1 #2980b9);
            border-radius: 10px; margin: 10px;
        """)
        layout.addWidget(header)
        
        # Tabs con configuración
        self.tab_widget = QTabWidget()
        
        # Tab de configuración
        config_tab = QWidget()
        config_layout = QVBoxLayout()
        
        quality_group = QGroupBox("Configuración de Calidad")
        quality_layout = QVBoxLayout()
        
        self.quality_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_threshold_slider.setRange(10, 80)
        self.quality_threshold_slider.setValue(30)
        self.quality_threshold_slider.valueChanged.connect(self.update_quality_threshold)
        
        self.quality_label = QLabel("Umbral mínimo de calidad: 30%")
        
        self.use_cnn_checkbox = QCheckBox("Usar CNN (más lento, más preciso)")
        self.use_enhancement_checkbox = QCheckBox("Mejorar calidad de imagen")
        self.use_enhancement_checkbox.setChecked(True)
        
        quality_layout.addWidget(self.quality_label)
        quality_layout.addWidget(self.quality_threshold_slider)
        quality_layout.addWidget(self.use_cnn_checkbox)
        quality_layout.addWidget(self.use_enhancement_checkbox)
        quality_group.setLayout(quality_layout)
        
        config_layout.addWidget(quality_group)
        config_layout.addStretch()
        config_tab.setLayout(config_layout)
        
        # Tab de carga mejorado
        upload_tab = QWidget()
        upload_layout = QVBoxLayout()
        
        upload_controls = QHBoxLayout()
        self.upload_btn = QPushButton("Seleccionar Fotos para Analizar")
        self.upload_btn.clicked.connect(self.select_photos)
        
        self.batch_process_btn = QPushButton("Procesar Lote")
        self.batch_process_btn.setEnabled(False)
        
        upload_controls.addWidget(self.upload_btn)
        upload_controls.addWidget(self.batch_process_btn)
        
        self.upload_progress = QProgressBar()
        self.upload_progress.hide()
        
        self.upload_log = QTextEdit()
        self.upload_log.setPlaceholderText("Los resultados del procesamiento aparecerán aquí...")
        
        upload_layout.addLayout(upload_controls)
        upload_layout.addWidget(self.upload_progress)
        upload_layout.addWidget(QLabel("Log de Procesamiento Detallado:"))
        upload_layout.addWidget(self.upload_log)
        upload_tab.setLayout(upload_layout)
        
        # Tab de búsqueda mejorado
        search_tab = QWidget()
        search_layout = QVBoxLayout()
        
        search_controls = QHBoxLayout()
        search_btn = QPushButton("Buscar Cara Similar")
        search_btn.clicked.connect(self.search_face)
        
        self.analyze_btn = QPushButton("Análisis Detallado")
        self.analyze_btn.clicked.connect(self.analyze_face_detailed)
        
        search_controls.addWidget(search_btn)
        search_controls.addWidget(self.analyze_btn)
        
        # Configuración de búsqueda
        search_config = QGroupBox("Configuración de Búsqueda")
        search_config_layout = QVBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        self.threshold_label = QLabel("Umbral de similitud: 70%")
        
        self.max_results_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_results_slider.setRange(5, 50)
        self.max_results_slider.setValue(10)
        self.max_results_slider.valueChanged.connect(self.update_max_results_label)
        
        self.max_results_label = QLabel("Máximo resultados: 10")
        
        search_config_layout.addWidget(self.threshold_label)
        search_config_layout.addWidget(self.threshold_slider)
        search_config_layout.addWidget(self.max_results_label)
        search_config_layout.addWidget(self.max_results_slider)
        search_config.setLayout(search_config_layout)
        
        self.search_results = QTextEdit()
        self.search_results.setPlaceholderText("Los resultados de búsqueda aparecerán aquí...")
        
        search_layout.addLayout(search_controls)
        search_layout.addWidget(search_config)
        search_layout.addWidget(QLabel("Resultados de Búsqueda:"))
        search_layout.addWidget(self.search_results)
        search_tab.setLayout(search_layout)
        
        # Agregar tabs
        self.tab_widget.addTab(config_tab, "Configuración")
        self.tab_widget.addTab(upload_tab, "Procesar Fotos")
        self.tab_widget.addTab(search_tab, "Buscar Caras")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar mejorado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sistema listo - Procesador de precisión mejorada inicializado")
        
    def setup_database(self):
        """Configurar base de datos con esquema mejorado"""
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect("data/faces_enhanced.db")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                filename TEXT,
                path TEXT,
                faces_count INTEGER,
                processing_time REAL,
                avg_quality REAL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                photo_id TEXT,
                embeddings TEXT,
                bbox TEXT,
                quality_score REAL,
                method TEXT,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
        
    def update_quality_threshold(self, value):
        threshold = value / 100.0
        self.processor.quality_threshold = threshold
        self.quality_label.setText(f"Umbral mínimo de calidad: {value}%")
        
    def update_threshold_label(self, value):
        self.threshold_label.setText(f"Umbral de similitud: {value}%")
        
    def update_max_results_label(self, value):
        self.max_results_label.setText(f"Máximo resultados: {value}")
        
    def select_photos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar fotos para análisis", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if files:
            self.process_photos_enhanced(files)
    
    def process_photos_enhanced(self, file_paths):
        """Procesar fotos con análisis detallado"""
        self.upload_progress.show()
        self.upload_log.clear()
        
        def process_in_thread():
            total_files = len(file_paths)
            total_faces = 0
            processing_times = []
            
            for i, file_path in enumerate(file_paths):
                start_time = time.time()
                
                try:
                    self.upload_log.append(f"📁 Procesando ({i+1}/{total_files}): {os.path.basename(file_path)}")
                    
                    # Cargar imagen
                    image = cv2.imread(file_path)
                    if image is None:
                        self.upload_log.append("❌ Error: No se pudo cargar la imagen")
                        continue
                    
                    self.upload_log.append(f"📐 Dimensiones: {image.shape[1]}x{image.shape[0]}")
                    
                    # Detectar caras con método mejorado
                    faces = self.processor.detect_faces_multiple_methods(image)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    if faces:
                        total_faces += len(faces)
                        avg_quality = np.mean([f.get('quality_score', 0) for f in faces])
                        
                        self.upload_log.append(f"✅ Detectadas {len(faces)} caras")
                        self.upload_log.append(f"📊 Calidad promedio: {avg_quality:.2f}")
                        
                        # Mostrar detalles de cada cara
                        for j, face in enumerate(faces):
                            quality = face.get('quality_score', 0)
                            method = face.get('method', 'unknown')
                            confidence = face.get('confidence', 0)
                            
                            self.upload_log.append(
                                f"   Cara {j+1}: Calidad={quality:.2f}, "
                                f"Método={method}, Confianza={confidence:.2f}"
                            )
                        
                        # Guardar en base de datos
                        self.save_to_database_enhanced(file_path, faces, processing_time, avg_quality)
                        
                    else:
                        self.upload_log.append("⚠️ No se detectaron caras")
                    
                    self.upload_log.append(f"⏱️ Tiempo de procesamiento: {processing_time:.2f}s")
                    self.upload_log.append("─" * 50)
                    
                    # Actualizar progreso
                    progress = int(((i + 1) / total_files) * 100)
                    self.upload_progress.setValue(progress)
                    
                except Exception as e:
                    self.upload_log.append(f"❌ Error procesando {file_path}: {str(e)}")
                    
                # Permitir que la UI se actualice
                QApplication.processEvents()
            
            # Resumen final
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            self.upload_log.append("=" * 60)
            self.upload_log.append("📋 RESUMEN DEL PROCESAMIENTO:")
            self.upload_log.append(f"📸 Total fotos procesadas: {total_files}")
            self.upload_log.append(f"👥 Total caras detectadas: {total_faces}")
            self.upload_log.append(f"⏱️ Tiempo promedio por foto: {avg_processing_time:.2f}s")
            self.upload_log.append(f"🎯 Caras por foto promedio: {total_faces/max(total_files,1):.1f}")
            self.upload_log.append("✅ Procesamiento completado")
            
            self.upload_progress.hide()
            self.status_bar.showMessage(f"Procesadas {total_files} fotos, {total_faces} caras detectadas")
        
        # Ejecutar en thread separado
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    def save_to_database_enhanced(self, image_path, faces, processing_time, avg_quality):
        """Guardar en base de datos con información mejorada"""
        try:
            conn = sqlite3.connect("data/faces_enhanced.db")
            
            photo_id = str(abs(hash(image_path)))[-10:]
            
            # Insertar foto con estadísticas
            conn.execute(
                """INSERT OR REPLACE INTO photos 
                   (id, filename, path, faces_count, processing_time, avg_quality) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (photo_id, os.path.basename(image_path), image_path, 
                 len(faces), processing_time, avg_quality)
            )
            
            # Insertar caras con información detallada
            for i, face in enumerate(faces):
                face_id = f"{photo_id}_{i}"
                
                conn.execute(
                    """INSERT OR REPLACE INTO faces 
                       (id, photo_id, embeddings, bbox, quality_score, method, confidence) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (face_id, photo_id, json.dumps(face.get('embeddings', {})), 
                     json.dumps(face['bbox']), face.get('quality_score', 0),
                     face.get('method', 'unknown'), face.get('confidence', 0))
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error guardando en BD: {e}")
    
    def search_face(self):
        """Búsqueda de cara con análisis mejorado"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen para buscar", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if not file_path:
            return
            
        self.search_results.clear()
        self.search_results.append("🔍 Iniciando búsqueda avanzada...\n")
        
        try:
            # Cargar y analizar imagen
            image = cv2.imread(file_path)
            if image is None:
                self.search_results.append("❌ Error: No se pudo cargar la imagen")
                return
            
            self.search_results.append(f"📁 Analizando: {os.path.basename(file_path)}")
            self.search_results.append(f"📐 Dimensiones: {image.shape[1]}x{image.shape[0]}\n")
            
            # Detectar caras con método mejorado
            search_faces = self.processor.detect_faces_multiple_methods(image)
            
            if not search_faces:
                self.search_results.append("⚠️ No se detectaron caras en la imagen")
                return
            
            # Mostrar información de caras detectadas
            self.search_results.append(f"✅ Detectadas {len(search_faces)} caras:\n")
            
            for i, face in enumerate(search_faces):
                quality = face.get('quality_score', 0)
                method = face.get('method', 'unknown')
                confidence = face.get('confidence', 0)
                
                self.search_results.append(
                    f"  Cara {i+1}: Calidad={quality:.2f}, "
                    f"Método={method}, Confianza={confidence:.2f}"
                )
            
            # Usar la cara de mejor calidad para búsqueda
            best_face = max(search_faces, key=lambda x: x.get('quality_score', 0))
            self.search_results.append(f"\n🎯 Usando cara de mejor calidad (Score: {best_face.get('quality_score', 0):.2f})\n")
            
            # Configuración de búsqueda
            threshold = self.threshold_slider.value() / 100.0
            max_results = self.max_results_slider.value()
            
            # Buscar en base de datos
            matches = self.search_in_database_enhanced(best_face, threshold, max_results)
            
            # Mostrar resultados
            if matches:
                self.search_results.append(f"🎉 Encontradas {len(matches)} coincidencias:\n")
                
                for i, match in enumerate(matches):
                    similarity_pct = int(match['similarity'] * 100)
                    quality_score = match.get('quality_score', 0)
                    method = match.get('method', 'unknown')
                    
                    self.search_results.append(
                        f"{i+1:2d}. Similitud: {similarity_pct:2d}% | "
                        f"Calidad: {quality_score:.2f} | "
                        f"Método: {method} | "
                        f"ID: {match['photo_id']}"
                    )
                
                # Estadísticas
                avg_similarity = np.mean([m['similarity'] for m in matches])
                self.search_results.append(f"\n📊 Similitud promedio: {avg_similarity:.1%}")
                self.search_results.append(f"🔍 Umbral usado: {threshold:.1%}")
                
            else:
                self.search_results.append("😞 No se encontraron coincidencias con el umbral actual")
                self.search_results.append(f"💡 Intenta reducir el umbral (actual: {threshold:.1%})")
                
        except Exception as e:
            self.search_results.append(f"❌ Error en búsqueda: {str(e)}")
    
    def search_in_database_enhanced(self, search_face, threshold, max_results):
        """Búsqueda mejorada en base de datos"""
        matches = []
        
        try:
            conn = sqlite3.connect("data/faces_enhanced.db")
            cursor = conn.execute("SELECT * FROM faces")
            
            search_embeddings = search_face.get('embeddings', {})
            if not search_embeddings:
                return matches
            
            total_comparisons = 0
            
            for row in cursor.fetchall():
                try:
                    stored_embeddings = json.loads(row[2])  # embeddings column
                    
                    # Usar método de comparación mejorado
                    similarity = self.processor.compare_multiple_embeddings(
                        search_embeddings, stored_embeddings
                    )
                    
                    total_comparisons += 1
                    
                    if similarity >= threshold:
                        matches.append({
                            'face_id': row[0],
                            'photo_id': row[1],
                            'similarity': similarity,
                            'quality_score': row[4] if len(row) > 4 else 0,
                            'method': row[5] if len(row) > 5 else 'unknown',
                            'confidence': row[6] if len(row) > 6 else 0
                        })
                        
                except Exception as e:
                    continue
            
            conn.close()
            
            # Ordenar por similitud y calidad
            matches.sort(key=lambda x: (x['similarity'], x.get('quality_score', 0)), reverse=True)
            
            # Limitar resultados
            matches = matches[:max_results]
            
            print(f"Comparaciones realizadas: {total_comparisons}, Matches: {len(matches)}")
            
        except Exception as e:
            print(f"Error en búsqueda BD: {e}")
        
        return matches
    
    def analyze_face_detailed(self):
        """Análisis detallado de una cara"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen para análisis detallado", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if not file_path:
            return
            
        self.search_results.clear()
        self.search_results.append("🔬 ANÁLISIS DETALLADO DE CARA\n")
        self.search_results.append("=" * 50 + "\n")
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                self.search_results.append("❌ Error: No se pudo cargar la imagen")
                return
            
            # Información básica
            self.search_results.append(f"📁 Archivo: {os.path.basename(file_path)}")
            self.search_results.append(f"📐 Dimensiones: {image.shape[1]}x{image.shape[0]} píxeles")
            self.search_results.append(f"💾 Tamaño archivo: {os.path.getsize(file_path) / 1024:.1f} KB\n")
            
            # Análisis de calidad de imagen
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Nitidez (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500)
            
            # Contraste
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 50)
            
            # Brillo promedio
            brightness = gray.mean()
            
            self.search_results.append("📊 CALIDAD DE IMAGEN:")
            self.search_results.append(f"  🔍 Nitidez: {sharpness_score:.2f} (Laplacian: {laplacian_var:.1f})")
            self.search_results.append(f"  🌓 Contraste: {contrast_score:.2f} (Std: {contrast:.1f})")
            self.search_results.append(f"  💡 Brillo promedio: {brightness:.1f}/255\n")
            
            # Detectar caras
            faces = self.processor.detect_faces_multiple_methods(image)
            
            if not faces:
                self.search_results.append("⚠️ No se detectaron caras en la imagen")
                return
            
            self.search_results.append(f"👥 CARAS DETECTADAS: {len(faces)}\n")
            
            # Análisis detallado de cada cara
            for i, face in enumerate(faces):
                self.search_results.append(f"👤 CARA {i+1}:")
                self.search_results.append("-" * 30)
                
                bbox = face['bbox']
                quality = face.get('quality_score', 0)
                method = face.get('method', 'unknown')
                confidence = face.get('confidence', 0)
                
                # Información básica de la cara
                self.search_results.append(f"  📍 Posición: ({bbox['x']}, {bbox['y']})")
                self.search_results.append(f"  📏 Tamaño: {bbox['width']}x{bbox['height']} píxeles")
                self.search_results.append(f"  🎯 Método detección: {method}")
                self.search_results.append(f"  💯 Confianza detección: {confidence:.2f}")
                self.search_results.append(f"  ⭐ Score calidad: {quality:.2f}")
                
                # Análisis de embeddings
                embeddings = face.get('embeddings', {})
                self.search_results.append(f"  🧠 Embeddings generados: {len(embeddings)}")
                
                for emb_type, emb_data in embeddings.items():
                    if isinstance(emb_data, list):
                        self.search_results.append(f"    - {emb_type}: {len(emb_data)} dimensiones")
                
                # Análisis específico de la región de la cara
                face_image = face['face_image']
                if face_image.size > 0:
                    face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    
                    # Nitidez específica de la cara
                    face_sharpness = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                    face_contrast = face_gray.std()
                    face_brightness = face_gray.mean()
                    
                    # Relación de aspecto
                    aspect_ratio = bbox['width'] / max(bbox['height'], 1)
                    
                    self.search_results.append(f"  🔍 Nitidez cara: {face_sharpness:.1f}")
                    self.search_results.append(f"  🌓 Contraste cara: {face_contrast:.1f}")
                    self.search_results.append(f"  💡 Brillo cara: {face_brightness:.1f}")
                    self.search_results.append(f"  📐 Relación aspecto: {aspect_ratio:.2f}")
                
                self.search_results.append("")  # Línea en blanco
            
            # Recomendaciones
            self.search_results.append("💡 RECOMENDACIONES:")
            
            if laplacian_var < 100:
                self.search_results.append("  ⚠️ Imagen con poca nitidez - considera usar foto más nítida")
            
            if contrast < 30:
                self.search_results.append("  ⚠️ Bajo contraste - considera mejorar iluminación")
            
            if brightness < 50:
                self.search_results.append("  ⚠️ Imagen muy oscura - considera más iluminación")
            elif brightness > 200:
                self.search_results.append("  ⚠️ Imagen muy brillante - considera menos exposición")
            
            best_quality = max(faces, key=lambda x: x.get('quality_score', 0))
            if best_quality.get('quality_score', 0) < 0.5:
                self.search_results.append("  ⚠️ Calidad general baja - considera foto de mejor calidad")
            
            # Estadísticas finales
            avg_quality = np.mean([f.get('quality_score', 0) for f in faces])
            self.search_results.append(f"\n📊 Calidad promedio: {avg_quality:.2f}")
            
            if avg_quality >= 0.7:
                self.search_results.append("✅ Excelente calidad para reconocimiento facial")
            elif avg_quality >= 0.5:
                self.search_results.append("👍 Buena calidad para reconocimiento facial")
            elif avg_quality >= 0.3:
                self.search_results.append("⚠️ Calidad aceptable - podrían haber mejores resultados")
            else:
                self.search_results.append("❌ Calidad baja - recomendado usar imagen de mejor calidad")
                
        except Exception as e:
            self.search_results.append(f"❌ Error en análisis: {str(e)}")

def main():
    """Función principal"""
    if not PYQT_AVAILABLE:
        print("PyQt6 no disponible. Instala con: pip install PyQt6")
        return
        
    if not CV2_AVAILABLE:
        print("OpenCV no disponible. Instala con: pip install opencv-python")
        return
        
    if not FACE_REC_AVAILABLE:
        print("face_recognition no disponible. Instala con: pip install face-recognition")
        return
    
    import time
    app = QApplication(sys.argv)
    
    # Mostrar splash screen mientras carga
    splash_msg = QLabel("Cargando Face Recognition - Precisión Mejorada...")
    splash_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
    splash_msg.setStyleSheet("""
        QLabel {
            background-color: #2c3e50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            border-radius: 10px;
        }
    """)
    splash_msg.show()
    QApplication.processEvents()
    
    # Cargar aplicación
    time.sleep(1)  # Simular carga
    window = EnhancedFaceApp()
    
    splash_msg.hide()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
