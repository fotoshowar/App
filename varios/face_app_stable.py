#!/usr/bin/env python3
"""
Aplicación de reconocimiento facial estable
Solo usa face_recognition y OpenCV para evitar conflictos
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import threading

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QProgressBar, QTextEdit,
        QFileDialog, QScrollArea, QGridLayout, QSlider, 
        QGroupBox, QFrame, QMessageBox, QStatusBar
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False

class StableFaceProcessor:
    """Procesador usando solo face_recognition (sin PyTorch)"""
    
    def __init__(self):
        print("Inicializando procesador estable...")
        self.available_models = []
        
        if FACE_REC_AVAILABLE:
            self.available_models.append("face_recognition")
            print("face_recognition disponible")
        
        if not self.available_models:
            raise Exception("No hay modelos de reconocimiento disponibles")
    
    def detect_and_encode_faces(self, image):
        """Detectar y codificar caras usando solo face_recognition"""
        if not FACE_REC_AVAILABLE:
            return []
        
        try:
            # Convertir a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detectar ubicaciones de caras
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                return []
            
            # Generar encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            faces = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Extraer región de la cara
                face_image = image[top:bottom, left:right]
                
                faces.append({
                    'bbox': {'x': left, 'y': top, 'width': right-left, 'height': bottom-top},
                    'confidence': 0.9,
                    'face_image': face_image,
                    'method': 'face_recognition',
                    'embeddings': {'face_recognition': encoding.tolist()}
                })
            
            return faces
            
        except Exception as e:
            print(f"Error procesando caras: {e}")
            return []
    
    def compare_embeddings(self, embedding1, embedding2):
        """Comparar embeddings usando face_recognition"""
        try:
            distance = face_recognition.face_distance([embedding1], embedding2)[0]
            similarity = 1.0 - distance
            return max(0.0, similarity)
        except:
            return 0.0

class ProcessingThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    
    def __init__(self, processor, file_paths):
        super().__init__()
        self.processor = processor
        self.file_paths = file_paths
    
    def run(self):
        try:
            results = []
            total = len(self.file_paths)
            
            for i, file_path in enumerate(self.file_paths):
                self.progress.emit(f"Procesando {i+1}/{total}: {os.path.basename(file_path)}")
                
                image = cv2.imread(file_path)
                if image is not None:
                    faces = self.processor.detect_and_encode_faces(image)
                    results.append({
                        'file_path': file_path,
                        'faces': faces,
                        'success': True
                    })
                else:
                    results.append({
                        'file_path': file_path,
                        'faces': [],
                        'success': False,
                        'error': 'No se pudo cargar imagen'
                    })
            
            self.finished.emit({'results': results, 'success': True})
            
        except Exception as e:
            self.finished.emit({'success': False, 'error': str(e)})

class StableFaceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = StableFaceProcessor()
        self.init_ui()
        self.setup_database()
        
    def init_ui(self):
        self.setWindowTitle("Face Recognition - Versión Estable")
        self.setGeometry(100, 100, 1000, 700)
        
        # Tema oscuro
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QPushButton { 
                background-color: #0078d4; color: white; border: none; 
                padding: 10px; border-radius: 5px; font-weight: bold; 
            }
            QPushButton:hover { background-color: #106ebe; }
            QTabWidget::pane { border: 1px solid #0078d4; }
            QTabBar::tab { 
                background-color: #3c3c3c; color: white; 
                padding: 8px 16px; margin: 2px; 
            }
            QTabBar::tab:selected { background-color: #0078d4; }
            QTextEdit { background-color: #1e1e1e; color: white; border: 1px solid #3c3c3c; }
            QProgressBar { border: 1px solid #3c3c3c; border-radius: 3px; }
            QProgressBar::chunk { background-color: #0078d4; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header
        header = QLabel("Face Recognition - Versión Estable")
        header.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;")
        layout.addWidget(header)
        
        # Tabs
        self.tab_widget = QTabWidget()
        
        # Tab de carga
        upload_tab = QWidget()
        upload_layout = QVBoxLayout()
        
        self.upload_btn = QPushButton("Seleccionar Fotos")
        self.upload_btn.clicked.connect(self.select_photos)
        
        self.upload_progress = QProgressBar()
        self.upload_progress.hide()
        
        self.upload_log = QTextEdit()
        self.upload_log.setMaximumHeight(200)
        
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addWidget(self.upload_progress)
        upload_layout.addWidget(QLabel("Log:"))
        upload_layout.addWidget(self.upload_log)
        upload_tab.setLayout(upload_layout)
        
        # Tab de búsqueda
        search_tab = QWidget()
        search_layout = QVBoxLayout()
        
        search_btn = QPushButton("Buscar con Imagen")
        search_btn.clicked.connect(self.search_face)
        
        # Slider de umbral
        threshold_group = QGroupBox("Umbral de similitud")
        threshold_layout = QVBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        self.threshold_label = QLabel("70%")
        
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_group.setLayout(threshold_layout)
        
        self.search_results = QTextEdit()
        self.search_results.setMaximumHeight(300)
        
        search_layout.addWidget(search_btn)
        search_layout.addWidget(threshold_group)
        search_layout.addWidget(QLabel("Resultados:"))
        search_layout.addWidget(self.search_results)
        search_tab.setLayout(search_layout)
        
        self.tab_widget.addTab(upload_tab, "Subir")
        self.tab_widget.addTab(search_tab, "Buscar")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
        
    def setup_database(self):
        """Configurar base de datos"""
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect("data/faces_stable.db")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                filename TEXT,
                path TEXT,
                faces_count INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                photo_id TEXT,
                embeddings TEXT,
                bbox TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def update_threshold_label(self, value):
        self.threshold_label.setText(f"{value}%")
        
    def select_photos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar fotos", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if files:
            self.upload_progress.show()
            self.upload_log.clear()
            
            self.processing_thread = ProcessingThread(self.processor, files)
            self.processing_thread.finished.connect(self.on_processing_finished)
            self.processing_thread.progress.connect(self.on_processing_progress)
            self.processing_thread.start()
    
    def on_processing_progress(self, message):
        self.upload_log.append(message)
        
    def on_processing_finished(self, result):
        self.upload_progress.hide()
        
        if result['success']:
            total_faces = 0
            for item in result['results']:
                if item['success']:
                    faces_count = len(item['faces'])
                    total_faces += faces_count
                    self.upload_log.append(f"✅ {os.path.basename(item['file_path'])}: {faces_count} caras")
                    
                    # Guardar en BD
                    self.save_to_database(item['file_path'], item['faces'])
                else:
                    self.upload_log.append(f"❌ Error: {item['file_path']}")
            
            self.upload_log.append(f"\nTotal: {total_faces} caras procesadas")
        else:
            self.upload_log.append(f"Error: {result['error']}")
    
    def save_to_database(self, image_path, faces):
        try:
            conn = sqlite3.connect("data/faces_stable.db")
            
            photo_id = str(abs(hash(image_path)))[-8:]
            
            conn.execute(
                "INSERT OR REPLACE INTO photos (id, filename, path, faces_count) VALUES (?, ?, ?, ?)",
                (photo_id, os.path.basename(image_path), image_path, len(faces))
            )
            
            for i, face in enumerate(faces):
                face_id = f"{photo_id}_{i}"
                conn.execute(
                    "INSERT OR REPLACE INTO faces (id, photo_id, embeddings, bbox) VALUES (?, ?, ?, ?)",
                    (face_id, photo_id, json.dumps(face['embeddings']), json.dumps(face['bbox']))
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error guardando: {e}")
    
    def search_face(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Imagen para buscar", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not file_path:
            return
            
        try:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Error", "No se pudo cargar la imagen")
                return
            
            # Detectar caras
            search_faces = self.processor.detect_and_encode_faces(image)
            if not search_faces:
                self.search_results.setText("No se detectaron caras en la imagen")
                return
            
            # Buscar coincidencias
            search_embedding = search_faces[0]['embeddings']['face_recognition']
            threshold = self.threshold_slider.value() / 100.0
            
            matches = self.search_in_database(search_embedding, threshold)
            
            self.search_results.clear()
            if matches:
                self.search_results.append(f"Encontradas {len(matches)} coincidencias:\n")
                for match in matches:
                    similarity_pct = int(match['similarity'] * 100)
                    self.search_results.append(f"- {similarity_pct}% similar (ID: {match['photo_id']})")
            else:
                self.search_results.setText("No se encontraron coincidencias")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en búsqueda: {e}")
    
    def search_in_database(self, search_embedding, threshold):
        matches = []
        
        try:
            conn = sqlite3.connect("data/faces_stable.db")
            cursor = conn.execute("SELECT * FROM faces")
            
            for row in cursor.fetchall():
                try:
                    stored_embeddings = json.loads(row[2])
                    if 'face_recognition' in stored_embeddings:
                        stored_embedding = stored_embeddings['face_recognition']
                        similarity = self.processor.compare_embeddings(search_embedding, stored_embedding)
                        
                        if similarity >= threshold:
                            matches.append({
                                'face_id': row[0],
                                'photo_id': row[1],
                                'similarity': similarity
                            })
                except:
                    continue
            
            conn.close()
            
        except Exception as e:
            print(f"Error buscando: {e}")
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)

def main():
    if not PYQT_AVAILABLE:
        print("PyQt6 no disponible")
        return
        
    if not CV2_AVAILABLE:
        print("OpenCV no disponible")
        return
        
    if not FACE_REC_AVAILABLE:
        print("face_recognition no disponible")
        return
        
    app = QApplication(sys.argv)
    window = StableFaceApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
