"""
Aplicación de Escritorio para Reconocimiento Facial
Interfaz moderna y rápida usando PyQt6
"""

import sys
import os
import cv2
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import threading
import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QProgressBar, QTextEdit,
    QFileDialog, QScrollArea, QGridLayout, QSlider, QSpinBox,
    QGroupBox, QFrame, QSplitter, QStatusBar, QMessageBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, 
    QEasingCurve, QRect, QSize, pyqtProperty
)
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QIcon, QPalette, QColor,
    QPainter, QPen, QBrush, QLinearGradient
)

# Importar tu procesador existente
from advanced_face_processor import AdvancedFaceProcessor

class CameraThread(QThread):
    """Thread para captura de cámara en tiempo real"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        
    def start_capture(self):
        self.running = True
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.start()
        
    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
    def run(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Espejo horizontal para selfies
                frame = cv2.flip(frame, 1)
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30 FPS

class FaceProcessingThread(QThread):
    """Thread para procesamiento de caras en background"""
    processing_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(int, str)
    
    def __init__(self, face_processor, image_path, operation="detect"):
        super().__init__()
        self.face_processor = face_processor
        self.image_path = image_path
        self.operation = operation
        self.search_image = None
        self.threshold = 0.7
        
    def set_search_params(self, search_image, threshold):
        self.search_image = search_image
        self.threshold = threshold
        self.operation = "search"
        
    def run(self):
        try:
            if self.operation == "detect":
                self.progress_update.emit(20, "Cargando imagen...")
                image = cv2.imread(self.image_path)
                
                self.progress_update.emit(50, "Detectando caras...")
                faces = self.face_processor.detect_and_encode_faces(image)
                
                self.progress_update.emit(100, "Completado!")
                self.processing_complete.emit({
                    'success': True,
                    'faces': faces,
                    'image_path': self.image_path
                })
                
            elif self.operation == "search":
                self.progress_update.emit(30, "Procesando imagen de búsqueda...")
                search_faces = self.face_processor.detect_and_encode_faces(self.search_image)
                
                if not search_faces:
                    self.processing_complete.emit({
                        'success': False,
                        'error': 'No se detectaron caras en la imagen de búsqueda'
                    })
                    return
                
                self.progress_update.emit(60, "Comparando con base de datos...")
                # Aquí integrarías con tu base de datos existente
                results = self.search_in_database(search_faces[0], self.threshold)
                
                self.progress_update.emit(100, "Búsqueda completada!")
                self.processing_complete.emit({
                    'success': True,
                    'matches': results,
                    'total_found': len(results)
                })
                
        except Exception as e:
            self.processing_complete.emit({
                'success': False,
                'error': str(e)
            })
    
    def search_in_database(self, search_face, threshold):
        """Buscar caras similares en la base de datos"""
        # Implementar búsqueda en BD (simplificado)
        return []

class AnimatedButton(QPushButton):
    """Botón con animaciones suaves"""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
    def enterEvent(self, event):
        self.animate_scale(1.05)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.animate_scale(1.0)
        super().leaveEvent(event)
        
    def animate_scale(self, scale):
        current_geometry = self.geometry()
        center = current_geometry.center()
        new_size = QSize(
            int(current_geometry.width() * scale),
            int(current_geometry.height() * scale)
        )
        new_geometry = QRect(0, 0, new_size.width(), new_size.height())
        new_geometry.moveCenter(center)
        
        self.animation.setStartValue(current_geometry)
        self.animation.setEndValue(new_geometry)
        self.animation.start()

class ModernPhotoWidget(QWidget):
    """Widget para mostrar fotos con estilo moderno"""
    
    def __init__(self, photo_data, parent=None):
        super().__init__(parent)
        self.photo_data = photo_data
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Imagen
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 150)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                border-radius: 10px;
                background: #2c3e50;
            }
        """)
        self.image_label.setScaledContents(True)
        
        # Cargar imagen
        if os.path.exists(self.photo_data.get('path', '')):
            pixmap = QPixmap(self.photo_data['path'])
            self.image_label.setPixmap(pixmap)
        
        # Info
        info_label = QLabel(f"{self.photo_data.get('filename', 'Sin nombre')}\n"
                           f"{self.photo_data.get('faces_count', 0)} caras")
        info_label.setStyleSheet("color: white; font-size: 12px;")
        
        layout.addWidget(self.image_label)
        layout.addWidget(info_label)
        self.setLayout(layout)

class FaceRecognitionApp(QMainWindow):
    """Aplicación principal de reconocimiento facial"""
    
    def __init__(self):
        super().__init__()
        self.face_processor = None
        self.camera_thread = CameraThread()
        self.current_frame = None
        
        # Base de datos
        self.db_path = "data/face_recognition.db"
        self.init_database()
        
        self.init_ui()
        self.init_face_processor()
        
    def init_database(self):
        """Inicializar base de datos"""
        os.makedirs("data", exist_ok=True)
        # Usar el mismo esquema que tu main.py
        
    def init_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle("Face Recognition Desktop - Reconocimiento Facial Avanzado")
        self.setGeometry(100, 100, 1400, 900)
        
        # Tema oscuro moderno
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #34495e);
                color: white;
            }
            QTabWidget::pane {
                border: 2px solid #3498db;
                border-radius: 10px;
                background: #2c3e50;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background: #34495e;
                color: white;
                padding: 12px 24px;
                margin: 2px;
                border-radius: 8px;
                font-weight: bold;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
            QTabBar::tab:hover {
                background: #2980b9;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }
            QPushButton:pressed {
                background: #1f618d;
            }
            QPushButton:disabled {
                background: #7f8c8d;
                color: #bdc3c7;
            }
            QLabel {
                color: white;
            }
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 8px;
                background: #2c3e50;
                color: white;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 6px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3498db;
                height: 8px;
                background: #2c3e50;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 20px;
                border-radius: 10px;
            }
        """)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tabs
        self.tab_widget = QTabWidget()
        self.create_tabs()
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo - Cargando procesador de caras...")
        
    def create_header(self):
        """Crear header con estadísticas"""
        header_frame = QFrame()
        header_frame.setFixedHeight(120)
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 15px;
                margin: 10px;
            }
        """)
        
        layout = QHBoxLayout()
        
        # Título
        title_label = QLabel("🔍 Face Recognition Desktop")
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: white;
            margin: 20px;
        """)
        
        # Stats
        self.stats_label = QLabel("Fotos: 0 | Caras: 0")
        self.stats_label.setStyleSheet("""
            font-size: 16px;
            color: white;
            margin: 20px;
        """)
        
        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(self.stats_label)
        
        header_frame.setLayout(layout)
        return header_frame
        
    def create_tabs(self):
        """Crear todas las pestañas"""
        # Pestaña de carga
        self.upload_tab = self.create_upload_tab()
        self.tab_widget.addTab(self.upload_tab, "📤 Subir Fotos")
        
        # Pestaña de búsqueda
        self.search_tab = self.create_search_tab()
        self.tab_widget.addTab(self.search_tab, "🔍 Buscar Cara")
        
        # Pestaña de galería
        self.gallery_tab = self.create_gallery_tab()
        self.tab_widget.addTab(self.gallery_tab, "🖼️ Galería")
        
        # Pestaña de resultados
        self.results_tab = self.create_results_tab()
        self.tab_widget.addTab(self.results_tab, "🎯 Resultados")
        
    def create_upload_tab(self):
        """Crear pestaña de carga de archivos"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Botón de carga
        upload_btn = AnimatedButton("📁 Seleccionar Fotos")
        upload_btn.setFixedHeight(60)
        upload_btn.clicked.connect(self.select_photos)
        
        # Área de arrastrar y soltar
        drop_area = QLabel("Arrastra y suelta fotos aquí\n\nFormatos: JPG, PNG, GIF, WebP")
        drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_area.setFixedHeight(200)
        drop_area.setStyleSheet("""
            QLabel {
                border: 3px dashed #3498db;
                border-radius: 15px;
                background: rgba(52, 152, 219, 0.1);
                font-size: 18px;
                color: #3498db;
                margin: 20px;
            }
        """)
        
        # Progress bar
        self.upload_progress = QProgressBar()
        self.upload_progress.hide()
        
        # Log de procesamiento
        self.upload_log = QTextEdit()
        self.upload_log.setMaximumHeight(150)
        self.upload_log.setStyleSheet("""
            QTextEdit {
                background: #2c3e50;
                border: 2px solid #3498db;
                border-radius: 8px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                color: #ecf0f1;
            }
        """)
        
        layout.addWidget(upload_btn)
        layout.addWidget(drop_area)
        layout.addWidget(self.upload_progress)
        layout.addWidget(QLabel("Log de procesamiento:"))
        layout.addWidget(self.upload_log)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
        
    def create_search_tab(self):
        """Crear pestaña de búsqueda"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Control de umbral
        threshold_group = QGroupBox("Configuración de Búsqueda")
        threshold_layout = QVBoxLayout()
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        self.threshold_label = QLabel("Umbral de similitud: 70%")
        
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_group.setLayout(threshold_layout)
        
        # Splitter para cámara y controles
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel de cámara
        camera_widget = QWidget()
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel("Cámara desactivada")
        self.camera_label.setFixedSize(400, 300)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                border-radius: 10px;
                background: #2c3e50;
                font-size: 16px;
            }
        """)
        
        camera_controls = QHBoxLayout()
        self.camera_btn = AnimatedButton("📷 Activar Cámara")
        self.camera_btn.clicked.connect(self.toggle_camera)
        
        self.capture_btn = AnimatedButton("📸 Capturar y Buscar")
        self.capture_btn.clicked.connect(self.capture_and_search)
        self.capture_btn.setEnabled(False)
        
        camera_controls.addWidget(self.camera_btn)
        camera_controls.addWidget(self.capture_btn)
        
        camera_layout.addWidget(self.camera_label)
        camera_layout.addLayout(camera_controls)
        camera_widget.setLayout(camera_layout)
        
        # Panel de archivo
        file_widget = QWidget()
        file_layout = QVBoxLayout()
        
        file_search_btn = AnimatedButton("📁 Buscar con Archivo")
        file_search_btn.clicked.connect(self.search_with_file)
        
        # Progress para búsqueda
        self.search_progress = QProgressBar()
        self.search_progress.hide()
        
        self.search_status = QLabel("")
        
        file_layout.addWidget(QLabel("O buscar con archivo:"))
        file_layout.addWidget(file_search_btn)
        file_layout.addWidget(self.search_progress)
        file_layout.addWidget(self.search_status)
        file_layout.addStretch()
        file_widget.setLayout(file_layout)
        
        splitter.addWidget(camera_widget)
        splitter.addWidget(file_widget)
        splitter.setSizes([400, 300])
        
        layout.addWidget(threshold_group)
        layout.addWidget(splitter)
        
        tab.setLayout(layout)
        return tab
        
    def create_gallery_tab(self):
        """Crear pestaña de galería"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controles de galería
        controls = QHBoxLayout()
        refresh_btn = AnimatedButton("🔄 Actualizar")
        refresh_btn.clicked.connect(self.refresh_gallery)
        clear_btn = AnimatedButton("🗑️ Limpiar Todo")
        clear_btn.clicked.connect(self.clear_all_photos)
        
        controls.addWidget(refresh_btn)
        controls.addWidget(clear_btn)
        controls.addStretch()
        
        # Área de scroll para fotos
        self.gallery_scroll = QScrollArea()
        self.gallery_content = QWidget()
        self.gallery_layout = QGridLayout()
        self.gallery_content.setLayout(self.gallery_layout)
        self.gallery_scroll.setWidget(self.gallery_content)
        self.gallery_scroll.setWidgetResizable(True)
        
        layout.addLayout(controls)
        layout.addWidget(self.gallery_scroll)
        
        tab.setLayout(layout)
        return tab
        
    def create_results_tab(self):
        """Crear pestaña de resultados"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.results_label = QLabel("No hay resultados de búsqueda")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Área de scroll para resultados
        self.results_scroll = QScrollArea()
        self.results_content = QWidget()
        self.results_layout = QGridLayout()
        self.results_content.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_content)
        self.results_scroll.setWidgetResizable(True)
        
        clear_results_btn = AnimatedButton("🧹 Limpiar Resultados")
        clear_results_btn.clicked.connect(self.clear_results)
        
        layout.addWidget(self.results_label)
        layout.addWidget(self.results_scroll)
        layout.addWidget(clear_results_btn)
        
        tab.setLayout(layout)
        return tab
        
    def init_face_processor(self):
        """Inicializar procesador de caras en thread separado"""
        def load_processor():
            try:
                self.face_processor = AdvancedFaceProcessor(device='cpu')
                self.status_bar.showMessage("✅ Procesador de caras cargado correctamente")
            except Exception as e:
                self.status_bar.showMessage(f"❌ Error cargando procesador: {str(e)}")
                
        thread = threading.Thread(target=load_processor)
        thread.daemon = True
        thread.start()
        
    def update_threshold_label(self, value):
        """Actualizar etiqueta del umbral"""
        self.threshold_label.setText(f"Umbral de similitud: {value}%")
        
    def select_photos(self):
        """Seleccionar fotos para procesar"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar fotos",
            "",
            "Imágenes (*.jpg *.jpeg *.png *.gif *.webp *.bmp)"
        )
        
        if file_paths:
            self.process_photos(file_paths)
            
    def process_photos(self, file_paths):
        """Procesar fotos seleccionadas"""
        if not self.face_processor:
            QMessageBox.warning(self, "Error", "El procesador de caras aún no está listo")
            return
            
        self.upload_progress.show()
        self.upload_progress.setValue(0)
        self.upload_log.clear()
        
        def process_in_thread():
            total_files = len(file_paths)
            for i, file_path in enumerate(file_paths):
                try:
                    self.upload_log.append(f"📝 Procesando: {os.path.basename(file_path)}")
                    
                    # Crear thread de procesamiento
                    processor_thread = FaceProcessingThread(self.face_processor, file_path)
                    processor_thread.processing_complete.connect(self.on_photo_processed)
                    processor_thread.start()
                    processor_thread.wait()  # Esperar a que termine
                    
                    # Actualizar progress
                    progress = int(((i + 1) / total_files) * 100)
                    self.upload_progress.setValue(progress)
                    
                except Exception as e:
                    self.upload_log.append(f"❌ Error procesando {file_path}: {str(e)}")
                    
            self.upload_progress.hide()
            self.upload_log.append("✅ Procesamiento completado")
            
        thread = threading.Thread(target=process_in_thread)
        thread.daemon = True
        thread.start()
        
    def on_photo_processed(self, result):
        """Callback cuando una foto es procesada"""
        if result['success']:
            faces = result['faces']
            self.upload_log.append(f"✅ {len(faces)} caras detectadas")
            # Aquí guardarías en la base de datos
        else:
            self.upload_log.append(f"❌ Error: {result.get('error', 'Desconocido')}")
            
    def toggle_camera(self):
        """Activar/desactivar cámara"""
        if not self.camera_thread.running:
            self.camera_thread.frame_ready.connect(self.update_camera_frame)
            self.camera_thread.start_capture()
            self.camera_btn.setText("📷 Desactivar Cámara")
            self.capture_btn.setEnabled(True)
        else:
            self.camera_thread.stop_capture()
            self.camera_btn.setText("📷 Activar Cámara")
            self.capture_btn.setEnabled(False)
            self.camera_label.setText("Cámara desactivada")
            
    def update_camera_frame(self, frame):
        """Actualizar frame de la cámara"""
        self.current_frame = frame.copy()
        
        # Convertir a QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Escalar y mostrar
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
        
    def capture_and_search(self):
        """Capturar frame actual y buscar"""
        if self.current_frame is None:
            return
            
        if not self.face_processor:
            QMessageBox.warning(self, "Error", "El procesador de caras aún no está listo")
            return
            
        self.search_progress.show()
        threshold = self.threshold_slider.value() / 100.0
        
        # Crear thread de búsqueda
        search_thread = FaceProcessingThread(self.face_processor, None)
        search_thread.set_search_params(self.current_frame, threshold)
        search_thread.processing_complete.connect(self.on_search_complete)
        search_thread.progress_update.connect(self.update_search_progress)
        search_thread.start()
        
    def search_with_file(self):
        """Buscar con archivo seleccionado"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen para buscar",
            "",
            "Imágenes (*.jpg *.jpeg *.png *.gif *.webp *.bmp)"
        )
        
        if file_path and self.face_processor:
            image = cv2.imread(file_path)
            if image is not None:
                self.search_progress.show()
                threshold = self.threshold_slider.value() / 100.0
                
                search_thread = FaceProcessingThread(self.face_processor, None)
                search_thread.set_search_params(image, threshold)
                search_thread.processing_complete.connect(self.on_search_complete)
                search_thread.progress_update.connect(self.update_search_progress)
                search_thread.start()
                
    def update_search_progress(self, value, status):
        """Actualizar progreso de búsqueda"""
        self.search_progress.setValue(value)
        self.search_status.setText(status)
        
    def on_search_complete(self, result):
        """Callback cuando búsqueda es completada"""
        self.search_progress.hide()
        self.search_status.setText("")
        
        if result['success']:
            matches = result.get('matches', [])
            self.display_search_results(matches)
            self.tab_widget.setCurrentIndex(3)  # Cambiar a pestaña de resultados
        else:
            QMessageBox.information(self, "Búsqueda", result.get('error', 'No se encontraron coincidencias'))
            
    def display_search_results(self, matches):
        """Mostrar resultados de búsqueda"""
        # Limpiar resultados anteriores
        self.clear_results()
        
        if not matches:
            self.results_label.setText("No se encontraron coincidencias")
            return
            
        self.results_label.setText(f"Se encontraron {len(matches)} coincidencias")
        
        # Mostrar resultados en grid
        row, col = 0, 0
        for match in matches[:12]:  # Limitar a 12 resultados
            result_widget = ModernPhotoWidget(match)
            self.results_layout.addWidget(result_widget, row, col)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
                
    def refresh_gallery(self):
        """Actualizar galería con fotos de la base de datos"""
        # Limpiar galería actual
        for i in reversed(range(self.gallery_layout.count())):
            child = self.gallery_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Cargar fotos desde BD (simplificado)
        # Aquí integrarías con tu base de datos existente
        photos = self.get_photos_from_db()
        
        row, col = 0, 0
        for photo in photos:
            photo_widget = ModernPhotoWidget(photo)
            self.gallery_layout.addWidget(photo_widget, row, col)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
                
    def get_photos_from_db(self):
        """Obtener fotos de la base de datos"""
        # Implementar consulta a BD
        return []
        
    def clear_results(self):
        """Limpiar resultados de búsqueda"""
        for i in reversed(range(self.results_layout.count())):
            child = self.results_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        self.results_label.setText("No hay resultados de búsqueda")
        
    def clear_all_photos(self):
        """Limpiar todas las fotos"""
        reply = QMessageBox.question(
            self,
            "Confirmar",
            "¿Estás seguro de que quieres eliminar todas las fotos y caras?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Limpiar base de datos
            # Implementar limpieza
            self.refresh_gallery()
            self.status_bar.showMessage("Todas las fotos han sido eliminadas")
            
    def closeEvent(self, event):
        """Limpiar recursos al cerrar"""
        if self.camera_thread.running:
            self.camera_thread.stop_capture()
        event.accept()

def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Configurar aplicación
    app.setApplicationName("Face Recognition Desktop")
    app.setApplicationVersion("1.0.0")
    
    # Crear y mostrar ventana principal
    window = FaceRecognitionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
