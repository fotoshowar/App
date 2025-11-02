#!/usr/bin/env python3
"""
Aplicación Web Local de Reconocimiento Facial
Se ejecuta localmente y abre automáticamente el navegador
"""

import asyncio
import webbrowser
import threading
import time
import signal
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import sqlite3
import os
import cv2
import numpy as np
import json
import uuid
from datetime import datetime
import aiofiles

# Importar procesador optimizado
try:
    from optimized_advanced_face_processor import OptimizedAdvancedFaceProcessor as FaceProcessor
except ImportError:
    print("Usando procesador básico...")
    from advanced_face_processor import AdvancedFaceProcessor as FaceProcessor

class LocalFaceRecognitionApp:
    def __init__(self, port=8080, auto_open=True):
        self.port = port
        self.auto_open = auto_open
        self.app = FastAPI(title="Face Finder Local")
        self.face_processor = None
        
        # Directorios
        self.upload_dir = Path("uploads")
        self.photos_dir = self.upload_dir / "photos"
        self.faces_dir = self.upload_dir / "faces"
        
        for directory in [self.upload_dir, self.photos_dir, self.faces_dir]:
            directory.mkdir(exist_ok=True)
        
        # Base de datos
        self.db_path = "face_recognition_local.db"
        self.init_database()
        
        # Configurar aplicación
        self.setup_routes()
        self.setup_middleware()
        
        # Inicializar procesador en hilo separado
        threading.Thread(target=self.init_face_processor, daemon=True).start()
    
    def init_database(self):
        """Inicializar base de datos SQLite"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                faces_count INTEGER DEFAULT 0,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                embeddings TEXT NOT NULL,
                bounding_box TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_face_processor(self):
        """Inicializar procesador de caras"""
        print("Inicializando procesador de reconocimiento facial...")
        try:
            self.face_processor = FaceProcessor(
                device='cpu',
                optimization_level='high',
                cache_embeddings=True,
                use_threading=True
            )
            print("Procesador inicializado correctamente")
        except Exception as e:
            print(f"Error inicializando procesador: {e}")
            self.face_processor = None
    
    def setup_middleware(self):
        """Configurar middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Configurar rutas de la aplicación"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_main_page():
            return self.get_html_interface()
        
        @self.app.post("/api/upload-photo")
        async def upload_photo(file: UploadFile = File(...)):
            return await self.handle_upload_photo(file)
        
        @self.app.post("/api/search-face")
        async def search_face(file: UploadFile = File(...), threshold: float = Query(0.85)):
            return await self.handle_search_face(file, threshold)
        
        @self.app.get("/api/photos")
        async def get_photos():
            return await self.handle_get_photos()
        
        @self.app.get("/api/stats")
        async def get_stats():
            return await self.handle_get_stats()
        
        @self.app.delete("/api/photos/{photo_id}")
        async def delete_photo(photo_id: str):
            return await self.handle_delete_photo(photo_id)
        
        @self.app.get("/api/image/{image_type}/{image_id}")
        async def get_image(image_type: str, image_id: str):
            return await self.handle_get_image(image_type, image_id)
        
        @self.app.get("/api/status")
        async def get_status():
            return {
                "processor_ready": self.face_processor is not None,
                "status": "ready" if self.face_processor else "loading"
            }
    
    def get_html_interface(self) -> str:
        """Generar interfaz HTML completa"""
        return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Finder Local</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 15px 25px;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .stat-number { font-size: 2rem; font-weight: bold; }
        .stat-label { font-size: 0.9rem; opacity: 0.8; }
        
        .tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        
        .tab {
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            padding: 12px 24px;
            margin: 0 5px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .tab:hover, .tab.active {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        .content {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.05);
        }
        
        .upload-area.dragover {
            border-color: #4CAF50;
            background: rgba(76,175,80,0.1);
        }
        
        .btn {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 5px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .btn-secondary {
            background: linear-gradient(45deg, #2196F3, #1976D2);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f44336, #d32f2f);
        }
        
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .photo-card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }
        
        .photo-card:hover {
            transform: scale(1.05);
        }
        
        .photo-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .photo-info {
            padding: 15px;
            text-align: center;
        }
        
        .hidden { display: none; }
        
        .status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .status.loading {
            background: rgba(255, 193, 7, 0.9);
            color: black;
        }
        
        .status.ready {
            background: rgba(76, 175, 80, 0.9);
            color: white;
        }
        
        .threshold-control {
            margin: 20px 0;
            text-align: center;
        }
        
        .threshold-slider {
            width: 300px;
            margin: 10px;
        }
        
        #webcamModal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .modal-content {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            backdrop-filter: blur(15px);
        }
        
        #webcamVideo {
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            width: 0%;
            transition: width 0.3s ease;
            animation: progress 2s infinite;
        }
        
        @keyframes progress {
            0% { width: 0%; }
            50% { width: 70%; }
            100% { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Face Finder Local</h1>
            <p>Reconocimiento facial completamente local y privado</p>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="totalPhotos">0</div>
                    <div class="stat-label">Fotos</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalFaces">0</div>
                    <div class="stat-label">Caras</div>
                </div>
            </div>
        </div>
        
        <div class="status loading" id="status">Cargando modelos...</div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('upload')">📤 Subir</button>
            <button class="tab" onclick="showTab('search')">🔍 Buscar</button>
            <button class="tab" onclick="showTab('gallery')">🖼️ Galería</button>
            <button class="tab" onclick="showTab('results')">🎯 Resultados</button>
        </div>
        
        <!-- Tab: Upload -->
        <div id="upload" class="content">
            <h2>Subir Fotos</h2>
            <div class="upload-area" onclick="document.getElementById('photoInput').click()">
                <div style="font-size: 3rem; margin-bottom: 10px;">📷</div>
                <div style="font-size: 1.2rem; margin-bottom: 10px;">
                    Arrastra fotos aquí o haz clic para seleccionar
                </div>
                <div style="opacity: 0.7;">Formatos: JPG, PNG, GIF, WebP</div>
            </div>
            <input type="file" id="photoInput" accept="image/*" multiple style="display: none;">
            <div id="uploadProgress" class="hidden">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <p>Procesando fotos...</p>
            </div>
        </div>
        
        <!-- Tab: Search -->
        <div id="search" class="content hidden">
            <h2>Buscar Cara</h2>
            <div class="threshold-control">
                <label for="thresholdSlider">Umbral de Similitud: <span id="thresholdValue">85%</span></label>
                <br>
                <input type="range" id="thresholdSlider" class="threshold-slider" min="50" max="95" value="85">
                <br>
                <small>Menor = más coincidencias | Mayor = más precisión</small>
            </div>
            
            <button class="btn btn-secondary" onclick="openWebcam()">📷 Usar Webcam</button>
            
            <div style="text-align: center; margin: 20px 0;">O</div>
            
            <div class="upload-area" onclick="document.getElementById('searchInput').click()">
                <div style="font-size: 2rem; margin-bottom: 10px;">🖼️</div>
                <div>Subir foto para buscar</div>
            </div>
            <input type="file" id="searchInput" accept="image/*" style="display: none;">
        </div>
        
        <!-- Tab: Gallery -->
        <div id="gallery" class="content hidden">
            <h2>Galería de Fotos</h2>
            <div id="galleryContainer" class="gallery"></div>
        </div>
        
        <!-- Tab: Results -->
        <div id="results" class="content hidden">
            <h2>Resultados de Búsqueda</h2>
            <div id="resultsContainer" class="gallery"></div>
        </div>
    </div>
    
    <!-- Webcam Modal -->
    <div id="webcamModal" class="hidden">
        <div class="modal-content">
            <h3>Capturar con Webcam</h3>
            <video id="webcamVideo" width="640" height="480" autoplay></video>
            <br>
            <button class="btn" onclick="captureAndSearch()">📸 Capturar y Buscar</button>
            <button class="btn btn-danger" onclick="closeWebcam()">❌ Cerrar</button>
        </div>
    </div>

    <script>
        let currentThreshold = 0.85;
        let webcamStream = null;
        
        // Configurar eventos
        document.getElementById('photoInput').addEventListener('change', handlePhotoUpload);
        document.getElementById('searchInput').addEventListener('change', handleSearchUpload);
        document.getElementById('thresholdSlider').addEventListener('input', updateThreshold);
        
        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            uploadPhotos(files);
        });
        
        function showTab(tabName) {
            // Ocultar todas las tabs
            document.querySelectorAll('.content').forEach(tab => tab.classList.add('hidden'));
            document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
            
            // Mostrar tab seleccionada
            document.getElementById(tabName).classList.remove('hidden');
            event.target.classList.add('active');
            
            if (tabName === 'gallery') {
                loadGallery();
            }
        }
        
        function updateThreshold() {
            const slider = document.getElementById('thresholdSlider');
            currentThreshold = slider.value / 100;
            document.getElementById('thresholdValue').textContent = slider.value + '%';
        }
        
        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                const statusEl = document.getElementById('status');
                
                if (data.processor_ready) {
                    statusEl.textContent = 'Modelos cargados ✓';
                    statusEl.className = 'status ready';
                    setTimeout(() => statusEl.style.display = 'none', 3000);
                } else {
                    statusEl.textContent = 'Cargando modelos...';
                    statusEl.className = 'status loading';
                    setTimeout(checkStatus, 2000);
                }
            } catch (error) {
                console.error('Error checking status:', error);
                setTimeout(checkStatus, 5000);
            }
        }
        
        function handlePhotoUpload() {
            const files = Array.from(document.getElementById('photoInput').files);
            uploadPhotos(files);
        }
        
        async function uploadPhotos(files) {
            if (files.length === 0) return;
            
            const progressEl = document.getElementById('uploadProgress');
            progressEl.classList.remove('hidden');
            
            for (let file of files) {
                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    const response = await fetch('/api/upload-photo', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        alert(`✅ ${file.name} subida exitosamente!\n🔍 ${result.faces_detected} caras detectadas`);
                    } else {
                        alert(`❌ Error subiendo ${file.name}`);
                    }
                } catch (error) {
                    alert(`❌ Error subiendo ${file.name}: ${error.message}`);
                }
            }
            
            progressEl.classList.add('hidden');
            updateStats();
            loadGallery();
        }
        
        function handleSearchUpload() {
            const file = document.getElementById('searchInput').files[0];
            if (file) {
                searchWithFile(file);
            }
        }
        
        async function searchWithFile(file) {
            try {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch(`/api/search-face?threshold=${currentThreshold}`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displaySearchResults(result.matches);
                    showTab('results');
                    
                    if (result.matches_found === 0) {
                        alert('🤔 No se encontraron coincidencias con el umbral actual.');
                    } else {
                        alert(`✅ ¡Encontré ${result.matches_found} coincidencias!`);
                    }
                }
            } catch (error) {
                alert('❌ Error en la búsqueda: ' + error.message);
            }
        }
        
        async function loadGallery() {
            try {
                const response = await fetch('/api/photos');
                const data = await response.json();
                
                const container = document.getElementById('galleryContainer');
                
                if (data.photos.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.7;">No hay fotos subidas aún</p>';
                    return;
                }
                
                container.innerHTML = data.photos.map(photo => `
                    <div class="photo-card">
                        <img src="/api/image/photo/${photo.id}" alt="${photo.filename}">
                        <div class="photo-info">
                            <h4>${photo.filename}</h4>
                            <p>${photo.faces_count} caras detectadas</p>
                            <button class="btn btn-danger" onclick="deletePhoto('${photo.id}')">
                                🗑️ Eliminar
                            </button>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading gallery:', error);
            }
        }
        
        function displaySearchResults(matches) {
            const container = document.getElementById('resultsContainer');
            
            if (matches.length === 0) {
                container.innerHTML = '<p style="text-align: center; opacity: 0.7;">No se encontraron coincidencias</p>';
                return;
            }
            
            container.innerHTML = matches.map((match, index) => `
                <div class="photo-card">
                    <div style="position: relative;">
                        <img src="/api/image/photo/${match.photo_id}" alt="${match.photo_filename}">
                        <div style="position: absolute; top: 5px; left: 5px; background: rgba(255,152,0,0.9); padding: 5px 10px; border-radius: 15px; font-weight: bold;">
                            #${index + 1}
                        </div>
                        <div style="position: absolute; top: 5px; right: 5px; background: rgba(76,175,80,0.9); padding: 5px 10px; border-radius: 15px; font-weight: bold;">
                            ${Math.round(match.similarity * 100)}%
                        </div>
                    </div>
                    <div class="photo-info">
                        <h4>${match.photo_filename}</h4>
                        <p>Similitud: ${Math.round(match.similarity * 100)}%</p>
                    </div>
                </div>
            `).join('');
        }
        
        async function deletePhoto(photoId) {
            if (!confirm('¿Estás seguro de que quieres eliminar esta foto?')) return;
            
            try {
                const response = await fetch(`/api/photos/${photoId}`, {
                    method: 'DELETE'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('✅ Foto eliminada correctamente');
                    loadGallery();
                    updateStats();
                } else {
                    alert('❌ Error eliminando la foto');
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('totalPhotos').textContent = data.stats.total_photos;
                document.getElementById('totalFaces').textContent = data.stats.total_faces;
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        async function openWebcam() {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
                document.getElementById('webcamVideo').srcObject = webcamStream;
                document.getElementById('webcamModal').classList.remove('hidden');
            } catch (error) {
                alert('❌ Error accediendo a la webcam: ' + error.message);
            }
        }
        
        function closeWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
            }
            document.getElementById('webcamModal').classList.add('hidden');
        }
        
        async function captureAndSearch() {
            const video = document.getElementById('webcamVideo');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            // Convertir a blob
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('file', blob, 'webcam-capture.jpg');
                
                try {
                    const response = await fetch(`/api/search-face?threshold=${currentThreshold}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        displaySearchResults(result.matches);
                        showTab('results');
                        closeWebcam();
                        
                        if (result.matches_found === 0) {
                            alert('🤔 No se encontraron coincidencias.');
                        } else {
                            alert(`✅ ¡Encontré ${result.matches_found} coincidencias!`);
                        }
                    }
                } catch (error) {
                    alert('❌ Error en la búsqueda: ' + error.message);
                }
            }, 'image/jpeg');
        }
        
        // Inicializar
        checkStatus();
        updateStats();
        loadGallery();
    </script>
</body>
</html>
        """
    
    async def handle_upload_photo(self, file: UploadFile):
        """Manejar subida de foto"""
        if not self.face_processor:
            raise HTTPException(status_code=503, detail="Procesador no está listo")
        
        try:
            # Guardar archivo
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            filename = f"{file_id}.{file_extension}"
            file_path = self.photos_dir / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Procesar imagen
            image = cv2.imread(str(file_path))
            if image is None:
                raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
            
            # Detectar caras
            detected_faces = self.face_processor.detect_and_encode_faces(image)
            
            # Guardar en base de datos
            photo_id = str(uuid.uuid4())
            conn = sqlite3.connect(self.db_path)
            
            conn.execute(
                "INSERT INTO photos (id, filename, original_path, faces_count) VALUES (?, ?, ?, ?)",
                (photo_id, file.filename, str(file_path), len(detected_faces))
            )
            
            # Guardar caras
            for face_data in detected_faces:
                face_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO faces (id, photo_id, embeddings, bounding_box, confidence) VALUES (?, ?, ?, ?, ?)",
                    (face_id, photo_id, json.dumps(face_data['embeddings']),
                     json.dumps(face_data['bbox']), face_data['confidence'])
                )
            
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "photo_id": photo_id,
                "filename": file.filename,
                "faces_detected": len(detected_faces)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_search_face(self, file: UploadFile, threshold: float):
        """Manejar búsqueda de cara"""
        if not self.face_processor:
            raise HTTPException(status_code=503, detail="Procesador no está listo")
        
        try:
            # Guardar archivo temporal
            temp_path = self.upload_dir / f"temp_{uuid.uuid4()}.jpg"
            
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Procesar imagen
            image = cv2.imread(str(temp_path))
            if image is None:
                raise HTTPException(status_code=400, detail="No se pudo procesar la imagen")
            
            # Detectar caras
            detected_faces = self.face_processor.detect_and_encode_faces(image)
            
            if not detected_faces:
                return {
                    "success": True,
                    "matches_found": 0,
                    "matches": [],
                    "message": "No se detectaron caras en la imagen"
                }
            
            # Usar la primera cara detectada
            search_embeddings = detected_faces[0]['embeddings']
            
            # Buscar en base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT * FROM faces")
            faces = cursor.fetchall()
            conn.close()
            
            matches = []
            for face in faces:
                face_id, photo_id, embeddings_str, bbox_str, confidence = face
                face_embeddings = json.loads(embeddings_str)
                
                # Comparar embeddings
                similarity = self.face_processor.compare_multi_embeddings(
                    search_embeddings, face_embeddings
                )
                
                if similarity >= threshold:
                    # Obtener información de la foto
                    conn = sqlite3.connect(self.db_path)
                    photo_cursor = conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
                    photo_row = photo_cursor.fetchone()
                    conn.close()
                    
                    if photo_row:
                        matches.append({
                            "photo_id": photo_id,
                            "photo_filename": photo_row[1],  # filename
                            "face_id": face_id,
                            "similarity": similarity
                        })
            
            # Ordenar por similitud
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limpiar archivo temporal
            if temp_path.exists():
                temp_path.unlink()
            
            return {
                "success": True,
                "matches_found": len(matches),
                "matches": matches
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_get_photos(self):
        """Obtener todas las fotos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT * FROM photos ORDER BY upload_date DESC")
            photos = []
            
            for row in cursor.fetchall():
                photos.append({
                    "id": row[0],
                    "filename": row[1],
                    "original_path": row[2],
                    "faces_count": row[3],
                    "upload_date": row[4]
                })
            
            conn.close()
            
            return {
                "success": True,
                "photos": photos
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_get_stats(self):
        """Obtener estadísticas"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            photos_cursor = conn.execute("SELECT COUNT(*) FROM photos")
            total_photos = photos_cursor.fetchone()[0]
            
            faces_cursor = conn.execute("SELECT COUNT(*) FROM faces")
            total_faces = faces_cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "success": True,
                "stats": {
                    "total_photos": total_photos,
                    "total_faces": total_faces
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_delete_photo(self, photo_id: str):
        """Eliminar foto"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Obtener información de la foto
            cursor = conn.execute("SELECT original_path FROM photos WHERE id = ?", (photo_id,))
            photo_row = cursor.fetchone()
            
            if not photo_row:
                raise HTTPException(status_code=404, detail="Foto no encontrada")
            
            # Eliminar archivo
            photo_path = Path(photo_row[0])
            if photo_path.exists():
                photo_path.unlink()
            
            # Eliminar registros
            conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
            conn.commit()
            conn.close()
            
            return {"success": True}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_get_image(self, image_type: str, image_id: str):
        """Servir imágenes"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if image_type == "photo":
                cursor = conn.execute("SELECT original_path FROM photos WHERE id = ?", (image_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Foto no encontrada")
                image_path = row[0]
            else:
                raise HTTPException(status_code=400, detail="Tipo de imagen inválido")
            
            conn.close()
            
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Archivo no encontrado")
            
            return FileResponse(image_path)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """Ejecutar la aplicación"""
        print(f"🚀 Iniciando Face Finder Local en puerto {self.port}")
        print(f"📱 La aplicación se abrirá automáticamente en tu navegador")
        print(f"🔗 URL: http://localhost:{self.port}")
        
        # Abrir navegador automáticamente
        if self.auto_open:
            threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{self.port}")).start()
        
        # Configurar signal handler para cierre limpio
        def signal_handler(sig, frame):
            print("\n🛑 Cerrando aplicación...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Ejecutar servidor
        try:
            uvicorn.run(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
                access_log=False
            )
        except Exception as e:
            print(f"❌ Error ejecutando la aplicación: {e}")
            sys.exit(1)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Finder Local - Reconocimiento Facial Local")
    parser.add_argument("--port", type=int, default=8080, help="Puerto del servidor (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="No abrir navegador automáticamente")
    
    args = parser.parse_args()
    
    # Crear y ejecutar aplicación
    app = LocalFaceRecognitionApp(
        port=args.port,
        auto_open=not args.no_browser
    )
    
    app.run()

if __name__ == "__main__":
    main()
