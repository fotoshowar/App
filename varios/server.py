from fastapi import FastAPI, File, UploadFile, HTTPException, Request
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
import asyncio
import logging
import traceback
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('photo_manager.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Photo Manager API", version="1.0.0")

# CORS mejorado para ngrok
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
    # Headers para ngrok
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept, ngrok-skip-browser-warning"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response

# Directorios
UPLOAD_DIR = Path("uploads")
DB_FILE = Path("photo_database.json")

for directory in [UPLOAD_DIR]:
    directory.mkdir(exist_ok=True)

class PhotoDatabase:
    def __init__(self):
        self.db_file = DB_FILE
        self.data = self.load_database()
    
    def load_database(self) -> Dict:
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data.setdefault('photos', {})
                logger.info(f"BD cargada: {len(data['photos'])} fotos")
                return data
            except Exception as e:
                logger.error(f"Error cargando BD: {e}")
        return {'photos': {}}
    
    def save_database(self):
        try:
            if self.db_file.exists():
                backup_path = self.db_file.with_suffix('.json.backup')
                shutil.copy2(self.db_file, backup_path)
            
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando BD: {e}")
            raise
    
    def add_photo(self, photo_id: str, filename: str, filepath: str):
        try:
            img = cv2.imread(filepath)
            image_height, image_width = img.shape[:2] if img is not None else (0, 0)
            file_size = Path(filepath).stat().st_size
            
            photo_data = {
                'id': photo_id,
                'filename': filename,
                'filepath': filepath,
                'upload_date': datetime.now().isoformat(),
                'image_width': image_width,
                'image_height': image_height,
                'file_size': file_size
            }
            
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
    
    def get_stats(self) -> Dict:
        return {
            'total_photos': len(self.data['photos'])
        }
    
    def delete_photo(self, photo_id: str) -> Dict:
        try:
            if photo_id not in self.data['photos']:
                return {
                    'success': False, 
                    'error': 'Foto no encontrada'
                }
            
            photo_data = self.data['photos'][photo_id]
            
            # Eliminar archivo
            try:
                photo_filepath = Path(photo_data['filepath'])
                if photo_filepath.exists():
                    photo_filepath.unlink()
                    logger.info(f"Archivo eliminado: {photo_data['filepath']}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo: {e}")
            
            # Eliminar de la base de datos
            del self.data['photos'][photo_id]
            self.save_database()
            
            logger.info(f"Foto {photo_id} eliminada")
            
            return {
                'success': True,
                'message': 'Foto eliminada correctamente'
            }
            
        except Exception as e:
            logger.error(f"Error eliminando foto: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_photo_dimensions(self, photo_id: str, width: int, height: int):
        try:
            if photo_id in self.data['photos']:
                self.data['photos'][photo_id]['image_width'] = width
                self.data['photos'][photo_id]['image_height'] = height
                self.data['photos'][photo_id]['last_modified'] = datetime.now().isoformat()
                self.save_database()
                return True
            return False
        except Exception as e:
            logger.error(f"Error actualizando dimensiones: {e}")
            return False

# Instancia global
database = PhotoDatabase()

# Servir HTML desde raiz
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        html_file = Path("index.html")
        
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            logger.info("Sirviendo HTML desde raiz")
            return HTMLResponse(content=html_content)
        else:
            error_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Photo Manager - Error</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; text-align: center; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1 class="error">Archivo HTML no encontrado</h1>
    <p>No se encontro: index.html</p>
    <p>Coloca el archivo App.js o index.html en la raiz del proyecto</p>
    <ul>
        <li><a href="/docs">Documentacion API</a></li>
        <li><a href="/api-status">Estado API</a></li>
    </ul>
</body>
</html>"""
            return HTMLResponse(content=error_html, status_code=404)
    except Exception as e:
        logger.error(f"Error sirviendo HTML: {e}")
        return HTMLResponse(content="<h1>Error del servidor</h1>", status_code=500)

@app.get("/api-status")
async def api_status():
    return {
        "message": "Photo Manager API v1.0",
        "status": "running",
        "database": database.get_stats()
    }

@app.post("/api/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Debe ser una imagen")
        
        photo_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename or "image.jpg")[1].lower()
        if not file_extension:
            file_extension = ".jpg"
        
        filename = f"{photo_id}{file_extension}"
        filepath = UPLOAD_DIR / filename
        
        content = await file.read()
        with open(filepath, "wb") as buffer:
            buffer.write(content)
        
        database.add_photo(photo_id, file.filename or "unknown.jpg", str(filepath))
        
        return {
            "success": True,
            "message": "Foto subida correctamente",
            "photo_id": photo_id
        }
    except Exception as e:
        logger.error(f"Error upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/photos")
async def get_photos():
    return {"success": True, "photos": database.get_all_photos()}

@app.get("/api/image/photo/{photo_id}")
async def get_photo_image(photo_id: str):
    photo = database.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Foto no encontrada")
    
    filepath = Path(photo['filepath'])
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return FileResponse(path=filepath, media_type="image/jpeg")

@app.post("/api/photos/{photo_id}/rotate")
async def rotate_photo(photo_id: str):
    try:
        photo = database.get_photo(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Foto no encontrada")
        
        photo_filepath = Path(photo['filepath'])
        if not photo_filepath.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        img = cv2.imread(str(photo_filepath))
        if img is None:
            raise HTTPException(status_code=500, detail="No se pudo leer la imagen")
        
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(photo_filepath), rotated_img)
        
        height, width = rotated_img.shape[:2]
        database.update_photo_dimensions(photo_id, width, height)
        
        logger.info(f"Foto {photo_id} rotada correctamente")
        
        return {
            "success": True,
            "message": "Foto rotada correctamente",
            "photo_id": photo_id,
            "new_width": width,
            "new_height": height
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rotando foto {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error rotando foto: {str(e)}")

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    try:
        result = database.delete_photo(photo_id)
        
        if not result['success']:
            raise HTTPException(status_code=404, detail=result.get('error', 'Error eliminando foto'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando foto {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando foto: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    print("Photo Manager API v1.0")
    print("Servidor: http://localhost:8000")
    print("Para ngrok: ngrok http 8000 --host-header=localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
