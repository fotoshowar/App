# test_app.py
import pytest
import os
import sys
import configparser
from fastapi.testclient import TestClient
from main import app  # Importa tu aplicación FastAPI

  # --- Prueba 2: Verificar la inicialización del procesador facial ---
def test_advanced_face_processor_initialization():
    """Verifica que el AdvancedFaceProcessor puede inicializar sus rutas."""
    # Importamos aquí para evitar problemas de inicialización temprana
    from advanced_face_processor import AdvancedFaceProcessor
    
    # Esto no debería lanzar una excepción
    processor = AdvancedFaceProcessor(device='cpu')
    
    # Verifica que el directorio de modelos se haya creado y sea correcto
    assert processor.models_dir is not None, "El directorio de modelos no fue inicializado"
    assert os.path.isdir(processor.models_dir), f"El directorio de modelos no existe en {processor.models_dir}"

# --- Prueba 3: Verificar que la API FastAPI se inicia ---
def test_fastapi_app_startup():
    """Verifica que la aplicación FastAPI se puede crear y tiene los endpoints."""
    # El cliente de prueba permite hacer "peticiones" falsas a tu API sin necesidad de un servidor real
    client = TestClient(app)
    
    # Verifica que el endpoint principal responde
    response = client.get("/")
    assert response.status_code == 200, "El endpoint principal '/' no responde con estado 200"
    
    # Puedes añadir más pruebas para otros endpoints
    # response = client.get("/admin")
    # assert response.status_code == 200, "El endpoint '/admin' no responde con estado 200"
