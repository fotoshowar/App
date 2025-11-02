# test_app.py
import pytest
import os
import sys
import configparser
from fastapi.testclient import TestClient
from main import app  # Importa tu aplicación FastAPI

# --- Prueba 1: Verificar que el archivo de configuración se carga ---
def test_load_config():
    """Verifica que se puede leer el archivo config.ini."""
    # Simula cómo PyInstaller encuentra el archivo
    if getattr(sys, 'frozen', False):
        config_path = os.path.join(sys._MEIPASS, 'config.ini')
    else:
        config_path = 'config.ini'
    
    assert os.path.exists(config_path), f"El archivo de configuración no se encontró en {config_path}"
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Verifica que las claves importantes existan
    assert config.has_option('server', 'port'), "La opción 'port' no se encontró en la sección 'server'"
    assert config.has_option('general', 'base_url'), "La opción 'base_url' no se encontró en la sección 'general'"
    assert config.getint('server', 'port') == 8888, "El puerto en config.ini no es 8888"

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
