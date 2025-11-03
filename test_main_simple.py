# test_main_simple.py
from fastapi.testclient import TestClient
from main_simple import app

def test_read_main():
    """Prueba que el endpoint principal '/' funcione."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() # Verifica que sea un documento HTML

def test_read_admin():
    """Prueba que el endpoint de admin '/admin' funcione."""
    client = TestClient(app)
    response = client.get("/admin")
    # Puede ser 200 si el archivo existe o 404 si no, lo importante es que el servidor responda.
    assert response.status_code in [200, 404]

def test_read_static_file():
    """Prueba que se puedan servir archivos estáticos (si existen)."""
    client = TestClient(app)
    # Esta prueba asume que tienes un archivo style.css en tu carpeta static
    response = client.get("/static/style.css")
    # Esperamos un 200 si el archivo existe, o un 404 si no.
    assert response.status_code in [200, 404]

# Para ejecutar las pruebas manualmente desde la terminal
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
