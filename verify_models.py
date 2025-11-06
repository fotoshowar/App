import sys
import os

def check_models():
    """Verifica que los modelos estén disponibles en el ejecutable"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        models_path = os.path.join(base_path, "models")
        
        print(f"Base path: {base_path}")
        print(f"Models path: {models_path}")
        
        if os.path.exists(models_path):
            print("Models directory found!")
            for file in os.listdir(models_path):
                if file.endswith('.dat'):
                    print(f"✅ Model found: {file}")
        else:
            print("❌ Models directory not found!")
            return False
    else:
        # En modo desarrollo
        models_path = os.path.join(os.path.dirname(__file__), "models")
        print(f"Development mode - Models path: {models_path}")
        
        if os.path.exists(models_path):
            print("Models directory found in development!")
            for file in os.listdir(models_path):
                if file.endswith('.dat'):
                    print(f"✅ Model found: {file}")
        else:
            print("❌ Models directory not found in development!")
            return False
    
    return True

if __name__ == "__main__":
    check_models()
