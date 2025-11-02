import sys
import traceback

print("Probando importación de AdvancedFaceProcessor...")
print("=" * 60)

try:
    # Probar importaciones individuales primero
    print("\n1. Probando cv2...")
    import cv2
    print(f"   ✓ cv2 version: {cv2.__version__}")
    
    print("\n2. Probando numpy...")
    import numpy as np
    print(f"   ✓ numpy version: {np.__version__}")
    
    print("\n3. Probando face_recognition...")
    import face_recognition
    print("   ✓ face_recognition OK")
    
    print("\n4. Probando torch...")
    import torch
    print(f"   ✓ torch version: {torch.__version__}")
    print(f"   CUDA disponible: {torch.cuda.is_available()}")
    
    print("\n5. Probando torchvision...")
    import torchvision.transforms as transforms
    print("   ✓ torchvision OK")
    
    print("\n6. Probando facenet_pytorch...")
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("   ✓ facenet_pytorch OK")
    
    print("\n7. Probando dlib...")
    import dlib
    print(f"   ✓ dlib version: {dlib.__version__ if hasattr(dlib, '__version__') else 'installed'}")
    
    print("\n8. Probando scipy...")
    from scipy.spatial.distance import cosine
    print("   ✓ scipy OK")
    
    print("\n9. Probando sklearn...")
    from sklearn.metrics.pairwise import cosine_similarity
    print("   ✓ sklearn OK")
    
    print("\n10. Probando insightface...")
    import insightface
    print(f"   ✓ insightface version: {insightface.__version__ if hasattr(insightface, '__version__') else 'installed'}")
    
    print("\n" + "=" * 60)
    print("TODAS LAS DEPENDENCIAS OK - Ahora probando importación completa...")
    print("=" * 60 + "\n")
    
    # Ahora probar la importación completa
    from advanced_face_processor import AdvancedFaceProcessor
    print("✓✓✓ AdvancedFaceProcessor importado EXITOSAMENTE ✓✓✓")
    
    # Intentar instanciar
    print("\nIntentando instanciar AdvancedFaceProcessor...")
    processor = AdvancedFaceProcessor(device='cpu')
    print("✓✓✓ AdvancedFaceProcessor instanciado EXITOSAMENTE ✓✓✓")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ ERROR ENCONTRADO:")
    print("=" * 60)
    print(f"Tipo: {type(e).__name__}")
    print(f"Mensaje: {str(e)}")
    print("\nTraceback completo:")
    print("-" * 60)
    traceback.print_exc()
    print("=" * 60)
