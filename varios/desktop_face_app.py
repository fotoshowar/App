import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from pathlib import Path
import sqlite3
import json
import numpy as np
from datetime import datetime
import uuid

# Importar tu procesador avanzado
from advanced_face_processor import AdvancedFaceProcessor

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Finder Desktop")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Inicializar procesador
        self.face_processor = None
        self.init_processor()
        
        # Base de datos
        self.db_path = "face_recognition.db"
        self.init_database()
        
        # Variables
        self.photos = []
        self.current_search_results = []
        
        self.setup_ui()
        self.load_photos()
        
    def init_processor(self):
        """Inicializar procesador en hilo separado"""
        def load_processor():
            try:
                self.face_processor = AdvancedFaceProcessor(device='cpu')
                self.update_status("Procesador de caras cargado correctamente")
            except Exception as e:
                self.update_status(f"Error cargando procesador: {e}")
        
        self.update_status("Cargando modelos de reconocimiento facial...")
        thread = threading.Thread(target=load_processor)
        thread.daemon = True
        thread.start()
    
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
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Custom.TButton', font=('Arial', 10))
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(main_frame, text="🔍 Face Finder Desktop", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Frame de botones principales
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(fill='x', pady=(0, 20))
        
        # Botones principales
        self.upload_btn = tk.Button(button_frame, text="📤 Subir Fotos", 
                                   command=self.upload_photos, bg='#4CAF50', fg='white',
                                   font=('Arial', 12), padx=20, pady=10)
        self.upload_btn.pack(side='left', padx=(0, 10))
        
        self.search_btn = tk.Button(button_frame, text="🔍 Buscar Cara", 
                                   command=self.search_face, bg='#2196F3', fg='white',
                                   font=('Arial', 12), padx=20, pady=10)
        self.search_btn.pack(side='left', padx=(0, 10))
        
        self.webcam_btn = tk.Button(button_frame, text="📷 Usar Webcam", 
                                   command=self.open_webcam, bg='#FF9800', fg='white',
                                   font=('Arial', 12), padx=20, pady=10)
        self.webcam_btn.pack(side='left', padx=(0, 10))
        
        # Slider de umbral
        threshold_frame = tk.Frame(main_frame, bg='#2b2b2b')
        threshold_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(threshold_frame, text="Umbral de Similitud:", 
                bg='#2b2b2b', fg='white', font=('Arial', 10)).pack(side='left')
        
        self.threshold_var = tk.DoubleVar(value=0.85)
        self.threshold_scale = tk.Scale(threshold_frame, from_=0.5, to=0.95, 
                                       resolution=0.05, orient='horizontal',
                                       variable=self.threshold_var, bg='#2b2b2b', fg='white')
        self.threshold_scale.pack(side='left', padx=10)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Pestaña de galería
        self.gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gallery_frame, text="Galería")
        self.setup_gallery_tab()
        
        # Pestaña de resultados
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Resultados")
        self.setup_results_tab()
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                             relief='sunken', anchor='w', bg='#404040', fg='white')
        status_bar.pack(side='bottom', fill='x', pady=(10, 0))
    
    def setup_gallery_tab(self):
        """Configurar pestaña de galería"""
        # Frame con scrollbar
        canvas = tk.Canvas(self.gallery_frame, bg='#2b2b2b')
        scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=canvas.yview)
        self.gallery_scroll_frame = ttk.Frame(canvas)
        
        self.gallery_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.gallery_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_results_tab(self):
        """Configurar pestaña de resultados"""
        canvas = tk.Canvas(self.results_frame, bg='#2b2b2b')
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        self.results_scroll_frame = ttk.Frame(canvas)
        
        self.results_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def update_status(self, message):
        """Actualizar barra de estado"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def upload_photos(self):
        """Subir y procesar fotos"""
        if not self.face_processor:
            messagebox.showerror("Error", "El procesador de caras aún no está listo")
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Seleccionar Fotos",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.gif *.webp")]
        )
        
        if not file_paths:
            return
        
        def process_photos():
            for i, file_path in enumerate(file_paths):
                self.update_status(f"Procesando {i+1}/{len(file_paths)}: {Path(file_path).name}")
                self.process_single_photo(file_path)
            
            self.update_status("Fotos procesadas exitosamente")
            self.load_photos()
        
        thread = threading.Thread(target=process_photos)
        thread.daemon = True
        thread.start()
    
    def process_single_photo(self, file_path):
        """Procesar una sola foto"""
        try:
            # Cargar imagen
            image = cv2.imread(file_path)
            if image is None:
                return
            
            # Detectar caras
            detected_faces = self.face_processor.detect_and_encode_faces(image)
            
            # Guardar en base de datos
            photo_id = str(uuid.uuid4())
            conn = sqlite3.connect(self.db_path)
            
            conn.execute(
                "INSERT INTO photos (id, filename, original_path, faces_count) VALUES (?, ?, ?, ?)",
                (photo_id, Path(file_path).name, file_path, len(detected_faces))
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
            
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")
    
    def load_photos(self):
        """Cargar fotos en la galería"""
        # Limpiar galería actual
        for widget in self.gallery_scroll_frame.winfo_children():
            widget.destroy()
        
        # Obtener fotos de la base de datos
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM photos ORDER BY upload_date DESC")
        photos = cursor.fetchall()
        conn.close()
        
        # Mostrar fotos en grid
        row, col = 0, 0
        cols = 3
        
        for photo in photos:
            photo_id, filename, path, faces_count, upload_date = photo
            
            # Frame para foto
            photo_frame = tk.Frame(self.gallery_scroll_frame, bg='#404040', relief='raised', bd=1)
            photo_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            try:
                # Cargar y redimensionar imagen
                img = Image.open(path)
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(img)
                
                # Mostrar imagen
                img_label = tk.Label(photo_frame, image=photo_img, bg='#404040')
                img_label.image = photo_img  # Mantener referencia
                img_label.pack(pady=5)
                
            except Exception as e:
                # Imagen placeholder si hay error
                placeholder = tk.Label(photo_frame, text="📷", font=('Arial', 40), 
                                     bg='#404040', fg='white')
                placeholder.pack(pady=20)
            
            # Información
            info_label = tk.Label(photo_frame, text=f"{filename}\n{faces_count} caras", 
                                 bg='#404040', fg='white', font=('Arial', 9))
            info_label.pack()
            
            # Botón eliminar
            delete_btn = tk.Button(photo_frame, text="Eliminar", 
                                 command=lambda pid=photo_id: self.delete_photo(pid),
                                 bg='#f44336', fg='white', font=('Arial', 8))
            delete_btn.pack(pady=2)
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
    
    def search_face(self):
        """Buscar cara desde archivo"""
        if not self.face_processor:
            messagebox.showerror("Error", "El procesador de caras aún no está listo")
            return
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar foto para buscar",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.gif *.webp")]
        )
        
        if not file_path:
            return
        
        def search():
            self.update_status("Buscando cara...")
            self.perform_search(file_path)
        
        thread = threading.Thread(target=search)
        thread.daemon = True
        thread.start()
    
    def open_webcam(self):
        """Abrir ventana de webcam"""
        webcam_window = tk.Toplevel(self.root)
        webcam_window.title("Webcam - Face Search")
        webcam_window.geometry("640x580")
        webcam_window.configure(bg='#2b2b2b')
        
        # Label para video
        video_label = tk.Label(webcam_window, bg='#2b2b2b')
        video_label.pack(pady=10)
        
        # Botones
        btn_frame = tk.Frame(webcam_window, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        capture_btn = tk.Button(btn_frame, text="Capturar y Buscar", 
                               bg='#4CAF50', fg='white', font=('Arial', 12))
        capture_btn.pack(side='left', padx=5)
        
        close_btn = tk.Button(btn_frame, text="Cerrar", 
                             command=webcam_window.destroy,
                             bg='#f44336', fg='white', font=('Arial', 12))
        close_btn.pack(side='left', padx=5)
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        
        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                video_label.configure(image=photo)
                video_label.image = photo
            
            if webcam_window.winfo_exists():
                webcam_window.after(10, update_frame)
        
        def capture_and_search():
            ret, frame = cap.read()
            if ret:
                # Guardar frame temporalmente
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Buscar
                def search():
                    self.perform_search(temp_path)
                    # Limpiar archivo temporal
                    if Path(temp_path).exists():
                        Path(temp_path).unlink()
                
                thread = threading.Thread(target=search)
                thread.daemon = True
                thread.start()
                
                webcam_window.destroy()
        
        capture_btn.configure(command=capture_and_search)
        
        def on_close():
            cap.release()
            webcam_window.destroy()
        
        webcam_window.protocol("WM_DELETE_WINDOW", on_close)
        update_frame()
    
    def perform_search(self, image_path):
        """Realizar búsqueda de cara"""
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", "No se pudo cargar la imagen")
                return
            
            # Detectar caras
            detected_faces = self.face_processor.detect_and_encode_faces(image)
            
            if not detected_faces:
                messagebox.showinfo("Sin resultados", "No se detectaron caras en la imagen")
                return
            
            # Usar la primera cara detectada
            search_embeddings = detected_faces[0]['embeddings']
            threshold = self.threshold_var.get()
            
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
                    matches.append({
                        'photo_id': photo_id,
                        'face_id': face_id,
                        'similarity': similarity
                    })
            
            # Ordenar por similitud
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.current_search_results = matches
            self.display_search_results()
            
            # Cambiar a pestaña de resultados
            self.notebook.select(1)
            
            self.update_status(f"Encontradas {len(matches)} coincidencias")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la búsqueda: {e}")
            self.update_status("Error en búsqueda")
    
    def display_search_results(self):
        """Mostrar resultados de búsqueda"""
        # Limpiar resultados anteriores
        for widget in self.results_scroll_frame.winfo_children():
            widget.destroy()
        
        if not self.current_search_results:
            no_results = tk.Label(self.results_scroll_frame, 
                                 text="No se encontraron coincidencias", 
                                 bg='#2b2b2b', fg='white', font=('Arial', 14))
            no_results.pack(pady=50)
            return
        
        # Obtener información de fotos
        photo_info = {}
        conn = sqlite3.connect(self.db_path)
        for match in self.current_search_results:
            cursor = conn.execute("SELECT * FROM photos WHERE id = ?", (match['photo_id'],))
            photo = cursor.fetchone()
            if photo:
                photo_info[match['photo_id']] = photo
        conn.close()
        
        # Mostrar resultados
        row, col = 0, 0
        cols = 3
        
        for i, match in enumerate(self.current_search_results[:20]):  # Limitar a 20 resultados
            photo_id = match['photo_id']
            similarity = match['similarity']
            
            if photo_id not in photo_info:
                continue
            
            photo = photo_info[photo_id]
            _, filename, path, _, _ = photo
            
            # Frame para resultado
            result_frame = tk.Frame(self.results_scroll_frame, bg='#404040', relief='raised', bd=1)
            result_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
            
            # Ranking
            rank_label = tk.Label(result_frame, text=f"#{i+1}", 
                                 bg='#FF9800', fg='white', font=('Arial', 12, 'bold'))
            rank_label.pack()
            
            try:
                # Imagen
                img = Image.open(path)
                img.thumbnail((180, 180), Image.Resampling.LANCZOS)
                photo_img = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(result_frame, image=photo_img, bg='#404040')
                img_label.image = photo_img
                img_label.pack(pady=5)
                
            except Exception as e:
                placeholder = tk.Label(result_frame, text="📷", font=('Arial', 30), 
                                     bg='#404040', fg='white')
                placeholder.pack(pady=20)
            
            # Información
            info_text = f"{filename}\nSimilitud: {similarity:.1%}"
            info_label = tk.Label(result_frame, text=info_text, 
                                 bg='#404040', fg='white', font=('Arial', 9))
            info_label.pack()
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
    
    def delete_photo(self, photo_id):
        """Eliminar foto"""
        if messagebox.askyesno("Confirmar", "¿Eliminar esta foto?"):
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
            conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
            conn.commit()
            conn.close()
            
            self.load_photos()
            self.update_status("Foto eliminada")

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
