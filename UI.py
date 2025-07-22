# ----------------------------
# MÓDULO: ui_d.py -
# ArgSign V1.2.5 
# ----------------------------
# Importación de librerías necesarias
import threading  # Para ejecución concurrente
import time       # Para manejo de tiempos y delays
import shutil     # Para operaciones de archivos (copiar/eliminar)
import cv2        # OpenCV para procesamiento de imágenes
import numpy as np  # Matemáticas y operaciones con arrays
import tkinter as tk  # Para la interfaz gráfica
from tkinter import simpledialog, messagebox, ttk  # Componentes de UI de Tkinter
from pathlib import Path  # Manejo de rutas de archivos
from PIL import Image, ImageTk  # Manipulación de imágenes para Tkinter
import json  # Para manejo de datos en formato JSON

# Importación de módulos personalizados
from colector_modulo import Colector       # Captura de video y detección de manos
from dataset_modulo import GestorDataset   # Gestión del dataset de señas
from entrenador_modulo import Entrenador   # Entrenamiento del modelo
from traductor_modulo import Traductor     # Traducción en tiempo real

class Interfaz:
    def __init__(self, root: tk.Tk):
        # Inicialización de la interfaz principal
        self.root = root
        self.root.title("ArgSign V1.2.5")  # Título de la ventana
        # Configurar acción al cerrar la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Módulos de la aplicación
        self.colector_captura = None  # Instancia para modo captura
        self.traductor = None         # Instancia para modo traducción
        self.gestor = GestorDataset() # Gestor del dataset
        self.entrenador = Entrenador()  # Entrenador del modelo

        # Estado de la seña actual
        self.sign_active = False  # Indica si hay una seña en proceso
        self.current_sign = None  # Nombre de la seña actual
        self.metadata = {}        # Metadatos de la seña actual

        # Flags de operación
        self.is_capturing = False   # En proceso de captura automática
        self.is_recording = False   # En proceso de grabación de video
        self.is_training = False    # En proceso de entrenamiento
        self.is_translating = False # En modo traducción

        # Construir la interfaz y actualizar botones
        self.build_ui()
        self.update_buttons()
        
        # Mostrar mensaje inicial en el canvas
        self.show_message("PRESIONE 'NUEVA SEÑA' PARA COMENZAR")

    def build_ui(self):
        # Canvas para mostrar el video
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=2, pady=10)

        # Etiqueta para mostrar traducciones
        self.lbl_trad = tk.Label(self.root, text="------", font=("Helvetica", 16, "bold"))
        self.lbl_trad.grid(row=1, column=0, columnspan=2)

        # Barra de progreso para operaciones largas
        self.barra = ttk.Progressbar(self.root, length=300, mode="indeterminate")
        self.barra.grid(row=2, column=0, columnspan=2, pady=5)
        self.barra.grid_remove()  # Ocultar inicialmente
        
        # Etiqueta de estado
        self.lbl_estado = tk.Label(self.root, text="", font=("Helvetica", 12))
        self.lbl_estado.grid(row=3, column=0, columnspan=2)
        self.lbl_estado.grid_remove()  # Ocultar inicialmente

        # Botones de la interfaz
        self.btn_new = tk.Button(self.root, text="Nueva seña", command=self.on_new, width=20)
        self.btn_discard = tk.Button(self.root, text="Descartar seña", command=self.on_discard, width=20)
        self.btn_capture = tk.Button(self.root, text="Captura automática", command=self.on_capture, width=20)
        self.btn_record = tk.Button(self.root, text="Grabar video", command=self.on_record, width=20)
        self.btn_final = tk.Button(self.root, text="Finalizar seña", command=self.on_finalize, width=20)
        self.btn_train = tk.Button(self.root, text="Entrenar modelo", command=self.on_train, width=20)
        # Botón para traducción (ocupa dos columnas)
        self.btn_traduc = tk.Button(self.root, text="Traducir en tiempo real", 
                                    command=self.on_translate, width=42)

        # Posicionamiento de botones en la grilla
        self.btn_new.grid(row=4, column=0, padx=5, pady=5)
        self.btn_discard.grid(row=4, column=1, padx=5, pady=5)
        self.btn_capture.grid(row=5, column=0, padx=5, pady=5)
        self.btn_record.grid(row=5, column=1, padx=5, pady=5)
        self.btn_final.grid(row=6, column=0, padx=5, pady=5)
        self.btn_train.grid(row=6, column=1, padx=5, pady=5)
        self.btn_traduc.grid(row=7, column=0, columnspan=2, pady=10)

    def show_message(self, text):
        """Muestra un mensaje centrado en el canvas"""
        self.canvas.delete("all")  # Limpiar canvas
        # Crear texto centrado
        self.canvas.create_text(320, 240, text=text, 
                               fill="white", font=("Helvetica", 14), tags="msg")

    def update_buttons(self):
        """Actualiza el estado de los botones según las operaciones en curso"""
        # Deshabilitar botones si hay operaciones en curso
        busy = self.is_capturing or self.is_recording or self.is_training
        for b in (self.btn_new, self.btn_discard, self.btn_capture, 
                  self.btn_record, self.btn_final, self.btn_train, self.btn_traduc):
            b.config(state="disabled" if busy else "normal")
        
        # Actualizar texto del botón de traducción
        if self.is_translating:
            self.btn_traduc.config(text="Detener traducción")
        else:
            self.btn_traduc.config(text="Traducir en tiempo real",
                                  state="normal" if Path("argsign_model.keras").exists() else "disabled")
            
        # Habilitar botones solo si hay una seña activa
        self.btn_discard.config(state="normal" if self.sign_active else "disabled")
        self.btn_capture.config(state="normal" if self.sign_active else "disabled")
        self.btn_record.config(state="normal" if self.sign_active else "disabled")
        self.btn_final.config(state="normal" if self.sign_active else "disabled")

    # ---- Gestión de cámara ----
    def start_capture_camera(self):
        """Inicia la cámara para el modo de captura"""
        if self.colector_captura is None:
            self.colector_captura = Colector()  # Crear instancia de captura
            self.preview_loop()  # Iniciar bucle de vista previa

    def stop_all_cameras(self):
        """Detiene todas las cámaras activas y libera recursos"""
        if self.colector_captura:
            self.colector_captura.liberar()  # Liberar cámara de captura
            self.colector_captura = None
            
        if self.traductor:
            # Liberar cámara del módulo de traducción
            self.traductor.colector.liberar()
            self.traductor = None
            
        self.show_message("CÁMARA DETENIDA")  # Mostrar mensaje

    def preview_loop(self):
        """Bucle principal para mostrar la vista previa de la cámara"""
        if self.colector_captura and not self.is_translating:
            # Obtener frame procesado y sin procesar
            raw, annot, _ = self.colector_captura.obtener_frame()
            if annot is not None:
                # Convertir imagen anotada a formato Tkinter
                img = Image.fromarray(annot)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Actualizar el canvas con la nueva imagen
                self.canvas.delete("all")
                self.img_id = self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
                self.canvas.imgtk = imgtk  # Mantener referencia
                
                # Guardar último frame sin procesar para captura
                self.last_raw = raw
                
            # Programar siguiente actualización (≈30 fps)
            self.root.after(33, self.preview_loop)

    # ---- Funciones de captura ----
    def on_new(self):
        """Crea una nueva seña y configura sus metadatos"""
        self.stop_all_cameras()  # Asegurar que no hay cámaras activas
        
        # Solicitar nombre de la seña
        nombre = simpledialog.askstring("Nueva seña", "Nombre de la seña:", parent=self.root)
        if not nombre:  # Si se cancela
            return
            
        # Crear clave única para la seña
        clave = nombre.strip().replace(" ", "_").lower()
        
        # Preguntar características de la seña
        bimanual = messagebox.askyesno("Bimanual?", "¿Requiere ambas manos?", parent=self.root)
        dinamica = messagebox.askyesno("Dinámica?", "¿Requiere movimiento?", parent=self.root)
        
        try:
            # Crear carpeta para la seña con sus metadatos
            self.gestor.crear_carpeta_senia(clave, {"bimanual": bimanual, "dinamica": dinamica})
        except FileExistsError as e:
            messagebox.showwarning("Atención", str(e), parent=self.root)
            return
            
        # Actualizar estado de la aplicación
        self.sign_active = True
        self.current_sign = clave
        self.metadata = {"bimanual": bimanual, "dinamica": dinamica}
        self.start_capture_camera()  # Iniciar cámara
        self.update_buttons()

    def on_discard(self):
        """Descarta la seña actual y elimina sus datos"""
        if not self.sign_active:
            return
            
        # Confirmar eliminación
        if messagebox.askyesno("Descartar", "¿Eliminar todos los datos de esta seña?", parent=self.root):
            # Eliminar carpeta de la seña
            shutil.rmtree(self.gestor.carpeta_salida / self.current_sign, ignore_errors=True)
            # Resetear estado
            self.sign_active = False
            self.current_sign = None
            self.update_buttons()
            self.stop_all_cameras()
            self.show_message("SEÑA DESCARTADA")

    def on_capture(self):
        """Inicia el proceso de captura automática de imágenes"""
        if not self.sign_active or self.is_capturing:
            return
            
        # Solicitar cantidad de imágenes a capturar
        total = simpledialog.askinteger(
            "Cantidad", "¿Cuántas imágenes capturar?",
            initialvalue=50, minvalue=10, maxvalue=1000, parent=self.root
        )
        if not total:  # Si se cancela
            return
            
        # Configurar UI para captura
        self.is_capturing = True
        self.barra.config(mode="determinate", maximum=total, value=0)
        self.barra.grid()  # Mostrar barra
        self.lbl_estado.config(text=f"Capturando {total} imágenes...")
        self.lbl_estado.grid()  # Mostrar etiqueta
        self.update_buttons()
        
        # Iniciar hilo para captura en segundo plano
        threading.Thread(target=self._capture_worker, args=(total,), daemon=True).start()

    def _capture_worker(self, total):
        """Hilo de trabajo para captura automática de imágenes"""
        capturadas = 0
        # Determinar requerimiento de manos según metadatos
        req_manos = 2 if self.metadata["bimanual"] else 1
        
        while capturadas < total and self.sign_active:
            raw = self.last_raw  # Obtener último frame
            if raw is not None:
                # Extraer landmarks (puntos clave) de las manos
                lms = self.gestor.extraer_landmarks(raw)
                # Verificar si se detectaron las manos requeridas
                if len(lms) == req_manos:
                    try:
                        # Guardar imagen
                        self.gestor.guardar_imagen(raw)
                        capturadas += 1
                        # Actualizar barra de progreso
                        self.root.after(0, lambda v=capturadas: self.barra.config(value=v))
                    except Exception as e:
                        print(f"Error captura: {e}")
            time.sleep(0.1)  # Pequeña pausa
            
        # Finalizar captura en el hilo principal
        self.root.after(0, self._finish_capture)

    def _finish_capture(self):
        """Finaliza el proceso de captura automática"""
        self.is_capturing = False
        self.barra.grid_remove()    # Ocultar barra
        self.lbl_estado.grid_remove()  # Ocultar etiqueta
        self.update_buttons()
        messagebox.showinfo("Captura", "Captura automática completada", parent=self.root)

    def on_record(self):
        """Inicia el proceso de grabación de video"""
        if not self.sign_active or self.is_recording:
            return
            
        # Configurar UI para grabación
        self.is_recording = True
        self.lbl_estado.config(text="Preparándose para grabar...")
        self.lbl_estado.grid()  # Mostrar etiqueta
        self.update_buttons()
        
        # Iniciar hilo para grabación en segundo plano
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self):
        """Hilo de trabajo para grabación de video"""
        # Determinar requerimiento de manos según metadatos
        req_manos = 2 if self.metadata["bimanual"] else 1
        
        # Esperar detección de manos (timeout 10s)
        self.root.after(0, lambda: self.lbl_estado.config(text="Esperando detección de manos..."))
        start_time = time.time()
        while time.time() - start_time < 10:
            raw = self.last_raw
            if raw is not None:
                # Verificar detección de manos
                lms = self.gestor.extraer_landmarks(raw)
                if len(lms) == req_manos:
                    break
            time.sleep(0.1)
        else:  # Timeout
            self.root.after(0, lambda: messagebox.showwarning("Tiempo agotado", "No se detectaron manos", parent=self.root))
            self.is_recording = False
            self.update_buttons()
            return
            
        # Grabar durante 3 segundos
        self.root.after(0, lambda: self.lbl_estado.config(text="Grabando..."))
        frames = []  # Almacenar frames
        start_time = time.time()
        while time.time() - start_time < 3:
            raw = self.last_raw
            if raw is not None:
                frames.append(raw.copy())  # Guardar copia del frame
            time.sleep(0.05)  # ≈20 fps
            
        # Guardar video si se capturaron frames
        if frames:
            try:
                self.gestor.guardar_video(frames, fps=20)
                self.root.after(0, lambda: messagebox.showinfo("Video", "Video guardado correctamente", parent=self.root))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error al guardar video: {e}", parent=self.root))
                
        # Finalizar grabación
        self.is_recording = False
        self.root.after(0, lambda: self.lbl_estado.grid_remove())
        self.update_buttons()

    # ---- Entrenamiento ----
    def on_train(self):
        """Inicia el proceso de entrenamiento del modelo"""
        if self.is_training:
            return
            
        # Verificar existencia de datos
        if not any((self.gestor.carpeta_salida).glob("*/*.json")):
            messagebox.showwarning("Entrenamiento", "No hay datos para entrenar", parent=self.root)
            return
            
        # Configurar UI para entrenamiento
        self.is_training = True
        self.barra.config(mode="indeterminate")
        self.barra.grid()  # Mostrar barra
        self.barra.start()  # Iniciar animación
        self.lbl_estado.config(text="Entrenando modelo...")
        self.lbl_estado.grid()  # Mostrar etiqueta
        self.update_buttons()
        
        # Iniciar hilo de entrenamiento en segundo plano
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        """Hilo de trabajo para entrenamiento del modelo"""
        try:
            # Cargar datos preparados
            X_train, X_test, y_train, y_test = self.entrenador.cargar_datos()
            
            # Construir y entrenar modelo
            self.entrenador.construir_modelo(input_dim=X_train.shape[1])
            self.entrenador.entrenar(X_train, y_train)
            
            # Evaluar modelo
            resultados = self.entrenador.evaluar(X_test, y_test)
            # Guardar modelo entrenado
            self.entrenador.guardar_modelo()
            
            # Guardar clases para uso en traducción
            clases = self.entrenador.label_encoder.classes_.tolist()
            with open("dataset/classes.json", "w") as f:
                json.dump(clases, f)
                
            # Preparar mensaje de resultados
            msg = "Entrenamiento completado\n"
            msg += f"Precisión: {resultados['reporte']['accuracy']:.2f}"
        except Exception as e:
            msg = f"Error en entrenamiento:\n{str(e)}"
        
        # Mostrar resultados en hilo principal
        self.root.after(0, lambda: self._finish_train(msg))

    def _finish_train(self, msg):
        """Finaliza el proceso de entrenamiento"""
        self.is_training = False
        self.barra.stop()  # Detener animación
        self.barra.grid_remove()  # Ocultar barra
        self.lbl_estado.grid_remove()  # Ocultar etiqueta
        self.update_buttons()
        messagebox.showinfo("Entrenamiento", msg, parent=self.root)

    # ---- Traducción ----
    def on_translate(self):
        """Activa/desactiva el modo de traducción en tiempo real"""
        if self.is_translating:
            # Desactivar modo traducción
            self.is_translating = False
            self.stop_all_cameras()  # Liberar cámaras
            self.lbl_trad.config(text="------")  # Resetear etiqueta
            self.update_buttons()
            return
            
        # Verificar existencia del modelo entrenado
        if not Path("argsign_model.keras").exists():
            messagebox.showwarning("Traducción", "Primero debe entrenar el modelo", parent=self.root)
            return
            
        # Detener cámaras anteriores
        self.stop_all_cameras()
        
        try:
            # Cargar clases desde archivo
            with open("dataset/classes.json", "r") as f:
                clases = json.load(f)
                
            # Crear instancia de traductor
            self.traductor = Traductor(
                modelo_path="argsign_model.keras",
                classes=clases,
                buffer_size=5  # Tamaño del buffer para predicciones
            )
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar traducción: {str(e)}", parent=self.root)
            return
            
        # Activar modo traducción
        self.is_translating = True
        self.update_buttons()
        self.translate_loop()  # Iniciar bucle de traducción

    def translate_loop(self):
        """Bucle principal para traducción en tiempo real"""
        if self.is_translating and self.traductor:
            # Procesar frame y obtener predicción
            frame, label = self.traductor.traducir_frame()
            
            if frame is not None:
                # Convertir frame a formato Tkinter
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Actualizar canvas con el nuevo frame
                self.canvas.delete("all")
                self.img_id = self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
                self.canvas.imgtk = imgtk  # Mantener referencia
                
            # Actualizar etiqueta de traducción
            self.lbl_trad.config(text=label.upper() if label else "------")
            
            # Programar siguiente iteración (≈30 fps)
            self.root.after(33, self.translate_loop)

    # ---- Finalización ----
    def on_finalize(self):
        """Finaliza la seña actual y limpia recursos"""
        if not self.sign_active:
            return
            
        # Validar datos para señas bimanuales
        if self.metadata.get("bimanual", False):
            imgs = list((self.gestor.carpeta_salida / self.current_sign).glob("*.jpg"))
            if not imgs:
                if not messagebox.askyesno("Validación", 
                                          "No hay imágenes válidas. ¿Desea descartar esta seña?",
                                          parent=self.root):
                    return
                    
        # Resetear estado
        self.sign_active = False
        self.current_sign = None
        self.stop_all_cameras()  # Liberar cámaras
        self.update_buttons()
        messagebox.showinfo("Seña finalizada", "La seña ha sido guardada correctamente", parent=self.root)

    def on_close(self):
        """Maneja el cierre de la aplicación: libera recursos y cierra ventana"""
        self.stop_all_cameras()  # Liberar todas las cámaras
        self.root.destroy()      # Cerrar aplicación

if __name__ == "__main__":
    # Punto de entrada de la aplicación
    root = tk.Tk()          # Crear ventana principal
    app = Interfaz(root)    # Iniciar aplicación
    root.mainloop()         # Iniciar bucle principal de Tkinter
