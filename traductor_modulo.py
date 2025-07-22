# ----------------------------
# MÓDULO: traductor_modulo.py
# ArgSign V1.2.5 – Traductor
# ----------------------------
# Importación de librerías
import json                                    # Carga de clases desde archivo JSON
from pathlib import Path                        # Manejo de rutas de archivos
from typing import Optional, List               # Tipado avanzado para funciones
import cv2                                     # Procesamiento de imágenes y ventanas
import numpy as np                             # Manipulación de arrays numéricos
import tensorflow as tf                        # Carga y uso de modelos de predicción
from colector_modulo import Colector           # Captura de imágenes y landmarks
from dataset_modulo import GestorDataset       # Extracción de landmarks desde frames
# Clase principal para traducción en tiempo real
class Traductor:
    """Traduce señas mediante cámara usando modelo preentrenado"""
    # Constructor de la clase
    def __init__(
        self,
        modelo_path: str = "argsign_model.keras",  # Ruta al modelo entrenado
        classes: List[str] = [],                   # Lista de nombres de clases
        buffer_size: int = 5                       # Tamaño de buffer para suavizado
    ):
        self.colector = Colector()                 # Instancia para captura de cámara
        self.gestor = GestorDataset()              # Instancia para extracción de landmarks
        self.modelo = tf.keras.models.load_model(modelo_path) # Carga modelo preentrenado
        self.clases = classes                      # Almacena lista de clases posibles
        self.buffer_size = buffer_size             # Configura tamaño de buffer
        self.historial: List[Optional[str]] = []   # Buffer para historial de predicciones
    # Predicción para un vector de características
    def _predecir_vector(self, vec: np.ndarray) -> Optional[str]:
        """Predice clase para vector de entrada (126 features)"""
        X = vec.reshape(1, -1).astype(np.float32)  # Reformatea a array 2D
        probs = self.modelo.predict(X, verbose=0)  # Obtiene probabilidades de clases
        idx = int(np.argmax(probs, axis=1)[0])     # Índice de clase con mayor probabilidad
        return self.clases[idx]                    # Retorna nombre de clase
    # Procesamiento de frame para traducción
    def traducir_frame(self) -> (Optional[np.ndarray], Optional[str]):
        """Captura frame, extrae landmarks y predice seña"""
        # Obtiene frame de cámara (BGR original y RGB anotado)
        frame_bgr, frame_anotado, _ = self.colector.obtener_frame()
        if frame_bgr is None:
            return None, None                      # Retorna si falla captura
        lms = self.gestor.extraer_landmarks(frame_bgr)  # Extrae landmarks de frame BGR
        etiqueta = None                             # Inicializa etiqueta de predicción
        # Procesa si hay 1 o 2 manos detectadas
        if len(lms) in (1, 2):
            vec = np.zeros((2, 21, 3), dtype=np.float32)  # Array para landmarks (2 manos)
            for entry in lms:
                idx = 0 if entry["label"] == "Left" else 1  # Índice mano (0=izq,1=der)
                vec[idx] = np.array(entry["landmarks"], dtype=np.float32)  # Almacena coordenadas
            etiqueta = self._predecir_vector(vec.reshape(-1))  # Predice con vector aplanado
        # Suavizado de predicciones con buffer
        if etiqueta is not None:
            self.historial.append(etiqueta)         # Añade predicción actual al buffer
            if len(self.historial) > self.buffer_size:  # Controla tamaño máximo
                self.historial.pop(0)               # Elimina predicción más antigua
            # Selecciona etiqueta más frecuente en buffer
            etiqueta = max(set(self.historial), key=self.historial.count)
        else:
            self.historial.clear()                 # Limpia buffer si no hay predicción
        # Dibuja etiqueta en frame anotado
        if etiqueta:
            cv2.putText(
                frame_anotado,                     # Imagen donde dibujar
                etiqueta.upper(),                  # Texto en mayúsculas
                (10, 30),                         # Posición (x,y)
                cv2.FONT_HERSHEY_SIMPLEX,          # Tipo de fuente
                1.0,                               # Escala de texto
                (255, 255, 255), 2,                # Color blanco, grosor 2
                cv2.LINE_AA                        # Antialiasing
            )
        return frame_anotado, etiqueta             # Retorna frame anotado y etiqueta
    # Bucle principal de ejecución
    def iniciar(self, ventana: str = "Traductor ArgSign"):
        """Bucle principal para mostrar traducción en tiempo real"""
        cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)  # Crea ventana redimensionable
        try:
            while True:
                frame, etiqueta = self.traducir_frame()  # Procesa frame actual
                if frame is None:
                    break
                # Convierte RGB a BGR y muestra en ventana
                cv2.imshow(ventana, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                key = cv2.waitKey(1) & 0xFF        # Lee tecla presionada
                if key == ord('q'):                # Sale si presiona 'q'
                    break
                # Verifica si se cerró la ventana
                if cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:                  # Maneja Ctrl+C desde consola
            pass
        finally:
            self.colector.liberar()               # Libera recursos de cámara
            cv2.destroyAllWindows()                # Cierra todas las ventanas
# Ejecución directa como script
if __name__ == "__main__":
    try:
        # Carga lista de clases desde archivo JSON
        with open("dataset/classes.json", "r", encoding="utf-8") as f:
            clases = json.load(f)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'dataset/classes.json'")
        exit(1)
    trad = Traductor(classes=clases)      # Crea instancia de traductor
    trad.iniciar()                        # Inicia bucle de traducción
