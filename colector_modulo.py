# ----------------------------
# MÓDULO: colector_modulo.py
# ArgSign V1.2.5 - Captura de frames y preprocesamiento
# ----------------------------
# Importación de librerías
import cv2                                 # OpenCV para procesamiento de imágenes/video
import mediapipe as mp                     # Framework para detección de landmarks
import numpy as np                         # Manejo eficiente de arrays numéricos
# Definición de clase principal
class Colector:
    """Clase para captura de video, detección de manos y dibujo de landmarks"""
    # Constructor de la clase
    def __init__(
        self,
        indice_camara: int = 0,                   # Índice de dispositivo de cámara
        resolucion: tuple[int, int] = (640, 480),  # Resolución de captura (ancho, alto)
        modo_imagen_estatica: bool = False,        # Modo detección (True: imagen estática, False: video)
        max_manos: int = 2,                        # Máximo de manos a detectar simultáneamente
        confianza_minima: float = 0.5              # Umbral mínimo de confianza para detecciones
    ):
        # Asignación de parámetros a atributos
        self.indice_camara = indice_camara          # Almacena índice de cámara
        self.resolucion = resolucion                # Almacena resolución de video
        self.cap: cv2.VideoCapture | None = None    # Objeto para captura de video (inicializado como None)
        # Configuración del modelo de detección de manos
        self.mano_modelo = mp.solutions.hands.Hands(
            static_image_mode=modo_imagen_estatica,  # Modo de procesamiento (estático/video)
            max_num_hands=max_manos,                 # Máximo de manos a detectar
            min_detection_confidence=confianza_minima # Confianza mínima requerida
        )
        # Utilidades para dibujo
        self.mp_dibujo = mp.solutions.drawing_utils   # Herramientas para dibujar landmarks
        self.mp_styles = mp.solutions.drawing_styles  # Estilos predefinidos para dibujo
        self.mp_hands = mp.solutions.hands            # Referencia al módulo de manos
        # Personalización de estilo para puntos de landmarks
        self.estilo_puntos = self.mp_dibujo.DrawingSpec(
            color=(0, 0, 255),     # Color BGR (rojo en este caso)
            thickness=2,            # Grosor de línea para puntos
            circle_radius=2         # Radio de círculos de landmarks
        )
        # Personalización de estilo para conexiones
        self.estilo_segmentos = self.mp_dibujo.DrawingSpec(
            color=(128, 128, 128),  # Color BGR (gris)
            thickness=2             # Grosor de líneas de conexión
        )

        # Inicialización de cámara
        self._iniciar_camara()       # Llama método privado para configurar cámara
    # Método privado para configuración de cámara
    def _iniciar_camara(self):
        self.cap = cv2.VideoCapture(self.indice_camara)  # Crea objeto VideoCapture
        if not self.cap.isOpened():                       # Verifica apertura exitosa
            raise IOError(f"No se pudo abrir la cámara índice {self.indice_camara}")  # Error si falla
        ancho, alto = self.resolucion                    # Desempaqueta resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)    # Establece ancho de frame
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)    # Establece altura de frame
    # Captura y procesamiento de frames
    def obtener_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """Obtiene frame de cámara, aplica espejo y detección de manos"""
        ret, frame = self.cap.read()                    # Captura frame desde cámara
        if not ret:                                     # Verifica captura exitosa
            return None, None, False                    # Retorna valores nulos si falla  
        frame = cv2.flip(frame, 1)                      # Espeja imagen horizontalmente
        # Procesa frame para detección de manos
        anotado, hay_mano = self._procesar_frame(frame)  
        return frame, anotado, hay_mano                 # Retorna frame original, anotado y estado de detección
    # Procesamiento interno de frame
    def _procesar_frame(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Detección de manos y dibujo de landmarks en frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # Convierte BGR a RGB
        anotado = rgb.copy()                            # Crea copia para dibujar
        resultados = self.mano_modelo.process(rgb)      # Procesa frame con modelo de manos
        hay_mano = False                                # Inicializa bandera de detección
        if resultados.multi_hand_landmarks:             # Verifica si hay manos detectadas
            hay_mano = True                            # Actualiza bandera
            # Dibuja landmarks para cada mano detectada
            for lm in resultados.multi_hand_landmarks:  
                self.mp_dibujo.draw_landmarks(
                    anotado,                           # Imagen destino
                    lm,                                # Landmarks de la mano
                    self.mp_hands.HAND_CONNECTIONS,     # Conexiones anatómicas
                    self.estilo_puntos,                # Estilo personalizado para puntos
                    self.estilo_segmentos              # Estilo personalizado para conexiones
                )
        return anotado, hay_mano                      # Retorna frame anotado y estado de detección
    # Método para cambiar modo de detección
    def cambiar_modo(self, modo_estatico: bool = True):  
        self.mano_modelo.close()                      # Cierra modelo actual
        # Recrea modelo con nuevo modo
        self.mano_modelo = mp.solutions.hands.Hands(
            static_image_mode=modo_estatico,          # Nuevo modo (estático/video)
            max_num_hands=self.mano_modelo._max_num_hands,  # Mantiene mismo máximo de manos
            min_detection_confidence=self.mano_modelo._min_detection_confidence  # Mantiene confianza
        )
    # Liberación de recursos
    def liberar(self):                                
        if self.cap and self.cap.isOpened():          # Verifica si cámara está activa
            self.cap.release()                       # Libera dispositivo de cámara
        self.mano_modelo.close()                     # Cierra modelo de MediaPipe
        cv2.destroyAllWindows()                      # Cierra ventanas de OpenCV