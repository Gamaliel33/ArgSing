# ----------------------------
# MÓDULO: dataset_modulo.py
# ArgSign V1.2.5 - Gestión de almacenamiento y extracción de landmarks
# ----------------------------
# Importación de librerías
import json                                           # Manipulación de archivos JSON
from pathlib import Path                              # Manejo de rutas de archivos multiplataforma
from typing import List, Dict, Any, Tuple             # Tipado avanzado para funciones
import cv2                                            # Procesamiento de imágenes/video
import mediapipe as mp                                # Detección de landmarks
import numpy as np                                    # Manejo de arrays numéricos
# Clase principal para gestión de dataset
class GestorDataset:
    """Gestión de almacenamiento, extracción de landmarks y metadatos"""
    # Constante para archivo de metadatos
    METADATOS_POR_DEFECTO = "metadata_senias.json"    # Nombre predeterminado para archivo de metadatos
    # Constructor de la clase
    def __init__(self, carpeta_salida: str = "dataset"):
        self.carpeta_salida = Path(carpeta_salida)    # Convierte ruta a objeto Path
        self.carpeta_salida.mkdir(parents=True, exist_ok=True)  # Crea carpeta si no existe
        self.senia_actual: str | None = None          # Nombre de la seña actual (inicialmente nulo)
        self.carpeta_senia: Path | None = None        # Ruta de carpeta específica para la seña
        self.indice_imagen = 1                        # Contador secuencial para imágenes
        self.indice_video = 1                         # Contador secuencial para videos
        self.archivo_meta = self.carpeta_salida / self.METADATOS_POR_DEFECTO  # Ruta completa a metadatos
        self.meta = self._cargar_metadatos()          # Carga metadatos existentes      
        # Configuración del modelo de detección de manos
        self.mano_modelo = mp.solutions.hands.Hands(
            static_image_mode=True,                   # Modo imagen estática (no video)
            max_num_hands=2,                          # Máximo 2 manos por detección
            min_detection_confidence=0.5              # Confianza mínima para detección
        )
        self.mp_hands = mp.solutions.hands            # Referencia al módulo de manos
    # Carga de metadatos desde archivo JSON
    def _cargar_metadatos(self) -> Dict[str, Any]:
        """Carga metadatos existentes desde archivo JSON"""
        if self.archivo_meta.exists():                # Verifica existencia del archivo
            try:
                return json.loads(                    # Decodifica contenido JSON
                    self.archivo_meta.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:              # Maneja errores de formato
                return {}                            # Retorna diccionario vacío en caso de error
        return {}                                    # Retorna vacío si no existe archivo
    # Guardado de metadatos en disco
    def _guardar_metadatos(self):
        """Persiste metadatos en archivo JSON"""
        self.archivo_meta.write_text(                 # Escribe texto en archivo
            json.dumps(self.meta, indent=2, ensure_ascii=False),  # Serializa a JSON con formato
            encoding='utf-8'                         # Codificación UTF-8
        )
    # Creación de estructura para nueva seña
    def crear_carpeta_senia(self, nombre: str, metadatos: Dict[str, Any]):
        """Prepara entorno para almacenar datos de una nueva seña"""
        self.senia_actual = nombre                   # Almacena nombre de seña actual
        self.carpeta_senia = self.carpeta_salida / nombre  # Construye ruta de carpeta
        # Verifica si la carpeta contiene datos previos
        if self.carpeta_senia.exists() and any(self.carpeta_senia.iterdir()):
            raise FileExistsError(f"La carpeta '{nombre}' ya contiene datos.")  # Error si tiene contenido
        self.carpeta_senia.mkdir(exist_ok=True)      # Crea carpeta (si no existe)
        self.meta[nombre] = metadatos                # Añade metadatos al diccionario
        self._guardar_metadatos()                    # Guarda metadatos actualizados
        self.indice_imagen = 1                      # Reinicia contador de imágenes
        self.indice_video = 1                        # Reinicia contador de videos
    # Extracción de landmarks desde imagen
    def extraer_landmarks(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detecta landmarks de manos en frame BGR"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte BGR a RGB
        resultados = self.mano_modelo.process(rgb)    # Procesa imagen con MediaPipe
        lista: List[Dict[str, Any]] = []              # Lista para resultados
        # Verifica si hay detecciones de manos
        if resultados.multi_hand_landmarks and resultados.multi_handedness:
            # Combina información de lateralidad y landmarks
            for mano, lm in zip(resultados.multi_handedness, resultados.multi_hand_landmarks):
                etiqueta = mano.classification[0].label  # "Left" o "Right"
                # Extrae coordenadas normalizadas (x,y,z) de cada landmark
                coords = [[p.x, p.y, p.z] for p in lm.landmark]  
                lista.append({"label": etiqueta, "landmarks": coords})  # Añade a resultados
        return lista                                  # Retorna lista de detecciones
    # Almacenamiento de imagen y sus landmarks
    def guardar_imagen(self, frame: np.ndarray):
        """Guarda frame como imagen JPG con su JSON de landmarks"""
        if not self.senia_actual or not self.carpeta_senia:
            raise RuntimeError("Debe crear carpeta de seña primero.")  # Validación de estado
        base = f"{self.senia_actual}_{self.indice_imagen:03d}"  # Nombre base con padding
        img_path = self.carpeta_senia / f"{base}.jpg"  # Ruta completa para imagen
        cv2.imwrite(str(img_path), frame)             # Guarda imagen en disco
        lms = self.extraer_landmarks(frame)           # Extrae landmarks
        json_path = self.carpeta_senia / f"{base}.json"  # Ruta para archivo JSON
        json_path.write_text(                         # Guarda landmarks en JSON
            json.dumps(lms, indent=2), encoding="utf-8"
        )
        self.indice_imagen += 1                       # Incrementa contador de imágenes
    # Almacenamiento de secuencia de frames como video
    def guardar_video(self, frames: List[np.ndarray], fps: int = 20, codec: str = "XVID"):
        """Guarda lista de frames como video AVI"""
        if not self.senia_actual or not self.carpeta_senia:
            raise RuntimeError("Debe crear carpeta de seña primero.")  # Validación de estado
        base = f"{self.senia_actual}_video_{self.indice_video}"  # Nombre base para video
        ruta = self.carpeta_senia / f"{base}.avi"    # Ruta completa para video
        alto, ancho = frames[0].shape[:2]             # Dimensiones del primer frame
        fourcc = cv2.VideoWriter_fourcc(*codec)       # Configura códec de video
        # Crea objeto VideoWriter
        writer = cv2.VideoWriter(str(ruta), fourcc, fps, (ancho, alto))
        for f in frames:
            writer.write(f)                           # Escribe cada frame
        writer.release()                              # Cierra archivo de video
        self.indice_video += 1                        # Incrementa contador de videos
    # Validación de contenido bimanual
    def validar_bimanual(self) -> Tuple[int, int]:
        """Verifica que señas bimanuales tengan 2 manos en +90% de imágenes"""
        if not self.senia_actual or not self.carpeta_senia:
            raise RuntimeError("No hay seña activa.")  # Validación de estado
        meta = self.meta.get(self.senia_actual, {})   # Metadatos de seña actual
        if not meta.get("bimanual"):                  # Verifica si es bimanual
            return 0, 0                               # Retorna si no aplica
        # Busca todas las imágenes de la seña
        imgs = sorted(self.carpeta_senia.glob(f"{self.senia_actual}_*.jpg"))
        tot = len(imgs)                               # Total de imágenes
        v = 0                                         # Contador de válidas
        for im in imgs:
            # Ruta del JSON asociado
            json_path = im.with_suffix(".json")
            data = json.loads(json_path.read_text(encoding="utf-8"))  # Carga landmarks
            if len(data) == 2:                        # Verifica dos manos detectadas
                v += 1                                # Incrementa contador válidas
        return v, tot                                 # Retorna estadísticas
