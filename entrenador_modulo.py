# ----------------------------
# MÓDULO: entrenador_modulo.py
# ArgSign V1.2.5 – Entrenador con carga incremental de nuevos datos (CORREGIDO)
# ----------------------------
# Importación de librerías
import json                                  # Manejo de archivos JSON
from pathlib import Path                     # Manejo de rutas de archivos
from typing import Tuple, List, Dict, Union  # Tipado avanzado
import numpy as np                           # Operaciones numéricas con arrays
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento/prueba
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Codificación de etiquetas
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de evaluación
import tensorflow as tf                      # Framework de deep learning
from tensorflow.keras import Sequential      # Modelo secuencial de capas
from tensorflow.keras.layers import Input, Dense, Dropout  # Tipos de capas neuronales
from tensorflow.keras.callbacks import EarlyStopping  # Detención temprana del entrenamiento
from dataset_modulo import GestorDataset     # Clase para gestión de dataset
# Clase principal para entrenamiento de modelo
class Entrenador:
    """Entrenamiento de modelo de clasificación con carga incremental"""
    # Constructor de la clase
    def __init__(
        self,
        carpeta_dataset: str = "dataset",    # Ruta al dataset (por defecto 'dataset')
        test_size: float = 0.2,              # Proporción de datos para prueba
        random_state: int = 42                # Semilla para reproducibilidad
    ):
        self.carpeta_dataset = Path(carpeta_dataset)  # Convierte ruta a objeto Path
        self.test_size = test_size                    # Almacena tamaño de conjunto de prueba
        self.random_state = random_state              # Almacena semilla aleatoria
        self.gestor = GestorDataset(carpeta_dataset)  # Instancia de gestor de dataset
        self.label_encoder = LabelEncoder()           # Codificador de etiquetas a enteros
        self.onehot_encoder = OneHotEncoder(sparse_output=False)  # Codificador one-hot
        self.modelo: Union[tf.keras.Model, None] = None  # Modelo de red neuronal (inicialmente nulo)
    # Carga de datos para entrenamiento
    def cargar_datos(
        self,
        new_only: bool = False,                # Bandera para cargar solo datos nuevos
        processed_counts: Dict[str, int] = None # Conteo de muestras procesadas por clase
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]
    ]:
        """Carga datos de landmarks desde archivos JSON"""
        X_list: List[np.ndarray] = []          # Lista para características (features)
        y_list: List[str] = []                 # Lista para etiquetas
        meta = self.gestor._cargar_metadatos()  # Carga metadatos del dataset
        for clase in meta:                      # Itera cada clase en metadatos
            carpeta_clase = self.carpeta_dataset / clase  # Ruta a carpeta de clase
            if not carpeta_clase.exists():       # Si carpeta no existe, salta
                continue
            # Obtiene conteo previo de muestras procesadas
            prev_count = processed_counts.get(clase, 0) if processed_counts else 0
            # Itera archivos JSON de landmarks
            for jf in sorted(carpeta_clase.glob(f"{clase}_*.json")):
                try:
                    # Extrae índice de muestra del nombre de archivo
                    idx = int(jf.stem.split("_")[-1])
                except ValueError:
                    idx = 0
                # Si es carga solo de nuevos y ya fue procesado, salta
                if new_only and idx <= prev_count:
                    continue
                data = json.loads(jf.read_text(encoding="utf-8"))  # Lee landmarks
                # Array para almacenar landmarks: 2 manos, 21 puntos, 3 coordenadas
                vec = np.zeros((2, 21, 3), dtype=np.float32)
                for entry in data:
                    hand_i = 0 if entry["label"] == "Left" else 1  # Índice mano (0=izq,1=der)
                    coords = np.array(entry["landmarks"], dtype=np.float32)  # Coordenadas
                    vec[hand_i] = coords                           # Almacena en array
                # Aplana el array (2*21*3=126) y añade a lista
                X_list.append(vec.reshape(-1))
                y_list.append(clase)                 # Añade etiqueta de clase
        # Manejo de caso sin datos
        if not X_list:
            if new_only:
                # Retorna arrays vacíos para datos nuevos
                return (
                    np.empty((0, 126), dtype=np.float32),
                    np.empty((0, len(self.label_encoder.classes_)), dtype=np.float32)
                )
            # Error si no hay datos en modo completo
            raise RuntimeError("No hay datos en 'dataset/'. Captura antes con la UI.")
        X = np.stack(X_list, axis=0)   # Convierte lista a array 2D
        y = np.array(y_list)           # Convierte etiquetas a array
        if not new_only:
            # Codifica etiquetas a enteros y luego a one-hot
            y_idxs = self.label_encoder.fit_transform(y)
            y_onehot = self.onehot_encoder.fit_transform(y_idxs.reshape(-1, 1))
            # Divide en conjuntos de entrenamiento y prueba
            return train_test_split(
                X, y_onehot,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_idxs        # Mantiene proporción de clases en división
            )
        else:
            # Para nuevos datos, usa encoders ya ajustados
            y_idxs = self.label_encoder.transform(y)
            y_onehot = self.onehot_encoder.transform(y_idxs.reshape(-1, 1))
            return X, y_onehot
    # Construcción de arquitectura del modelo
    def construir_modelo(
        self,
        input_dim: int,               # Dimensión de entrada (126 features)
        hidden_units: int = 128,      # Neuronas en capa oculta
        dropout_rate: float = 0.3     # Tasa de dropout para regularización
    ) -> None:
        """Crea arquitectura de red neuronal"""
        n_clases = len(self.label_encoder.classes_)  # Número de clases
        self.modelo = Sequential([    # Modelo secuencial de capas
            Input(shape=(input_dim,)), # Capa de entrada
            Dense(hidden_units, activation="relu"), # Capa densa con activación ReLU
            Dropout(dropout_rate),    # Capa de dropout para prevenir sobreajuste
            Dense(n_clases, activation="softmax") # Capa de salida con softmax
        ])
        # Compila el modelo con configuración de entrenamiento
        self.modelo.compile(
            optimizer="adam",         # Optimizador Adam
            loss="categorical_crossentropy", # Función de pérdida para clasificación
            metrics=["accuracy"]      # Métrica de precisión
        )
    # Entrenamiento del modelo
    def entrenar(
        self,
        X_train: np.ndarray,          # Características de entrenamiento
        y_train: np.ndarray,          # Etiquetas de entrenamiento (one-hot)
        epochs: int = 50,             # Máximo de épocas de entrenamiento
        batch_size: int = 32,         # Tamaño de lote
        patience: int = 5             # Paciencia para early stopping
    ) -> tf.keras.callbacks.History:
        """Entrena el modelo con detención temprana"""
        if self.modelo is None:       # Verifica que el modelo esté construido
            raise RuntimeError("Modelo no construido. Llama a construir_modelo().")
        # Configura detención temprana basada en pérdida de validación
        early = EarlyStopping(
            monitor="val_loss",       # Monitorea pérdida de validación
            patience=patience,        # Épocas sin mejora para detener
            restore_best_weights=True # Restaura mejores pesos al final
        )
        # Entrena el modelo
        history = self.modelo.fit(
            X_train, y_train,
            validation_split=0.1,     # Usa 10% para validación durante entrenamiento
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early],        # Con detención temprana
            verbose=2                 # Muestra progreso detallado
        )
        return history                # Retorna historial de entrenamiento
    # Evaluación del modelo entrenado
    def evaluar(
        self,
        X_test: np.ndarray,           # Características de prueba
        y_test: np.ndarray            # Etiquetas de prueba (one-hot)
    ) -> Dict[str, any]:
        """Evalúa modelo y muestra métricas"""
        if self.modelo is None:       # Verifica que el modelo exista
            raise RuntimeError("Modelo no construido.")
        y_prob = self.modelo.predict(X_test)  # Obtiene probabilidades de predicción
        y_pred = np.argmax(y_prob, axis=1)   # Clases predichas (índices)
        y_true = np.argmax(y_test, axis=1)   # Clases verdaderas (índices)
        # Genera reporte de clasificación
        reporte = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_, # Nombres de clases
            output_dict=True
        )
        matriz = confusion_matrix(y_true, y_pred) # Matriz de confusión
        # Imprime resultados
        print("=== Reporte de clasificación ===")
        print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))
        print("=== Matriz de confusión ===")
        print(matriz)
        return {"reporte": reporte, "matriz": matriz} # Retorna métricas
    # Guardado de clases en archivo JSON
    def guardar_clases(self, ruta: str = "dataset/classes.json"):
        """Persiste mapeo de clases para uso futuro"""
        clases = self.label_encoder.classes_.tolist() # Convierte a lista
        with open(ruta, 'w', encoding='utf-8') as f:  # Abre archivo
            json.dump(clases, f, ensure_ascii=False)  # Escribe JSON
    # Guardado completo del modelo
    def guardar_modelo(self, ruta: str = "argsign_model.keras") -> None:
        """Guarda modelo y clases asociadas"""
        if self.modelo is None:       # Verifica que el modelo exista
            raise RuntimeError("No hay modelo para guardar.")
        self.modelo.save(ruta)        # Guarda modelo en formato Keras
        self.guardar_clases()         # Guarda clases asociadas
    # Carga de modelo preentrenado
    def cargar_modelo(self, ruta: str = "argsign_model.keras") -> None:
        """Carga modelo previamente guardado"""
        self.modelo = tf.keras.models.load_model(ruta) # Carga modelo desde archivo
# Ejecución directa si es el script principal
if __name__ == "__main__":
    entrenador = Entrenador()         # Crea instancia de entrenador
    try:
        # Carga datos completos y divide en entrenamiento/prueba
        X_train, X_test, y_train, y_test = entrenador.cargar_datos()
    except RuntimeError as e:
        print(e)
        exit(1)
    # Construye modelo con dimensión de entrada
    entrenador.construir_modelo(input_dim=X_train.shape[1])
    entrenador.entrenar(X_train, y_train) # Entrena modelo
    entrenador.evaluar(X_test, y_test)     # Evalúa con datos de prueba
    entrenador.guardar_modelo()            # Guarda modelo entrenado