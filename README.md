# ArgSign
ArgSign es un proyecto básico en Python que busca convertir gestos de Lengua de Señas Argentinas en texto, como paso inicial para una futura aplicación.

---

## Estado Actual

* Módulos implementados:

  1. **Colector**: Captura de video y extracción de frames con mediapipe.
  2. **Dataset**: Organización y preprocesamiento de imágenes gestuales.
  3. **Entrenador**: Entrenamiento de modelos de Machine Learning (TensorFlow/Keras).
  4. **Traductor**: Inferencia en tiempo real de gestos a texto.
  5. **UI**: Interfaz gráfica básica con tkinter.

* Limitaciones:

  * Precisión del modelo aún mejorable (actual \~75% en validación).
  * Arquitectura monolítica; POO en desarrollo.
  * Ausencia de empaquetado y pruebas en entornos sin Python.
  * Interfaz gráfica muy sencilla y sin soporte multiplataforma.

---

## Requisitos

* Python 3.8+
* librerías:

  * numpy
  * opencv-python
  * mediapipe
  * tensorflow
  * scikit-learn
  * tkinter

Instala dependencias con:

```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
ArgSign/
├── colector_modulo.py
├── dataset_modulo.py
├── entrenador_modulo.py
├── traductor_modulo.py
├── UI.py
├── dataset/
│   └── (imágenes y clases.json)
├── models/
│   └── argsign_model.keras
└── README.md
```

---

## Uso

1. Generar o actualizar el dataset:

   ```bash
   python colector_modulo.py
   ```
2. Preprocesar y organizar datos:

   ```bash
   python dataset_modulo.py
   ```
3. Entrenar el modelo:

   ```bash
   python entrenador_modulo.py
   ```
4. Ejecutar el prototipo:

   ```bash
   python UI.py
   ```

---

## Próximos Pasos

* Refactorizar a POO y separar responsabilidades.
* Mejorar el modelo para aumentar precisión.
* Crear un ejecutable o instalador (PyInstaller).
* Diseñar una UI más completa y multiplataforma.
