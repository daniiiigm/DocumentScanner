📄 Document Scanner & Rectifier
Este proyecto permite detectar, extraer y rectificar documentos (como hojas escaneadas o fotografiadas) utilizando Python + OpenCV.

Se basa en detección de esquinas, generación de descriptores BRIEF y rectificación por homografía (RANSAC) para obtener imágenes de documentos alineadas y limpias.

🚀 Características
✅ Preprocesamiento de imágenes:

Conversión a escala de grises

Mejora de contraste y ecualización de histograma

Umbralización adaptativa (Otsu)

Limpieza con operaciones morfológicas

✅ Detección de esquinas con:

Harris Corner Detector

Selección de esquinas extremas por cuadrantes

✅ Extracción de descriptores BRIEF para identificación de puntos de referencia

✅ Rectificación del documento con homografía + RANSAC

✅ Visualización de la imagen rectificada en ventana emergente

📂 Estructura del proyecto
├── scanner.py                  # Script principal para detección y rectificación
├── descriptores_referencia.npy # Descriptores guardados (generados en la primera ejecución)
├── Muestra/
│   ├── learning/               # Imágenes de referencia con esquinas etiquetadas
│   └── testing/                # Imágenes a procesar
├── esquinas.txt                # Coordenadas de las esquinas de referencia
└── README.md                   # Este archivo

🛠️ Requisitos
Python 3.8+
OpenCV (con módulos xfeatures2d de opencv-contrib-python)
NumPy
Matplotlib (opcional para debug)

Instalación rápida:
pip install opencv-contrib-python numpy matplotlib

▶️ Uso
Coloca tus imágenes a escanear en Muestra/testing/.
(Opcional) Asegúrate de tener esquinas.txt y Muestra/learning/ para generar descriptores de referencia.

Ejecuta:
python scanner.py ruta/a/mi_imagen.jpg

Si no se especifica ruta, usará por defecto:
Muestra/testing/Hoja_01.jpg

Si se detectan correctamente al menos 4 esquinas, se abrirá una ventana con la imagen rectificada.

📊 Flujo de trabajo
Preprocesamiento → Limpieza de la imagen

Detección de esquinas → Harris + selección por cuadrantes

Cálculo de descriptores BRIEF → Coincidencia con referencias

Ordenamiento de esquinas → Top-Left → Top-Right → Bottom-Right → Bottom-Left

Rectificación con RANSAC → Imagen final alineada
