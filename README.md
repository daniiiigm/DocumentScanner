# 📄 Document Scanner & Rectifier

Este proyecto permite **detectar, extraer y rectificar documentos** (como hojas escaneadas o fotografiadas) utilizando **Python + OpenCV**.

Se basa en **detección de esquinas**, generación de **descriptores BRIEF** y **rectificación por homografía (RANSAC)** para obtener imágenes de documentos alineadas y limpias.

---

## 🚀 Características

- ✅ **Preprocesamiento de imágenes:**
  - Conversión a escala de grises
  - Mejora de contraste y ecualización de histograma
  - Umbralización adaptativa (Otsu)
  - Limpieza con operaciones morfológicas

- ✅ **Detección de esquinas con:**
  - Harris Corner Detector
  - Selección de esquinas extremas por cuadrantes

- ✅ **Extracción de descriptores BRIEF** para identificación de puntos de referencia  
- ✅ **Rectificación del documento** con **homografía + RANSAC**  
- ✅ **Visualización** de la imagen rectificada en ventana emergente

---

## 📂 Estructura del proyecto

```bash
├── scanner.py                  # Script principal para detección y rectificación
├── descriptores_referencia.npy # Descriptores guardados (generados en la primera ejecución)
├── Muestra/
│   ├── learning/               # Imágenes de referencia con esquinas etiquetadas
│   └── testing/                # Imágenes a procesar
├── esquinas.txt                # Coordenadas de las esquinas de referencia
└── README.md                   # Este archivo
