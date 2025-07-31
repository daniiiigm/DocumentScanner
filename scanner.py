import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def cargar_imagen(ruta):
    """Carga la imagen desde el archivo."""
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen.")
    return imagen

def convertir_a_gris(imagen):
    """Convierte la imagen a escala de grises."""
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

def mejorar_contraste(imagen_gris, alpha=1.4, beta=40):
    """Ajusta el contraste de la imagen, incrementando su fuerza."""
    imagen_contraste = cv2.convertScaleAbs(imagen_gris, alpha=alpha, beta=beta)
    return imagen_contraste

def ecualizar_histograma(imagen_gris):
    """Aplica ecualización del histograma para mejorar el contraste global."""
    return cv2.equalizeHist(imagen_gris)

def umbralizar_imagen(imagen_gris):
    """Aplica un umbral de Otsu para separar la hoja del fondo."""
    _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imagen_binaria

def aplicar_morfologia(imagen_binaria):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    imagen_limpia = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel, iterations=2)
    imagen_limpia = cv2.morphologyEx(imagen_limpia, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(imagen_limpia, connectivity=8)
    if num_labels > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        imagen_limpia = np.uint8(labels == max_label) * 255
    return imagen_limpia

def preprocesar_imagen(imagen):
    imagen_gris = convertir_a_gris(imagen)
    imagen_contraste = mejorar_contraste(imagen_gris)
    imagen_ecualizada = ecualizar_histograma(imagen_contraste)
    imagen_umbralizada = umbralizar_imagen(imagen_ecualizada)
    imagen_preprocesada = aplicar_morfologia(imagen_umbralizada)
    return imagen_preprocesada

def leer_esquinas(archivo):
    """Lee el archivo de texto con las coordenadas de las esquinas etiquetadas."""
    coord_esquinas = {}
    with open(archivo, "r") as f:
        for linea in f:
            datos = linea.strip().split()
            nombre_imagen = datos[0]
            coordenadas = [tuple(map(int, punto.split(','))) for punto in datos[1:]]
            coord_esquinas[nombre_imagen] = coordenadas
    return coord_esquinas

def extraer_descriptores(imagen, puntos_esquina):
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints = [cv2.KeyPoint(x, y, 1) for x, y in puntos_esquina]
    keypoints, descriptores = brief.compute(imagen, keypoints)
    return keypoints, descriptores

def generar_descriptores_referencia(ruta_imagenes, archivo_esquinas):
    """Genera y almacena los descriptores de referencia."""
    esquinas = leer_esquinas(archivo_esquinas)
    descriptores_referencia = {}
    for nombre_imagen, puntos_esquina in esquinas.items():
        ruta_completa = f"{ruta_imagenes}/{nombre_imagen}"
        imagen = cv2.imread(ruta_completa)
        if imagen is None:
            print(f"Error: No se pudo cargar {nombre_imagen}")
            continue
        keypoints, descriptores = extraer_descriptores(imagen, puntos_esquina)
        descriptores_referencia[nombre_imagen] = descriptores
    return descriptores_referencia

def detectar_esquinas_harris(imagen_preprocesada, blockSize=5, ksize=3, k=0.04, umbral_relativo=0.02):
    contornos, _ = cv2.findContours(imagen_preprocesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return []
    contorno = max(contornos, key=cv2.contourArea)
    mascara = np.zeros_like(imagen_preprocesada)
    cv2.drawContours(mascara, [contorno], -1, 255, thickness=2)
    imagen_float = np.float32(imagen_preprocesada)
    respuesta = cv2.cornerHarris(imagen_float, blockSize, ksize, k)
    respuesta = cv2.normalize(respuesta, None, 0, 255, cv2.NORM_MINMAX)
    respuesta = cv2.bitwise_and(respuesta, respuesta, mask=mascara)
    coordenadas = np.argwhere(respuesta > umbral_relativo * respuesta.max())
    return [cv2.KeyPoint(float(x[1]), float(x[0]), 20) for x in coordenadas]

def calcular_descriptores_brief(imagen_gris, keypoints):
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints_validos, descriptores = brief.compute(imagen_gris, keypoints)
    return keypoints_validos, descriptores

def encontrar_mejores_esquinas(descriptores_test, keypoints_test, descriptores_ref, shape_imagen):
    # 1. Preparar descriptores de referencia: juntar todos en un solo array
    todos_descriptores_ref = np.vstack([d for d in descriptores_ref.values()])
    # 2. Configurar el matcher KNN
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptores_test, todos_descriptores_ref, k=1)
    # 3. Extraer distancias y coordenadas
    distancias = [m[0].distance for m in matches]
    puntos = [keypoints_test[m[0].queryIdx].pt for m in matches]
    # 4. Dividir la imagen en 4 cuadrantes
    h, w = shape_imagen[:2]
    mid_x, mid_y = w // 2, h // 2
    cuadrantes = {
        0: (0, mid_x, 0, mid_y),
        1: (mid_x, w, 0, mid_y),
        2: (0, mid_x, mid_y, h),
        3: (mid_x, w, mid_y, h)
    }
    # 5. Encontrar el mejor punto por cuadrante
    mejores_puntos = []
    for cuadrante in cuadrantes.values():
        x_min, x_max, y_min, y_max = cuadrante
        candidatos = []
        for pt, dist in zip(puntos, distancias):
            x, y = pt
            if x_min <= x < x_max and y_min <= y < y_max:
                candidatos.append((pt, dist))
        if candidatos:
            mejor = min(candidatos, key=lambda x: x[1])
            kp = cv2.KeyPoint(mejor[0][0], mejor[0][1], 1)
            mejores_puntos.append(kp)
    # 6. Ordenar los puntos en el orden
    return seleccionar_extremos([kp.pt for kp in mejores_puntos])

def seleccionar_extremos(puntos):
    """
    Selecciona las esquinas del documento usando criterios geométricos:
    Top-Left: mínimo x+y
    Bottom-Right: máximo x+y
    Top-Right: mínimo (x-y)
    Bottom-Left: máximo (x-y)
    """
    puntos = np.array(puntos)
    tl = puntos[np.argmin(puntos[:, 0] + puntos[:, 1])]
    br = puntos[np.argmax(puntos[:, 0] + puntos[:, 1])]
    tr = puntos[np.argmin(puntos[:, 0] - puntos[:, 1])]
    bl = puntos[np.argmax(puntos[:, 0] - puntos[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def ordenar_esquinas(puntos):
    """Ordena las esquinas en el orden: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    puntos = np.array(puntos)
    centro = np.mean(puntos, axis=0)
    izquierda = puntos[puntos[:, 0] < centro[0]]
    derecha = puntos[puntos[:, 0] >= centro[0]]
    tl = izquierda[izquierda[:, 1].argmin()]
    bl = izquierda[izquierda[:, 1].argmax()]
    tr = derecha[derecha[:, 1].argmin()]
    br = derecha[derecha[:, 1].argmax()]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def mostrar_imagen_ajustada(nombre_ventana, imagen, max_ancho=1000, max_alto=700):
    alto, ancho = imagen.shape[:2]
    if ancho > max_ancho or alto > max_alto:
        factor = min(max_ancho / ancho, max_alto / alto)
        imagen = cv2.resize(imagen, (int(ancho * factor), int(alto * factor)))
    cv2.imshow(nombre_ventana, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rectificar_imagen_ransac(imagen, esquinas):
    tl, tr, br, bl = esquinas
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    H, mask = cv2.findHomography(esquinas, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
    imagen_rectificada = cv2.warpPerspective(imagen, H, (maxWidth, maxHeight))
    return imagen_rectificada

if __name__ == "__main__":
    # Se toma la ruta de la imagen de entrada (o se usa una por defecto)
    if len(sys.argv) > 1:
        ruta_imagen = sys.argv[1]
    else:
        ruta_imagen = "Muestra/testing/Hoja_01.jpg"

    imagen = cargar_imagen(ruta_imagen)
    imagen_preprocesada = preprocesar_imagen(imagen)
    descriptores_ref = generar_descriptores_referencia("Muestra/learning/", "esquinas.txt")
    np.save("descriptores_referencia.npy", descriptores_ref)
    puntos_harris = detectar_esquinas_harris(imagen_preprocesada)
    keypoints_validos, descriptores = calcular_descriptores_brief(imagen_preprocesada, puntos_harris)

    if len(keypoints_validos) >= 4 and descriptores is not None:
        descriptores_ref = np.load("descriptores_referencia.npy", allow_pickle=True).item()

        if len(puntos_harris) == 4:
            esquinas_ordenadas = ordenar_esquinas(puntos_harris)
        else:
            candidatos = [kp.pt for kp in keypoints_validos]
            extremos = seleccionar_extremos(candidatos)
            esquinas_ordenadas = ordenar_esquinas(extremos)

        # Rectificación usando RANSAC
        imagen_rectificada = rectificar_imagen_ransac(imagen, esquinas_ordenadas)
        mostrar_imagen_ajustada("Imagen rectificada", imagen_rectificada)

    else:
        print("No se encontraron suficientes esquinas válidas para la rectificación.")


