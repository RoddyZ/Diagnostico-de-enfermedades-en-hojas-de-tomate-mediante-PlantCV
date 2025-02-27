import os
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from plantcv import plantcv as pcv
import shutil

def detect_rotation_and_shadows(image_path, angle_threshold=30, shadow_threshold=50):
    """
    Detecta si una imagen tiene una rotación extrema o sombras fuertes.
    :param image_path: Ruta de la imagen de entrada.
    :param angle_threshold: Umbral de ángulo para detectar rotaciones extremas.
    :param shadow_threshold: Umbral para detectar imágenes con sombras fuertes.
    :return: True si la imagen debe descartarse, False en caso contrario.
    """
    img = cv2.imread(image_path)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    
    # Detectar líneas para estimar la inclinación de la hoja
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = [np.rad2deg(line[0][1]) - 90 for line in lines]  # Convertir a grados
        mean_angle = np.mean(angles)
        if abs(mean_angle) > angle_threshold:
            return True  # Imagen muy rotada
    
    # Evaluar presencia de sombras usando el histograma de intensidad
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    shadow_pixels = np.sum(hist[:shadow_threshold]) / np.sum(hist)  # % de píxeles oscuros
    if shadow_pixels > 0.3:
        return True  # Imagen con sombras fuertes
    
    return False

def move_to_deprecated(image_path, dataset_path, deprecated_path="dataset_deprecated"):
    """
    Mueve imágenes descartadas a la carpeta 'deprecated', manteniendo la jerarquía original.
    """
    relative_path = os.path.relpath(image_path, dataset_path)
    new_path = os.path.join(deprecated_path, relative_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    os.rename(image_path, new_path)

def detect_multiple_leaves(image_path, debug=False, target_size=(256, 256)):
    """
    Detecta si hay más de una hoja en la imagen y devuelve el número de hojas detectadas.
    También descarta imágenes con rotaciones extremas o sombras.
    :param image_path: Ruta de la imagen de entrada.
    :param debug: Si es True, muestra el resultado.
    :param target_size: Tamaño de la imagen normalizada.
    :return: Número de hojas detectadas en la imagen o None si la imagen es descartada.
    """
    if detect_rotation_and_shadows(image_path):
        return None, image_path  # Descartar imagen
    
    img, _, _ = pcv.readimage(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # Asegurar tamaño estándar
    grayscale_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    
    # Usar umbralización de Otsu en lugar de un umbral fijo
    binary_img = pcv.threshold.otsu(grayscale_img)
    filled_img = pcv.fill_holes(binary_img)
    mask = pcv.dilate(filled_img, ksize=5, i=2)
    
    # Usar watershed segmentation para detectar objetos (hojas)
    labeled_objects = pcv.watershed_segmentation(img, mask)
    
    # Convertir labeled_objects a formato binario para cv2.findContours()
    labeled_binary = np.where(labeled_objects > 0, 255, 0).astype(np.uint8)
    
    # Encontrar contornos de los objetos segmentados usando OpenCV
    contours, _ = cv2.findContours(labeled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ajustar min_leaf_area dinámicamente según el tamaño de la imagen
    reference_size = 256 * 256  # Tamaño base para imágenes normalizadas
    min_leaf_area = (5000 / reference_size) * (target_size[0] * target_size[1])  # Escalar según tamaño
    
    # Filtrar contornos pequeños (ruido)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_leaf_area]
    
    # Aplicar convexHull para fusionar contornos cercanos y evitar sobresegmentación
    merged_contours = [cv2.convexHull(cnt) for cnt in filtered_contours]
    
    # Contar el número de hojas detectadas después del filtrado y fusión
    num_leaves = len(merged_contours)
    
    if debug:
        print(f"Leaves detected: {num_leaves} in {image_path}")
        debug_img = img.copy()
        cv2.drawContours(debug_img, merged_contours, -1, (0, 255, 0), 2)
        pcv.print_image(debug_img, "segmented_debug.png")
    
    return num_leaves, image_path

def process_image(args):
    image_path, debug = args
    return detect_multiple_leaves(image_path, debug)

def copy_to_deprecated(image_path, dataset_path, deprecated_path="dataset_deprecated"):
    """
    Copia imágenes descartadas a la carpeta 'deprecated', manteniendo la jerarquía original.
    """
    relative_path = os.path.relpath(image_path, dataset_path)
    new_path = os.path.join(deprecated_path, relative_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.copy2(image_path, new_path)

def analyze_dataset(dataset_path, debug=False):
    """
    Recorre un dataset completo y cuenta cuántas imágenes tienen más de una hoja usando multiprocessing.
    También descarta imágenes con rotaciones extremas o sombras y las mueve a la carpeta 'deprecated'.
    """
    subsets = ["train", "valid", "test"]
    results = {subset: {"total": 0, "multiple_leaves": 0, "discarded": [], "files": []} for subset in subsets}
    image_paths = []

    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            continue

        for root, _, files in os.walk(subset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    image_paths.append((image_path, debug))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths), desc="Procesando imágenes"))

    index = 0
    for subset in subsets:
        for root, _, files in os.walk(os.path.join(dataset_path, subset)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    result = results_list[index]
                    index += 1
                    if result is None:
                        image_path = os.path.join(root, file)
                        copy_to_deprecated(image_path, dataset_path)
                        results[subset]["discarded"].append(file)
                        continue
                    num_leaves, image_path = result
                    if num_leaves is not None:
                        results[subset]["total"] += 1
                        if num_leaves > 1:
                            results[subset]["multiple_leaves"] += 1
                            results[subset]["files"].append(os.path.normpath(image_path)) # Normalizar la ruta

    if debug:
        for subset, stats in results.items():
            print(f"{subset.upper()} - Total Images: {stats['total']}, Multiple Leaves: {stats['multiple_leaves']}, Discarded: {len(stats['discarded'])}")

    return results