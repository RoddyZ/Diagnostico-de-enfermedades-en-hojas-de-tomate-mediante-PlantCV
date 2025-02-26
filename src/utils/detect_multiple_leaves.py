import os
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from plantcv import plantcv as pcv

def detect_multiple_leaves(image_path, debug=False, target_size=(256, 256)):
    """
    Detecta si hay más de una hoja en la imagen y devuelve el número de hojas detectadas.
    :param image_path: Ruta de la imagen de entrada.
    :param debug: Si es True, muestra el resultado.
    :param target_size: Tamaño de la imagen normalizada.
    :return: Número de hojas detectadas en la imagen.
    """
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
        print(f"Leaves detected: {num_leaves}")
        debug_img = img.copy()
        cv2.drawContours(debug_img, merged_contours, -1, (0, 255, 0), 2)
        pcv.print_image(debug_img, "segmented_debug.png")
    
    return num_leaves

def process_image(args):
    image_path, debug = args
    return detect_multiple_leaves(image_path, debug)

def analyze_dataset(dataset_path, debug=False):
    """
    Recorre un dataset completo y cuenta cuántas imágenes tienen más de una hoja usando multiprocessing.
    :param dataset_path: Ruta del dataset (carpeta que contiene train, valid y test).
    :param debug: Si es True, imprime detalles del análisis.
    :return: Diccionario con estadísticas por subconjunto y el total.
    """
    subsets = ["train", "valid", "test"]
    results = {subset: {"total": 0, "multiple_leaves": 0} for subset in subsets}
    image_paths = []
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            continue
        
        for class_folder in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image_paths.append((image_path, debug))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(process_image, image_paths), total=len(image_paths), desc="Procesando imágenes"))
    
    index = 0
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            continue
        
        for class_folder in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            for image_file in os.listdir(class_path):
                num_leaves = results_list[index]
                index += 1
                results[subset]["total"] += 1
                if num_leaves > 1:
                    results[subset]["multiple_leaves"] += 1
    
    if debug:
        for subset, stats in results.items():
            print(f"{subset.upper()} - Total Images: {stats['total']}, Multiple Leaves: {stats['multiple_leaves']}")
    
    return results
