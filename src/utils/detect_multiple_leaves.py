import os
import cv2
import numpy as np
from plantcv import plantcv as pcv

def detect_multiple_leaves(image_path, debug=False):
    """
    Detecta si hay más de una hoja en la imagen y devuelve el número de hojas detectadas.
    :param image_path: Ruta de la imagen de entrada.
    :param debug: Si es True, muestra el resultado.
    :return: Número de hojas detectadas en la imagen.
    """
    img, _, _ = pcv.readimage(image_path)
    grayscale_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary_img = pcv.threshold.binary(grayscale_img, threshold=128, object_type='light')
    filled_img = pcv.fill_holes(binary_img)
    mask = pcv.dilate(filled_img, ksize=5, i=2)
    
    # Usar watershed segmentation para detectar objetos (hojas)
    labeled_objects = pcv.watershed_segmentation(img, mask)
    
    # Convertir labeled_objects a formato binario para cv2.findContours()
    labeled_binary = np.where(labeled_objects > 0, 255, 0).astype(np.uint8)
    
    # Encontrar contornos de los objetos segmentados usando OpenCV
    contours, _ = cv2.findContours(labeled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contar el número de hojas detectadas
    num_leaves = len(contours)
    
    if debug:
        print(f"Leaves detected: {num_leaves}")
        pcv.print_image(labeled_binary, "segmented_debug.png")
    
    return num_leaves

def analyze_dataset(dataset_path, debug=False):
    """
    Recorre un dataset completo y cuenta cuántas imágenes tienen más de una hoja.
    :param dataset_path: Ruta del dataset (carpeta que contiene train, valid y test).
    :param debug: Si es True, imprime detalles del análisis.
    :return: Diccionario con estadísticas por subconjunto y el total.
    """
    subsets = ["train", "valid", "test"]
    results = {subset: {"total": 0, "multiple_leaves": 0} for subset in subsets}
    
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
                num_leaves = detect_multiple_leaves(image_path)
                results[subset]["total"] += 1
                if num_leaves > 1:
                    results[subset]["multiple_leaves"] += 1
    
    if debug:
        for subset, stats in results.items():
            print(f"{subset.upper()} - Total Images: {stats['total']}, Multiple Leaves: {stats['multiple_leaves']}")
    
    return results
