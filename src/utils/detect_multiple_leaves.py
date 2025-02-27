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
    """
    img = cv2.imread(image_path)
    if img is None:
        return True  # Si la imagen no se puede leer, descartarla
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = [np.rad2deg(line[0][1]) - 90 for line in lines]
        mean_angle = np.mean(angles)
        if abs(mean_angle) > angle_threshold:
            return True
    
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    shadow_pixels = np.sum(hist[:shadow_threshold]) / np.sum(hist)
    if shadow_pixels > 0.3:
        return True
    
    return False

def copy_to_deprecated(image_path, dataset_path, deprecated_path="dataset_deprecated"):
    """
    Copia imágenes descartadas o con múltiples hojas a la carpeta 'dataset_deprecated', manteniendo la jerarquía original.
    """
    try:
        relative_path = os.path.relpath(image_path, dataset_path)
        new_path = os.path.join(deprecated_path, *relative_path.split(os.sep))
        new_path = os.path.abspath(new_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(image_path, new_path)
        print(f"Copied to deprecated: {new_path}")
    except Exception as e:
        print(f"Error copying {image_path} to deprecated: {e}")

def detect_multiple_leaves(args):
    """
    Función compatible con multiprocessing para detectar múltiples hojas.
    """
    image_path, debug = args
    if detect_rotation_and_shadows(image_path):
        return None, image_path  # Indicar que la imagen fue descartada
    
    img, _, _ = pcv.readimage(image_path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    grayscale_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    
    binary_img = pcv.threshold.otsu(grayscale_img)
    filled_img = pcv.fill_holes(binary_img)
    mask = pcv.dilate(filled_img, ksize=5, i=2)
    
    labeled_objects = pcv.watershed_segmentation(img, mask)
    labeled_binary = np.where(labeled_objects > 0, 255, 0).astype(np.uint8)
    
    contours, _ = cv2.findContours(labeled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_leaf_area = 5000
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_leaf_area]
    merged_contours = [cv2.convexHull(cnt) for cnt in filtered_contours]
    
    num_leaves = len(merged_contours)
    
    if debug:
        print(f"Leaves detected: {num_leaves} in {image_path}")
        debug_img = img.copy()
        cv2.drawContours(debug_img, merged_contours, -1, (0, 255, 0), 2)
        pcv.print_image(debug_img, "segmented_debug.png")
    
    return num_leaves, image_path

def analyze_dataset(dataset_path, debug=False):
    """
    Recorre un dataset completo y cuenta cuántas imágenes tienen más de una hoja usando multiprocessing.
    También descarta imágenes con rotaciones extremas o sombras y las copia a la carpeta 'dataset_deprecated'.
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
                    image_path = os.path.abspath(image_path)
                    image_paths.append((image_path, debug))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(detect_multiple_leaves, image_paths), total=len(image_paths), desc="Procesando imágenes"))
    
    index = 0
    for subset in subsets:
        for root, _, files in os.walk(os.path.join(dataset_path, subset)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    result = results_list[index]
                    index += 1
                    image_path = os.path.join(root, file)
                    image_path = os.path.abspath(image_path)
                    
                    if result is None:
                        copy_to_deprecated(image_path, dataset_path)
                        results[subset]["discarded"].append(file)
                        continue
                    
                    num_leaves, image_path = result
                    if num_leaves is not None:
                        results[subset]["total"] += 1
                        if num_leaves > 1:
                            results[subset]["multiple_leaves"] += 1
                            results[subset]["files"].append(image_path)
                            copy_to_deprecated(image_path, dataset_path)  # Copiar imágenes con múltiples hojas
    
    if debug:
        for subset, stats in results.items():
            print(f"{subset.upper()} - Total Images: {stats['total']}, Multiple Leaves: {stats['multiple_leaves']}, Discarded: {len(stats['discarded'])}")
    
    return results
