import os
import cv2
import numpy as np
from tqdm import tqdm
from plantcv import plantcv as pcv

def correct_lighting(image_path, output_dir="dataset_processed", target_size=(256, 256), debug=False):
    """
    Corrige la iluminación de la imagen para mejorar la consistencia y la guarda en la carpeta de salida.
    """
    img, _, _ = pcv.readimage(image_path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # Asegurar tamaño normalizado
    corrected_img = pcv.white_balance(img, mode='hist')
    
    output_path = image_path.replace("dataset", output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcv.print_image(corrected_img, output_path)
    
    if debug:
        print(f"Corrected image saved at: {output_path}")
    
    return output_path

def process_dataset(dataset_path, output_dir="dataset_processed", debug=False):
    """
    Procesa masivamente todas las imágenes en train, test y valid, aplicando corrección de iluminación.
    """
    subsets = ["train", "test", "valid"]
    
    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        if not os.path.exists(subset_path):
            continue
        
        print(f"Procesando {subset}...")
        image_files = []
        
        for root, _, files in os.walk(subset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        for image_path in tqdm(image_files, desc=f"Corrigiendo iluminación en {subset}"):
            correct_lighting(image_path, output_dir, debug=debug)
    
    print("Procesamiento completo.")
