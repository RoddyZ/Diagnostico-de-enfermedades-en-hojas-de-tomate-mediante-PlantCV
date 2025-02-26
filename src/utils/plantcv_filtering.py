import os
import cv2
import numpy as np
from tqdm import tqdm

class PlantCVFilter:
    def __init__(self, debug=False, blur_threshold=100, target_size=(256, 256)):
        self.debug = debug
        self.blur_threshold = blur_threshold
        self.target_size = target_size
    
    def is_blurry(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold
    
    def resize_image(self, image_path, output_path):
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
    
    def filter_and_normalize_dataset(self, dataset_path, output_path="filtered_dataset"):
        os.makedirs(output_path, exist_ok=True)
        splits = ['train', 'test', 'valid']
        
        for split in splits:
            split_path = os.path.join(dataset_path, split)
            output_split_path = os.path.join(output_path, split)
            os.makedirs(output_split_path, exist_ok=True)
            
            if not os.path.exists(split_path):
                continue
            
            print(f"Filtrando y normalizando {split}...")
            
            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category)
                output_category_path = os.path.join(output_split_path, category)
                os.makedirs(output_category_path, exist_ok=True)
                
                if not os.path.isdir(category_path):
                    continue
                
                for img_name in tqdm(os.listdir(category_path), desc=f"Procesando {category}"):
                    img_path = os.path.join(category_path, img_name)
                    output_img_path = os.path.join(output_category_path, img_name)
                    
                    if self.is_blurry(img_path):
                        continue  # Omitir imágenes borrosas
                    
                    self.resize_image(img_path, output_img_path)
                    
                print(f"Filtrado de {category} en {split} completado.")
        
        print("Filtrado y normalización completados.")
