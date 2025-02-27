import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

class PlantCVFilter:
    def __init__(self, debug=False, blur_threshold=100, target_size=(256, 256), deprecated_path="dataset_deprecated"):
        self.debug = debug
        self.blur_threshold = blur_threshold
        self.target_size = target_size
        self.deprecated_path = deprecated_path
    
    def is_blurry(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold
    
    def resize_image(self, image_path, output_path):
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
    
    def move_to_deprecated(self, image_path, dataset_path):
        """
        Mueve imágenes borrosas a la carpeta 'dataset_deprecated', manteniendo la jerarquía original.
        """
        try:
            relative_path = os.path.relpath(image_path, dataset_path)
            new_path = os.path.join(self.deprecated_path, *relative_path.split(os.sep))
            new_path = os.path.abspath(new_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(image_path, new_path)
        except Exception as e:
            print(f"Error moving {image_path} to deprecated: {e}")
    
    def filter_and_normalize_dataset(self, dataset_path, output_path="filtered_dataset"):
        os.makedirs(output_path, exist_ok=True)
        splits = ['train', 'test', 'valid']
        blurry_removed = 0  # Contador de imágenes borrosas movidas

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
                        self.move_to_deprecated(img_path, dataset_path)
                        blurry_removed += 1  # Contar la imagen borrosa movida
                        continue  # Omitir imágenes borrosas

                    self.resize_image(img_path, output_img_path)

                print(f"Filtrado de {category} en {split} completado.")

        print("Filtrado y normalización completados.")

        # 🔹 Devolver información sobre el número de imágenes movidas
        return {"blurry_moved": blurry_removed}
