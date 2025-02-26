import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from tqdm import tqdm
import json
import multiprocessing

class PlantCVPreprocessor:
    def __init__(self, debug=False):
        self.debug = debug
    
    def count_leaves(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_leaves = len(contours)
        return num_leaves
    
    def detect_blurry_images(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian < 100
    
    def analyze_color(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        mean_color = cv2.mean(masked_img, mask=mask)
        return {"R": mean_color[2], "G": mean_color[1], "B": mean_color[0]}
    
    def analyze_texture(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5).var()
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5).var()
        return {"laplacian_var": laplacian, "sobel_x_var": sobelx, "sobel_y_var": sobely}
    
    def measure_affected_area(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        analysis = pcv.analyze.bound_horizontal(img, mask, line_position=50)
        return analysis
    
    def segment_leaf(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        gray = pcv.rgb2gray(img)
        threshold = pcv.threshold.otsu(gray, object_type='light')
        fill = pcv.fill(threshold, size=100)
        if self.debug:
            pcv.plot_image(fill)
        return fill
    
    def correct_lighting(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        corrected = pcv.white_balance(img, mode='hist')
        if self.debug:
            pcv.plot_image(corrected)
        return corrected
    
    def process_image(self, img_path, output_category_path):
        img_name = os.path.basename(img_path)
        corrected_img = self.correct_lighting(img_path)
        processed_img = self.segment_leaf(img_path)
        output_img_path = os.path.join(output_category_path, img_name)
        pcv.print_image(processed_img, output_img_path)
        
        data = {
            "image": img_name,
            "num_leaves": int(self.count_leaves(img_path)),  # Convertir a int
            "blurry": bool(self.detect_blurry_images(img_path)),  # Convertir a bool nativo de Python
            "color_analysis": {k: float(v) for k, v in self.analyze_color(img_path).items()},  # Convertir a float
            "texture_analysis": {k: float(v) for k, v in self.analyze_texture(img_path).items()},  # Convertir a float
            "affected_area": float(self.measure_affected_area(img_path))  # Convertir a float
        }
        return data
    
    def process_category(self, category_path, output_category_path):
        results = []
        image_paths = [os.path.join(category_path, img_name) for img_name in os.listdir(category_path)]
        
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.starmap(self.process_image, [(img_path, output_category_path) for img_path in image_paths]), total=len(image_paths), desc=f"Procesando {os.path.basename(category_path)}"))
        
        return results
    
    def preprocess_dataset(self, dataset_path, output_path="dataset_processed", save_metadata=True):
        os.makedirs(output_path, exist_ok=True)
        splits = ['train', 'test', 'valid']
        results = {}

        for split in splits:
            split_path = os.path.join(dataset_path, split)
            output_split_path = os.path.join(output_path, split)
            os.makedirs(output_split_path, exist_ok=True)

            if not os.path.exists(split_path):
                continue
            
            print(f"Procesando {split}...")
            results[split] = {}
            
            for category in os.listdir(split_path):
                category_path = os.path.join(split_path, category)
                output_category_path = os.path.join(output_split_path, category)
                os.makedirs(output_category_path, exist_ok=True)

                if not os.path.isdir(category_path):
                    continue
                
                results[split][category] = self.process_category(category_path, output_category_path)
                print(f"Procesamiento de {category} en {split} completado.")
        
        if save_metadata:
            with open(os.path.join(output_path, "processed_metadata.json"), "w") as f:
                json.dump(results, f, indent=4)
        
        return results

class PlantCVAnalyzer:
    def __init__(self, metadata_path):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def count_multiple_leaves(self):
        multiple_leaves = {}
        for split, categories in self.metadata.items():
            multiple_leaves[split] = {}
            for category, images in categories.items():
                count = sum(1 for img in images if img["num_leaves"] > 2)
                if count > 0:
                    multiple_leaves[split][category] = count
        return multiple_leaves

    def plot_color_distribution(self):
        import matplotlib.pyplot as plt
        from collections import Counter

        colors = {"train": [], "test": [], "valid": []}
        for split, categories in self.metadata.items():
            for category, images in categories.items():
                for img in images:
                    colors[split].append(tuple(img["color_analysis"].values()))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        for i, split in enumerate(["train", "test", "valid"]):
            color_counts = Counter(colors[split])
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            labels, values = zip(*sorted_colors)
            color_patches = [tuple(c / 255 for c in color) for color in labels]
            axes[i].barh(range(len(values)), values, color=color_patches)
            axes[i].set_yticks(range(len(values)))
            axes[i].set_yticklabels([str(color) for color in labels])
            axes[i].set_title(f"Color predominante en {split}")
        plt.tight_layout()
        plt.show()

    def blurry_images_summary(self):
        blurry_counts = {"train": 0, "test": 0, "valid": 0}