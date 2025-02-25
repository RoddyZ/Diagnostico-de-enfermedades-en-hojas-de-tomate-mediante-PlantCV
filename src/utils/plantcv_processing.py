import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from PIL import Image
from tqdm import tqdm
import json

class PlantCVProcessor:
    def __init__(self, debug=False):
        self.debug = debug
    
    def segment_leaf(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        gray = pcv.rgb2gray(img)
        threshold = pcv.threshold.otsu(gray, object_type='light')
        fill = pcv.fill(threshold, size=100)
        if self.debug:
            pcv.plot_image(fill)
        return fill
    
    def convert_to_grayscale(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        gray = pcv.rgb2gray(img)
        if self.debug:
            pcv.plot_image(gray)
        return gray
    
    def correct_lighting(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        corrected = pcv.white_balance(img, mode='hist')
        if self.debug:
            pcv.plot_image(corrected)
        return corrected
    
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
        return {
            "laplacian_var": laplacian,
            "sobel_x_var": sobelx,
            "sobel_y_var": sobely
        }
    
    def measure_affected_area(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        analysis = pcv.analyze.bound_horizontal(img, mask, line_position=50)
        return analysis
    
    def detect_blurry_images(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian < 100
    
    def count_leaves(self, image_path):
        img, _, _ = pcv.readimage(image_path)
        mask = self.segment_leaf(image_path)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_leaves = len(contours)
        return num_leaves
    
    def process_dataset(self, dataset_path, output_path="dataset_processed", save_metadata=True):
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
                
                results[split][category] = []

                for img_name in tqdm(os.listdir(category_path), desc=f"Procesando {category}", disable=not self.debug):
                    img_path = os.path.join(category_path, img_name)
                    
                    corrected_img = self.correct_lighting(img_path)
                    gray_img = self.convert_to_grayscale(img_path)
                    processed_img = self.segment_leaf(img_path)
                    
                    output_img_path = os.path.join(output_category_path, img_name)
                    pcv.print_image(processed_img, output_img_path)
                    
                    data = {
                        "image": img_name,
                        "num_leaves": self.count_leaves(img_path),
                        "blurry": self.detect_blurry_images(img_path),
                        "color_analysis": self.analyze_color(img_path),
                        "texture_analysis": self.analyze_texture(img_path),
                        "affected_area": self.measure_affected_area(img_path)
                    }
                    results[split][category].append(data)
                
                print(f"Procesamiento de {category} en {split} completado.")
        
        if save_metadata:
            with open(os.path.join(output_path, "processed_metadata.json"), "w") as f:
                json.dump(results, f, indent=4)
        
        return results