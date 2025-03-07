import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import shutil

class DatasetAnalyzer:
    def _init_(self, dataset_path):
        """
        Inicializa el analizador del dataset.
        :param dataset_path: Ruta principal donde están las carpetas train, test y valid.
        """
        self.dataset_path = dataset_path
        self.splits = ["train", "test", "valid"]
        self.classes = sorted(os.listdir(os.path.join(self.dataset_path, "train")))
        self.image_counts = {split: defaultdict(int) for split in self.splits}
        self.sample_images = {}
    
    def count_images(self):
        """
        Cuenta la cantidad de imágenes en cada conjunto (train, test, valid) y cada clase.
        """
        for split in self.splits:
            for cls in self.classes:
                class_path = os.path.join(self.dataset_path, split, cls)
                images = os.listdir(class_path) if os.path.exists(class_path) else []
                self.image_counts[split][cls] = len(images)
                
                # Tomar una imagen aleatoria si aún no se ha seleccionado
                if cls not in self.sample_images and images:
                    self.sample_images[cls] = os.path.join(class_path, random.choice(images))
    
    def summarize_dataset(self):
        """
        Imprime un resumen del dataset en términos de cantidad de clases e imágenes por conjunto.
        """
        summary = "Resumen del Dataset:\n---------------------\n"
        for split in self.splits:
            summary += f"{split.capitalize()} Set:\n"
            for cls, count in self.image_counts[split].items():
                summary += f"  - {cls}: {count} imágenes\n"
            summary += "\n"
        
        print(summary)
    
    def visualize_sample_images(self):
        """
        Visualiza una imagen aleatoria de cada clase en una matriz 5x7.
        """
        num_classes = len(self.sample_images)
        rows = (num_classes // 7) + (1 if num_classes % 7 != 0 else 0)
        
        fig, axes = plt.subplots(rows, 7, figsize=(20, 3 * rows))
        axes = axes.reshape(-1) if rows > 1 else [axes]
        
        for idx, (cls, img_path) in enumerate(self.sample_images.items()):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(cls, fontsize=8)
        
        # Ocultar los ejes vacíos
        for idx in range(num_classes, len(axes)):
            axes[idx].axis("off")
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self):
        """
        Crea gráficos de barras mostrando la cantidad de imágenes por clase en cada conjunto.
        """
        class_labels = self.classes
        train_counts = [self.image_counts["train"][cls] for cls in class_labels]
        test_counts = [self.image_counts["test"][cls] for cls in class_labels]
        valid_counts = [self.image_counts["valid"][cls] for cls in class_labels]

        # Dividir en 4 gráficos para mejorar legibilidad (10, 10, 10, 8 clases por gráfico)
        class_splits = [10, 10, 10, 8]
        start = 0

        for i, num_classes in enumerate(class_splits):
            fig, ax = plt.subplots(figsize=(15, 6))
            x_labels = class_labels[start:start + num_classes]
            x_indexes = np.arange(len(x_labels))

            ax.bar(x_indexes - 0.2, train_counts[start:start + num_classes], width=0.2, label="Train")
            ax.bar(x_indexes, test_counts[start:start + num_classes], width=0.2, label="Test")
            ax.bar(x_indexes + 0.2, valid_counts[start:start + num_classes], width=0.2, label="Valid")

            ax.set_xticks(x_indexes)
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_ylabel("Número de imágenes")
            ax.set_title(f"Distribución de Clases (Parte {i+1})")
            ax.legend()
            
            plt.show()
            start += num_classes


def dataset_summary(dataset_path):
    """
    Función principal para analizar y visualizar el dataset.
    :param dataset_path: Ruta principal del dataset.
    """
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.count_images()
    analyzer.summarize_dataset()
    analyzer.visualize_sample_images()
    analyzer.plot_class_distribution()


def create_test_set(train_dir, test_dir, percentage=10):
    """
    Crea un conjunto de prueba a partir de un directorio de entrenamiento.

    Args:
        train_dir (str): Ruta al directorio de entrenamiento.
        test_dir (str): Ruta al directorio de prueba.
        percentage (int, optional): Porcentaje de imágenes a mover al conjunto de prueba. Defaults to 10.
    """

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for subdir in os.listdir(train_dir):
        subdir_path = os.path.join(train_dir, subdir)
        if os.path.isdir(subdir_path):
            test_subdir_path = os.path.join(test_dir, subdir)
            if not os.path.exists(test_subdir_path):
                os.makedirs(test_subdir_path)

            images = os.listdir(subdir_path)
            num_test_images = int(len(images) * percentage / 100)
            test_images = random.sample(images, num_test_images)

            for image in test_images:
                src_path = os.path.join(subdir_path, image)
                dest_path = os.path.join(test_subdir_path, image)
                shutil.move(src_path, dest_path)