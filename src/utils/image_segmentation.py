import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def segment_leaves(input_dir, output_dir):
    """
    Segmenta las hojas de las imágenes en el directorio de entrada y guarda las imágenes segmentadas
    en el directorio de salida, respetando la estructura de carpetas.

    Parámetros:
        input_dir (str): Ruta al directorio con las imágenes originales (dataset_processed_filtered).
        output_dir (str): Ruta al directorio donde se guardarán las imágenes segmentadas (dataset_segmented).
    """
    # Crear el directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Recorrer las carpetas de train, valid
    for split in ['train', 'valid']:  # , 'test'
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)

        # Crear la carpeta de salida para el split actual
        Path(split_output_dir).mkdir(parents=True, exist_ok=True)

        # Recorrer las clases (carpetas) dentro de cada split
        for class_name in os.listdir(split_input_dir):
            class_input_dir = os.path.join(split_input_dir, class_name)
            class_output_dir = os.path.join(split_output_dir, class_name)

            # Crear la carpeta de salida para la clase actual
            Path(class_output_dir).mkdir(parents=True, exist_ok=True)

            # Recorrer las imágenes en la carpeta de la clase
            for image_name in tqdm(os.listdir(class_input_dir), desc=f"Procesando {split}/{class_name}"):
                image_path = os.path.join(class_input_dir, image_name)
                output_path = os.path.join(class_output_dir, image_name)

                # Cargar la imagen
                img = cv2.imread(image_path)

                # Convertir a espacio de color HSV para facilitar la segmentación
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Crear una máscara basada en el rango de color verde (para hojas)
                lower_green = np.array([25, 40, 40])  # Valores mínimos de H, S, V
                upper_green = np.array([90, 255, 255])  # Valores máximos de H, S, V
                green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

                # Convertir a escala de grises para detectar bordes
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Aplicar desenfoque gaussiano para reducir el ruido
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Detectar bordes con Canny
                edges = cv2.Canny(blurred, 50, 150)

                # Combinar la máscara de color verde con la máscara de bordes
                combined_mask = cv2.bitwise_or(green_mask, edges)

                # Aplicar operaciones morfológicas para mejorar la máscara
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Rellenar pequeños agujeros
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)  # Eliminar pequeños objetos

                # Aplicar la máscara combinada a la imagen original
                segmented_img = cv2.bitwise_and(img, img, mask=combined_mask)

                # Guardar la imagen segmentada
                cv2.imwrite(output_path, segmented_img)

    print(f"Segmentación completada. Imágenes guardadas en {output_dir}")