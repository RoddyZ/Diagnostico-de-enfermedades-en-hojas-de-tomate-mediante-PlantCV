import os
import cv2
import numpy as np
import pandas as pd
from plantcv import plantcv as pcv
from tqdm import tqdm
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

def extract_features(input_dir, output_csv):
    """
    Extrae características de color, textura y forma de las imágenes segmentadas y las guarda en un archivo CSV.

    Parámetros:
        input_dir (str): Ruta al directorio con las imágenes segmentadas (dataset_segmented).
        output_csv (str): Ruta al archivo CSV donde se guardarán las características.
    """
    # Lista para almacenar las características
    features_list = []

    # Recorrer las carpetas de train, valid
    for split in ['train', 'valid']:  # , 'test'
        split_input_dir = os.path.join(input_dir, split)

        # Recorrer las clases (carpetas) dentro de cada split
        for class_name in os.listdir(split_input_dir):
            class_input_dir = os.path.join(split_input_dir, class_name)

            # Recorrer las imágenes en la carpeta de la clase
            for image_name in tqdm(os.listdir(class_input_dir), desc=f"Procesando {split}/{class_name}"):
                image_path = os.path.join(class_input_dir, image_name)

                # Cargar la imagen usando PlantCV
                img, _, _ = pcv.readimage(filename=image_path)

                # Extraer características de color
                color_features = _extract_color_features(img)

                # Extraer características de textura
                texture_features = _extract_texture_features(img)

                # Extraer características de forma
                shape_features = _extract_shape_features(img)

                # Combinar todas las características
                features = {
                    "image_path": image_path,
                    "class": class_name,
                    "split": split,
                    **color_features,
                    **texture_features,
                    **shape_features,
                }

                # Agregar las características a la lista
                features_list.append(features)

    # Convertir la lista de características en un DataFrame
    features_df = pd.DataFrame(features_list)

    # Guardar el DataFrame en un archivo CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Extracción de características completada. Datos guardados en {output_csv}")

def _extract_color_features(img):
    """
    Extrae características de color de la imagen.

    Parámetros:
        img (numpy.ndarray): Imagen de la hoja segmentada.

    Retorna:
        dict: Diccionario con las características de color.
    """
    # Convertir la imagen a diferentes espacios de color
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Calcular histogramas de color
    hist_rgb = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_hsv = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_lab = cv2.calcHist([lab_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalizar los histogramas
    hist_rgb = cv2.normalize(hist_rgb, hist_rgb).flatten()
    hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
    hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()

    # Crear un diccionario con las características de color
    color_features = {}
    for i, value in enumerate(hist_rgb):
        color_features[f"hist_rgb_{i}"] = value
    for i, value in enumerate(hist_hsv):
        color_features[f"hist_hsv_{i}"] = value
    for i, value in enumerate(hist_lab):
        color_features[f"hist_lab_{i}"] = value

    return color_features

def _extract_texture_features(img):
    """
    Extrae características de textura de la imagen.

    Parámetros:
        img (numpy.ndarray): Imagen de la hoja segmentada.

    Retorna:
        dict: Diccionario con las características de textura.
    """
    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcular la matriz de co-ocurrencia de niveles de gris (GLCM)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], symmetric=True, normed=True)

    # Extraer características de textura de la GLCM
    contrast = graycoprops(glcm, "contrast")[0, 0]
    dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    # Crear un diccionario con las características de textura
    texture_features = {
        "texture_contrast": contrast,
        "texture_dissimilarity": dissimilarity,
        "texture_homogeneity": homogeneity,
        "texture_energy": energy,
        "texture_correlation": correlation,
    }

    return texture_features

def _extract_shape_features(img):
    """
    Extrae características de forma de la imagen.

    Parámetros:
        img (numpy.ndarray): Imagen de la hoja segmentada.

    Retorna:
        dict: Diccionario con las características de forma.
    """
    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarizar la imagen
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Verificar si se encontraron contornos
    if len(contours) == 0:
        # Si no se encontraron contornos, asignar valores predeterminados
        area = 0
        perimeter = 0
        aspect_ratio = 0
    else:
        # Calcular características de forma
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        aspect_ratio = float(img.shape[1]) / img.shape[0]  # Relación de aspecto (ancho/alto)

    # Crear un diccionario con las características de forma
    shape_features = {
        "shape_area": area,
        "shape_perimeter": perimeter,
        "shape_aspect_ratio": aspect_ratio,
    }

    return shape_features