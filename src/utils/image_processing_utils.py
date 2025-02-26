import os
import cv2
import numpy as np
from plantcv import plantcv as pcv

def blur_background(image_path, output_dir="dataset_processed", blur_ksize=15, debug=False):
    """
    Aplica un desenfoque al fondo de la imagen manteniendo la hoja nítida.
    :param image_path: Ruta de la imagen de entrada.
    :param output_dir: Carpeta donde se guardará la imagen procesada.
    :param blur_ksize: Tamaño del kernel de desenfoque (mayor número = más desenfoque).
    :param debug: Si es True, muestra el resultado.
    """
    img, _, _ = pcv.readimage(image_path)
    grayscale_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary_img = pcv.threshold.binary(grayscale_img, threshold=128, object_type='light')
    filled_img = pcv.fill_holes(binary_img)
    # Corrección: Incluir ksize en pcv.dilate
    mask = pcv.dilate(filled_img, ksize=5, i=2)  # Usar ksize=5 (o el tamaño que prefieras)

    # Asegurar que la máscara es de 3 canales para aplicarla correctamente
    mask_3c = cv2.merge([mask] * 3)
    blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred_background = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    result = np.where(mask_3c == 255, img, blurred_background)

    output_path = image_path.replace("dataset", output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcv.print_image(result, output_path)

    if debug:
        print(f"Blurred background image saved at: {output_path}")

    return output_path

def correct_lighting(image_path, output_dir="dataset_processed", debug=False):
    """
    Corrige la iluminación de la imagen para mejorar la consistencia y la guarda en la carpeta de salida.
    :param image_path: Ruta de la imagen de entrada.
    :param output_dir: Carpeta donde se guardará la imagen procesada.
    :param debug: Si es True, muestra el resultado.
    """
    img, _, _ = pcv.readimage(image_path)
    corrected_img = pcv.white_balance(img, mode='hist')
    
    output_path = image_path.replace("dataset", output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pcv.print_image(corrected_img, output_path)
    
    if debug:
        print(f"Corrected image saved at: {output_path}")
    
    return output_path
