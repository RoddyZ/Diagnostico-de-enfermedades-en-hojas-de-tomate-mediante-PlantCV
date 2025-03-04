import os
import cv2  # OpenCV para realizar volteos
from plantcv import plantcv as pcv
from tqdm import tqdm
from pathlib import Path

def data_augmentation(input_dir, output_dir):
    """
    Realiza data augmentation en las imágenes de entrenamiento y validación, y las guarda en una nueva carpeta.

    Parámetros:
        input_dir (str): Ruta al directorio con las imágenes segmentadas (dataset_segmented).
        output_dir (str): Ruta al directorio donde se guardarán las imágenes aumentadas (dataset_clean_augmentation).
    """
    # Crear el directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Recorrer las carpetas de train y valid
    for split in ['train', 'valid']:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)

        # Crear la carpeta de salida para el split actual
        Path(split_output_dir).mkdir(parents=True, exist_ok=True)

        # Obtener la lista de clases (subcarpetas)
        classes = os.listdir(split_input_dir)

        for class_name in tqdm(classes, desc=f"Aumentando {split}"):
            class_input_dir = os.path.join(split_input_dir, class_name)
            class_output_dir = os.path.join(split_output_dir, class_name)

            # Crear la carpeta de salida para la clase actual
            Path(class_output_dir).mkdir(parents=True, exist_ok=True)

            # Recorrer las imágenes en la carpeta de la clase
            for image_name in os.listdir(class_input_dir):
                image_path = os.path.join(class_input_dir, image_name)
                base_name, ext = os.path.splitext(image_name)

                try:
                    # Cargar la imagen usando PlantCV
                    img, _, _ = pcv.readimage(filename=image_path)

                    # Guardar la imagen original
                    output_path = os.path.join(class_output_dir, image_name)
                    pcv.print_image(img=img, filename=output_path)

                    # Aplicar rotaciones de 90°, 180° y 270°
                    for angle in [90, 180, 270]:
                        rotated_img = pcv.transform.rotate(img=img, rotation_deg=angle, crop=True)
                        output_path = os.path.join(class_output_dir, f"{base_name}_rot{angle}.jpg")
                        pcv.print_image(img=rotated_img, filename=output_path)

                    # Aplicar volteos horizontal y vertical con OpenCV
                    for flip_code, flip_name in [(0, "v"), (1, "h")]:  # 0 = vertical, 1 = horizontal
                        flipped_img = cv2.flip(img, flip_code)
                        output_path = os.path.join(class_output_dir, f"{base_name}_flip{flip_name}.jpg")
                        pcv.print_image(img=flipped_img, filename=output_path)

                except Exception as e:
                    print(f"Error procesando {image_name}: {e}")

    print(f"✅ Data augmentation completado. Imágenes guardadas en {output_dir}")
