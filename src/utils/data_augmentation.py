import os
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
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid']:
        split_input_dir = os.path.join(input_dir, split)
        split_output_dir = os.path.join(output_dir, split)

        Path(split_output_dir).mkdir(parents=True, exist_ok=True)

        classes = os.listdir(split_input_dir)

        for class_name in tqdm(classes, desc=f"Aumentando {split}"):
            class_input_dir = os.path.join(split_input_dir, class_name)
            class_output_dir = os.path.join(split_output_dir, class_name)

            Path(class_output_dir).mkdir(parents=True, exist_ok=True)

            for image_name in os.listdir(class_input_dir):
                image_path = os.path.join(class_input_dir, image_name)

                try:
                    img, _, _ = pcv.readimage(filename=image_path)
                except Exception as e:
                    print(f"Error cargando {image_name}: {e}")
                    continue  

                base_name = os.path.splitext(image_name)[0]

                output_path = os.path.join(class_output_dir, f"{base_name}.jpg")
                if not os.path.exists(output_path):
                    pcv.print_image(img=img, filename=output_path)

                for angle in [90, 180, 270]:
                    rotated_img = pcv.transform.rotate(img=img, rotation_deg=angle, crop=True)
                    output_path = os.path.join(class_output_dir, f"{base_name}_rot{angle}.jpg")
                    if not os.path.exists(output_path):
                        pcv.print_image(img=rotated_img, filename=output_path)

                for flip_direction in ['horizontal', 'vertical']:
                    flipped_img = pcv.transform.flip(img=img, direction=flip_direction)
                    output_path = os.path.join(class_output_dir, f"{base_name}_flip{flip_direction[0]}.jpg")
                    if not os.path.exists(output_path):
                        pcv.print_image(img=flipped_img, filename=output_path)

    print(f"✅ Data augmentation completado. Imágenes guardadas en {output_dir}")
