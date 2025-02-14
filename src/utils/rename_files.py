import os
import shutil

def rename_images_in_directory(base_directory):
    # Crear las carpetas necesarias para 'train' y 'valid'
    for subset in ['train', 'valid']:
        subset_path = os.path.join(base_directory, subset)

        if os.path.exists(subset_path):
            for category in os.listdir(subset_path):
                category_path = os.path.join(subset_path, category)

                # Verificar que sea una carpeta
                if os.path.isdir(category_path):
                    print(f"Renombrando imágenes en: {category_path}")

                    # Contador para las imágenes secuenciales
                    counter = 1
                    for filename in os.listdir(category_path):
                        file_path = os.path.join(category_path, filename)

                        # Verificar que sea un archivo y no una subcarpeta
                        if os.path.isfile(file_path):
                            # Obtener la extensión original del archivo
                            _, ext = os.path.splitext(filename)
                            if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:  # Tipos de imágenes válidas
                                # Crear el nuevo nombre de archivo secuencial
                                new_filename = f"{subset}_{category.lower().replace(' ', '_')}_{counter}{ext}"
                                new_file_path = os.path.join(category_path, new_filename)

                                # Renombrar el archivo
                                os.rename(file_path, new_file_path)
                                # Incrementar el contador
                                counter += 1

