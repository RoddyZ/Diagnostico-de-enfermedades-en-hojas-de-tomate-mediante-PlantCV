import os
import sys

def renombrar_carpetas(directorio_principal):
    """
    Renombra las carpetas en los directorios 'train' y 'valid' dentro del directorio principal,
    reemplazando los guiones bajos dobles '___' por uno solo '_', eliminando paréntesis, comas y espacios en blanco.

    Args:
        directorio_principal (str): Ruta al directorio principal que contiene 'train' y 'valid'.
    """

    for subdirectorio in ['train', 'valid','test']:
        ruta_subdirectorio = os.path.join(directorio_principal, subdirectorio)

        if os.path.exists(ruta_subdirectorio):
            for nombre_carpeta in os.listdir(ruta_subdirectorio):
                ruta_carpeta_antigua = os.path.join(ruta_subdirectorio, nombre_carpeta)

                if os.path.isdir(ruta_carpeta_antigua):
                    nombre_carpeta_nuevo = nombre_carpeta.replace(' ', '_')
                    nombre_carpeta_nuevo = nombre_carpeta_nuevo.replace('___', '_')
                    nombre_carpeta_nuevo = nombre_carpeta_nuevo.replace('(', '')
                    nombre_carpeta_nuevo = nombre_carpeta_nuevo.replace(')', '')
                    nombre_carpeta_nuevo = nombre_carpeta_nuevo.replace(',', '')
                    ruta_carpeta_nueva = os.path.join(ruta_subdirectorio, nombre_carpeta_nuevo)

                    if nombre_carpeta != nombre_carpeta_nuevo:
                        os.rename(ruta_carpeta_antigua, ruta_carpeta_nueva)
                        print(f'Renombrado: {nombre_carpeta} -> {nombre_carpeta_nuevo}')
        else:
            print(f'El subdirectorio {subdirectorio} no existe en {directorio_principal}')


def rename_images_in_directory(base_directory):
    # Crear las carpetas necesarias para 'train' y 'valid'
    for subset in ['train', 'valid','test']:
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

