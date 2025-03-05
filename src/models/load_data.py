import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

def cargarImagenes(input_directory="dataset_clean_augmentation"):
    # Verificar si la carpeta principal existe
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"⚠️ El directorio {input_directory} no existe.")

    # Directorios del dataset
    TRAIN_DIR = os.path.join(input_directory, "train")
    VALID_DIR = os.path.join(input_directory, "valid")
    TEST_DIR = os.path.join(input_directory, "test")

    for directory in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"⚠️ El directorio {directory} no existe o está vacío.")

    # Tamaño de imagen y batch
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32

    # Preprocesamiento específico para ResNet50
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Función para generar etiquetas de planta y enfermedad
    def generate_labels(class_names):
        plant_labels = []
        disease_labels = []
        for class_name in class_names:
            plant, disease = class_name.split("_", 1)  # Dividir en planta y enfermedad
            plant_labels.append(plant)
            disease_labels.append(disease)
        return plant_labels, disease_labels

    # Obtener nombres de clases (subcarpetas)
    disease_classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))

    # Crear diccionarios para mapear clases a índices
    plant_to_index = {plant: idx for idx, plant in enumerate(plant_classes)}
    disease_to_index = {disease: idx for idx, disease in enumerate(disease_classes)}

    # Generadores de imágenes con etiquetas personalizadas
    def custom_generator(directory):
        generator = datagen.flow_from_directory(
            directory, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode=None, shuffle=False
        )
        for images, filenames in generator:
            # Obtener etiquetas de planta y enfermedad desde los nombres de archivo
            plant_labels = []
            disease_labels = []
            for filename in filenames:
                class_name = os.path.basename(os.path.dirname(filename))
                plant, disease = class_name.split("_", 1)
                plant_labels.append(plant_to_index[plant])
                disease_labels.append(disease_to_index[disease])
            yield images, [np.array(plant_labels), np.array(disease_labels)]

    # Crear generadores
    train_generator = custom_generator(TRAIN_DIR)
    valid_generator = custom_generator(VALID_DIR)
    test_generator = custom_generator(TEST_DIR)

    # Número de clases
    num_classes_disease = len(disease_classes)
    num_classes_plant = len(plant_classes)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")
    return train_generator, valid_generator, test_generator, num_classes_disease, num_classes_plant, plant_classes