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

    # Crear generadores
    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    test_generator = datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )

    # Obtener clases a partir del generador
    disease_classes = sorted(train_generator.class_indices.keys())
    num_classes_disease = len(disease_classes)

    # Extraer nombres de plantas desde las enfermedades
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")

    return train_generator, valid_generator, test_generator, num_classes_disease, num_classes_plant, plant_classes
