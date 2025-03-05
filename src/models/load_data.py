#import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def cargarImagenes(input_directory ="dataset_clean_augmentation"):
    # Directorios del dataset
    DATASET_DIR = input_directory
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    VALID_DIR = os.path.join(DATASET_DIR, "valid")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    # Tamaño de imagen y batch
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32

    # Generadores de imágenes
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
    )
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )

    # Extraer nombres de enfermedades (subcarpetas)
    disease_classes = sorted(os.listdir(TRAIN_DIR))
    num_classes_disease = len(disease_classes)

    # Extraer nombres de plantas desde las enfermedades
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")
    return train_generator, valid_generator, test_generator, num_classes_disease, num_classes_plant, plant_classes
