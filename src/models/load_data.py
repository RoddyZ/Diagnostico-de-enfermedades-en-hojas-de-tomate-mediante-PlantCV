import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

def cargar_generadores_personalizados(input_directory="dataset_clean_augmentation", img_size=(256, 256), batch_size=32):
    """Carga generadores de datos personalizados para múltiples salidas e imprime las clases de enfermedades."""

    TRAIN_DIR = os.path.join(input_directory, "train")
    VALID_DIR = os.path.join(input_directory, "valid")
    TEST_DIR = os.path.join(input_directory, "test")

    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_generator = datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    disease_classes = sorted(train_generator.class_indices.keys())
    num_classes_disease = len(disease_classes)
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    def generador_multisalida(generador):
        while True:
            batch_x, batch_y = next(generador)
            plant_labels = tf.convert_to_tensor([[1 if plant in disease else 0 for plant in plant_classes] for disease in [disease_classes[tf.argmax(label).numpy()] for label in batch_y]], dtype=tf.float32)

            yield batch_x, {"disease_output": batch_y, "plant_output": plant_labels}

    train_gen_multisalida = generador_multisalida(train_generator)
    valid_gen_multisalida = generador_multisalida(valid_generator)
    test_gen_multisalida = generador_multisalida(test_generator)

    print(f"📌 Clases de Enfermedades ({num_classes_disease}):")
    for disease_class in disease_classes:
        print(f"   - {disease_class}")

    print(f"\n📌 Clases de Plantas ({num_classes_plant}):")
    for plant_class in plant_classes:
        print(f"   - {plant_class}")

    print(f"\n🌱 Tipos de plantas: {plant_classes}")

    return train_gen_multisalida, valid_gen_multisalida, test_gen_multisalida, num_classes_disease, num_classes_plant, plant_classes, train_generator.samples, valid_generator.samples, test_generator.samples


def cargar_generadores_efficientnet(input_directory="dataset_clean_augmentation", img_size=(256, 256), batch_size=32):
    """Carga generadores de datos personalizados con EfficientNet."""

    TRAIN_DIR = os.path.join(input_directory, "train")
    VALID_DIR = os.path.join(input_directory, "valid")
    TEST_DIR = os.path.join(input_directory, "test")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # Cambiamos la función

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_generator = datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    disease_classes = sorted(train_generator.class_indices.keys())
    num_classes_disease = len(disease_classes)
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    def generador_multisalida(generador):
        while True:
            batch_x, batch_y = next(generador)
            plant_labels = tf.convert_to_tensor([[1 if plant in disease else 0 for plant in plant_classes] 
                                                 for disease in [disease_classes[tf.argmax(label).numpy()] for label in batch_y]], dtype=tf.float32)

            yield batch_x, {"disease_output": batch_y, "plant_output": plant_labels}

    train_gen_multisalida = generador_multisalida(train_generator)
    valid_gen_multisalida = generador_multisalida(valid_generator)
    test_gen_multisalida = generador_multisalida(test_generator)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")

    return train_gen_multisalida, valid_gen_multisalida, test_gen_multisalida, num_classes_disease, num_classes_plant, plant_classes, train_generator.samples, valid_generator.samples, test_generator.samples




def cargar_generadores_mobilenet(input_directory="dataset_clean_augmentation", img_size=(256, 256), batch_size=32):
    """Carga generadores de datos para MobileNetV2."""

    TRAIN_DIR = os.path.join(input_directory, "train")
    VALID_DIR = os.path.join(input_directory, "valid")
    TEST_DIR = os.path.join(input_directory, "test")
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)  # Cambiamos la función

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_generator = datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    disease_classes = sorted(train_generator.class_indices.keys())
    num_classes_disease = len(disease_classes)
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    def generador_multisalida(generador):
        while True:
            batch_x, batch_y = next(generador)
            plant_labels = tf.convert_to_tensor([[1 if plant in disease else 0 for plant in plant_classes] 
                                                 for disease in [disease_classes[tf.argmax(label).numpy()] for label in batch_y]], dtype=tf.float32)

            yield batch_x, {"disease_output": batch_y, "plant_output": plant_labels}

    train_gen_multisalida = generador_multisalida(train_generator)
    valid_gen_multisalida = generador_multisalida(valid_generator)
    test_gen_multisalida = generador_multisalida(test_generator)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")

    return train_gen_multisalida, valid_gen_multisalida, test_gen_multisalida, num_classes_disease, num_classes_plant, plant_classes, train_generator.samples, valid_generator.samples, test_generator.samples


def cargar_generadores_densenet(input_directory="dataset_clean_augmentation", img_size=(224, 224), batch_size=32):
    """Carga generadores de datos para DenseNet121."""
    
    TRAIN_DIR = os.path.join(input_directory, "train")
    VALID_DIR = os.path.join(input_directory, "valid")
    TEST_DIR = os.path.join(input_directory, "test")

    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    valid_generator = datagen.flow_from_directory(
        VALID_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_generator = datagen.flow_from_directory(
        TEST_DIR, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    disease_classes = sorted(train_generator.class_indices.keys())
    num_classes_disease = len(disease_classes)
    plant_classes = sorted(set([name.split("_")[0] for name in disease_classes]))
    num_classes_plant = len(plant_classes)

    def generador_multisalida(generador):
        while True:
            batch_x, batch_y = next(generador)
            plant_labels = tf.convert_to_tensor([[1 if plant in disease else 0 for plant in plant_classes] 
                                                 for disease in [disease_classes[tf.argmax(label).numpy()] for label in batch_y]], dtype=tf.float32)

            yield batch_x, {"disease_output": batch_y, "plant_output": plant_labels}

    train_gen_multisalida = generador_multisalida(train_generator)
    valid_gen_multisalida = generador_multisalida(valid_generator)
    test_gen_multisalida = generador_multisalida(test_generator)

    print(f"📌 Clases de Enfermedades: {num_classes_disease}")
    print(f"📌 Clases de Plantas: {num_classes_plant}")
    print(f"🌱 Tipos de plantas: {plant_classes}")

    return train_gen_multisalida, valid_gen_multisalida, test_gen_multisalida, num_classes_disease, num_classes_plant, plant_classes, train_generator.samples, valid_generator.samples, test_generator.samples
