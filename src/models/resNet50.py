import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def construir_modelo_multisalida(num_classes_disease, num_classes_plant, input_shape=(256, 256, 3)):
    """Construye un modelo ResNet50 con múltiples salidas."""
    # Capa de entrada
    input_tensor = Input(shape=input_shape)

    # ResNet50 base (sin la capa superior)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Congelar las capas base para el transfer learning
    for layer in base_model.layers:
        layer.trainable = False

    # Capa de pooling global
    x = GlobalAveragePooling2D()(base_model.output)

    # Salida para la enfermedad
    disease_output = Dense(num_classes_disease, activation='softmax', name='disease_output')(x)

    # Salida para la planta
    plant_output = Dense(num_classes_plant, activation='softmax', name='plant_output')(x)

    # Modelo final
    model = Model(inputs=input_tensor, outputs=[disease_output, plant_output])

    return model

def construir_modelo_multisalida_v2(num_classes_disease, num_classes_plant, input_shape=(256, 256, 3)):
    """Construye un modelo ResNet50 mejorado con múltiples salidas."""

    # Capa de entrada
    input_tensor = Input(shape=input_shape)

    # ResNet50 base (sin la capa superior)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Congelar todas las capas iniciales
    for layer in base_model.layers[:-30]:  # Descongelar las últimas 30 capas
        layer.trainable = False

    # Capa de pooling global
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)  # Normalización por lotes
    x = Dropout(0.5)(x)  # Dropout para evitar sobreajuste

    # Salida para la enfermedad
    disease_output = Dense(num_classes_disease, activation='softmax',kernel_regularizer=l2(0.01), name='disease_output')(x)

    # Salida para la planta
    plant_output = Dense(num_classes_plant, activation='softmax', kernel_regularizer=l2(0.01), name='plant_output')(x)

    # Modelo final
    model = Model(inputs=input_tensor, outputs=[disease_output, plant_output], name="modelResNetV2")

    return model
