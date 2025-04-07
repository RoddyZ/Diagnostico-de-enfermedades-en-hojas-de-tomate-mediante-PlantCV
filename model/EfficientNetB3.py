from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def construir_modelo_efficientnet(num_classes_disease, num_classes_plant, input_shape=(256, 256, 3)):
    """Construye un modelo EfficientNet con múltiples salidas."""

    # Capa de entrada
    input_tensor = Input(shape=input_shape)

    # EfficientNet base (sin la capa superior)
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Congelar todas las capas iniciales y descongelar las últimas 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Capa de pooling global
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)  # Normalización
    x = Dropout(0.5)(x)  # Dropout para evitar sobreajuste

    # Salida para la enfermedad
    disease_output = Dense(num_classes_disease, activation='softmax',kernel_regularizer=l2(0.01), name='disease_output')(x)

    # Salida para la planta
    plant_output = Dense(num_classes_plant, activation='softmax',kernel_regularizer=l2(0.01), name='plant_output')(x)

    # Modelo final
    model = Model(inputs=input_tensor, outputs=[disease_output, plant_output], name="EfficientNetMultiOutput")

    return model
