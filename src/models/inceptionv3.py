from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


def construir_modelo_googlenet(num_classes_disease, num_classes_plant, capas_congeladas=20, dropout=0.5):
    """Construye un modelo basado en GoogleNet (InceptionV3) con dos salidas."""
    
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

    # Congelar las primeras capas
    for layer in base_model.layers[:capas_congeladas]:
        layer.trainable = False
    for layer in base_model.layers[capas_congeladas:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(dropout)(x)

    disease_output = Dense(num_classes_disease, activation="softmax", name="disease_output")(x)
    plant_output = Dense(num_classes_plant, activation="softmax", name="plant_output")(x)

    model = Model(inputs=base_model.input, outputs={"disease_output": disease_output, "plant_output": plant_output})

    return model