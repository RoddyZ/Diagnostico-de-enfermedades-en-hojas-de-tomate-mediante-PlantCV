import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensión para que coincida con batch_size=1
    img_array /= 255.0  # Normalizar
    return img_array

def predict(image_path, model_path):
    # Cargar el modelo
    model = load_model(model_path)

    # Cargar y preprocesar la imagen
    img_array = load_and_preprocess_image(image_path)

    # Realizar la predicción
    predictions = model.predict(img_array)

    # Si el modelo tiene 2 salidas (por ejemplo, una para enfermedad y otra para especie)
    if isinstance(predictions, list) and len(predictions) == 2:
        disease_pred = np.argmax(predictions[0], axis=1)[0]  # Predicción de la enfermedad
        plant_pred = np.argmax(predictions[1], axis=1)[0]    # Predicción de la especie

        disease_confidence = np.max(predictions[0])  # Confianza en la predicción de enfermedad
        plant_confidence = np.max(predictions[1])    # Confianza en la predicción de la especie

        return disease_pred, disease_confidence, plant_pred, plant_confidence
    else:
        # Si el modelo tiene una sola salida, asumimos que es clasificación única
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        return predicted_class, confidence
