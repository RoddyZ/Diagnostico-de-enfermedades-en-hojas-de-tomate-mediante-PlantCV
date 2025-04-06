import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Clases completas basadas en tu dataset (orden correcto)
DISEASE_CLASSES = [
    'Apple_Apple_scab',
    'Apple_Black_rot',
    'Apple_Cedar_apple_rust',
    'Apple_healthy',
    'Blueberry_healthy',
    'Cherry_including_sour_Powdery_mildew',
    'Cherry_including_sour_healthy',
    'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_maize_Common_rust',
    'Corn_maize_Northern_Leaf_Blight',
    'Corn_maize_healthy',
    'Grape_Black_rot',
    'Grape_Esca_Black_Measles',
    'Grape_Leaf_blight_Isariopsis_Leaf_Spot',
    'Grape_healthy',
    'Orange_Haunglongbing_Citrus_greening',
    'Peach_Bacterial_spot',
    'Peach_healthy',
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Potato_Early_blight',
    'Potato_Late_blight',
    'Potato_healthy',
    'Raspberry_healthy',
    'Soybean_healthy',
    'Squash_Powdery_mildew',
    'Strawberry_Leaf_scorch',
    'Strawberry_healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Diccionarios de traducción (mantienen el mismo orden)
PLANT_TRANSLATIONS = {
    'Apple': 'Manzano',
    'Blueberry': 'Arándano',
    'Cherry': 'Cerezo',
    'Corn': 'Maíz',
    'Grape': 'Uva',
    'Orange': 'Naranjo',
    'Peach': 'Durazno',
    'Pepper': 'Pimiento',
    'Potato': 'Papa',
    'Raspberry': 'Frambuesa',
    'Soybean': 'Soja',
    'Squash': 'Calabaza',
    'Strawberry': 'Fresa',
    'Tomato': 'Tomate'
}


DISEASE_TRANSLATIONS = {
    'Apple_scab'                                : 'Sarna',
    'Black_rot'                                 : 'la Pudrición negra',
    'Cedar_apple_rust'                          : 'la Roya del Manzano y del Cedro',
    'Apple_healthy'                             : 'Manzana sana',
    'Blueberry_healthy'                         : 'Arándano sano',
    'including_sour_healthy'                    : 'Cereza sana',
    'including_sour_Powdery_mildew'             : 'Mildiu Polvoriento Ácido',
    'maize_Cercospora_leaf_spot_Gray_leaf_spot' : 'la Mancha Foliar de Cercospora (Mancha Gris de la Hoja)',
    'maize_Common_rust'                         : 'la Roya común',
    'maize_healthy'                             : 'Maíz sano',
    'maize_Northern_Leaf_Blight'                : 'Tizón del norte',
    'Grape_Black_rot'                           : 'la Pudrición negra',
    'Esca_Black_Measles'                        : 'Esca (sarampión negro)',
    'Grape_healthy'                             : 'Uva sana',
    'Leaf_blight_Isariopsis_Leaf_Spot'          : 'la Mancha Foliar por Isariopsis',
    'Haunglongbing_Citrus_greening'             : 'Huanglongbing (enverdecimiento de los cítricos)',
    'Bacterial_spot'                            : 'la Mancha bacteriana',
    'Peach_healthy'                             : 'Melocotón sano',
    'bell_Bacterial_spot'                       : 'la Mancha bacteriana',
    'bell_healthy'                              : 'Pimiento sano',
    'Early_blight'                              : 'Tizón temprano',
    'Potato_healthy'                            : 'Papa sana',
    'Late_blight'                               : 'Tizón tardío',
    'Raspberry_healthy'                         : 'Frambuesa sana',
    'Soybean_healthy'                           : 'Soja sana',
    'Powdery_mildew'                            : 'Mildiu polvoriento',
    'Strawberry_healthy'                        : 'Fresa sana',
    'Leaf_scorch'                               : 'la Quemadura de la hoja',
    'Bacterial_spot'                            : 'la Mancha bacteriana',
    'Early_blight'                              : 'Tizón temprano',
    'Tomato_healthy'                            : 'Tomate sano',
    'Late_blight'                               : 'Tizón tardío',
    'Leaf_Mold'                                 : 'Moho',
    'Septoria_leaf_spot'                        : 'la Mancha foliar por Septoria',
    'Spider_mites_Two-spotted_spider_mite'      : 'Ácaros araña (ácaro araña de dos manchas)',
    'Target_Spot'                               : 'la Mancha Objetivo',
    'Tomato_mosaic_virus'                       : 'el Virus del mosaico',
    'Tomato_Yellow_Leaf_Curl_Virus'             : 'el Virus del Rizado Amarillo'
}

PLANT_CLASSES = sorted(list(set([name.split('_')[0] for name in DISEASE_CLASSES])))

class PlantDiseasePredictor:
    def __init__(self, model_path, weights_path):
        """
        Carga el modelo ResNet50 con los pesos pre-entrenados.

        Args:
            model_path: Ruta al archivo .h5 del modelo completo
            weights_path: Ruta al archivo .hdf5 con los pesos
        """
        self.model = load_model(model_path)
        self.model.load_weights(weights_path)
        self.img_size = (256, 256)  # Tamaño usado en el entrenamiento
        self.disease_classes = DISEASE_CLASSES
        self.plant_classes = PLANT_CLASSES

    def preprocess_image(self, img_path):
        """
        Preprocesa una imagen para que sea compatible con el modelo.

        Args:
            img_path: Ruta a la imagen a predecir

        Returns:
            Imagen preprocesada como numpy array
        """
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Preprocesamiento específico para ResNet50
        return x

    def predict(self, img_path):
        """
        Realiza la predicción de enfermedad y especie de planta,
        devolviendo solo la predicción con la probabilidad más alta.

        Args:
            img_path: Ruta a la imagen a predecir

        Returns:
            dict: Diccionario con las predicciones y probabilidades más altas.
        """
        # Preprocesar imagen
        processed_img = self.preprocess_image(img_path)

        # Hacer predicción
        predictions = self.model.predict(processed_img)

        # El modelo devuelve dos salidas: [enfermedad, planta]
        disease_probs, plant_probs = predictions[0][0], predictions[1][0]

        # Obtener el índice de la predicción de enfermedad con la probabilidad más alta
        predicted_disease_index = np.argmax(disease_probs)
        predicted_plant_index = np.argmax(plant_probs)

        # Obtener el nombre y la probabilidad de la predicción de enfermedad
        predicted_disease_name = self.disease_classes[predicted_disease_index]
        plant_name, disease_name = predicted_disease_name.split('_', 1)

        # Obtener el nombre y la probabilidad de la predicción de planta
        predicted_plant_name = self.plant_classes[predicted_plant_index]
        # Traducir a español
        translated_plant = PLANT_TRANSLATIONS.get(predicted_plant_name, predicted_plant_name)
        translated_disease = DISEASE_TRANSLATIONS.get(disease_name, disease_name)

        # Si la planta está sana, ajustar la frase
        if disease_name == 'healthy':
            translated_disease = f"{translated_plant} sano"
        else:
            translated_disease = f"{translated_disease}"

        return translated_disease, float(disease_probs[predicted_disease_index]), translated_plant, float(plant_probs[predicted_plant_index])

def load_predictor(model_path='./full_model.h5',
                    weights_path='./weights.28-0.02.hdf5'):
    """
    Carga el predictor para usar desde notebook.

    Returns:
        Objeto PlantDiseasePredictor configurado
    """
    # Verificar que los archivos existan
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontraron los pesos en {weights_path}")

    return PlantDiseasePredictor(model_path, weights_path)