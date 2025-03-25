import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Clases completas basadas en tu dataset
DISEASE_CLASSES = [
    'Apple_Apple_scab',
    'Apple_Black_rot',
    'Apple_Cedar_apple_rust',
    'Apple_healthy',
    'Blueberry_healthy',
    'Cherry_including_sour_healthy',
    'Cherry_including_sour_Powdery_mildew',
    'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_maize_Common_rust_',
    'Corn_maize_healthy',
    'Corn_maize_Northern_Leaf_Blight',
    'Grape_Black_rot',
    'Grape_Esca_Black_Measles',
    'Grape_healthy',
    'Grape_Leaf_blight_Isariopsis_Leaf_Spot',
    'Orange_Haunglongbing_Citrus_greening',
    'Peach_Bacterial_spot',
    'Peach_healthy',
    'Pepper_bell_Bacterial_spot',
    'Pepper_bell_healthy',
    'Potato_Early_blight',
    'Potato_healthy',
    'Potato_Late_blight',
    'Raspberry_healthy',
    'Soybean_healthy',
    'Squash_Powdery_mildew',
    'Strawberry_healthy',
    'Strawberry_Leaf_scorch',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus'
]

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

        return disease_name, float(disease_probs[predicted_disease_index]), predicted_plant_name,float(plant_probs[predicted_plant_index])
        

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