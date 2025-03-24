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
    
    def predict(self, img_path, top_n=3):
        """
        Realiza la predicción de enfermedad y especie de planta.
        
        Args:
            img_path: Ruta a la imagen a predecir
            top_n: Número de predicciones principales a devolver
            
        Returns:
            dict: Diccionario con las predicciones y probabilidades
        """
        # Preprocesar imagen
        processed_img = self.preprocess_image(img_path)
        
        # Hacer predicción
        predictions = self.model.predict(processed_img)
        
        # El modelo devuelve dos salidas: [enfermedad, planta]
        disease_probs, plant_probs = predictions[0][0], predictions[1][0]
        
        # Obtener los índices de las top_n predicciones
        top_disease_indices = np.argsort(disease_probs)[-top_n:][::-1]
        top_plant_indices = np.argsort(plant_probs)[-top_n:][::-1]
        
        # Preparar resultados para enfermedades
        disease_predictions = []
        for idx in top_disease_indices:
            disease_name = self.disease_classes[idx]
            plant, disease = disease_name.split('_', 1)
            disease_predictions.append({
                'disease': disease,
                'plant': plant,
                'full_name': disease_name,
                'confidence': float(disease_probs[idx])
            })
        
        # Preparar resultados para plantas
        plant_predictions = []
        for idx in top_plant_indices:
            plant_predictions.append({
                'plant': self.plant_classes[idx],
                'confidence': float(plant_probs[idx])
            })
        
        return {
            'top_diseases': disease_predictions,
            'top_plants': plant_predictions,
            'all_diseases': {self.disease_classes[i]: float(p) for i, p in enumerate(disease_probs)},
            'all_plants': {plant: float(plant_probs[i]) for i, plant in enumerate(self.plant_classes)}
        }

def load_predictor(model_path='./runs/run2_res_net_50V2/full_model.h5',
                  weights_path='./runs/run2_res_net_50V2/weights.28-0.02.hdf5'):
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