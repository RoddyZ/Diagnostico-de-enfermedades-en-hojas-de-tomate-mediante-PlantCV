import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Clases de plantas y enfermedades (actualiza según tus clases reales)
PLANT_CLASSES = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 
                 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 
                 'Squash', 'Strawberry', 'Tomato']

# Mapeo de índices a nombres de enfermedades (deberías completar esto con tus 38 clases)
DISEASE_CLASSES = {
    0:  'Apple_Apple_scab',
    1:  'Apple_Black_rot',
    2:  'Apple_Cedar_apple_rust',
    3:  'Apple_healthy',
    4:  'Blueberry_healthy',
    5:  'Cherry_including_sour_healthy',
    6:  'Cherry_including_sour_Powdery_mildew',
    7:  'Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot',
    8:  'Corn_maize_Common_rust_',
    9:  'Corn_maize_healthy',
    10: 'Corn_maize_Northern_Leaf_Blight',
    11: 'Grape_Black_rot',
    12: 'Grape_Esca_Black_Measles',
    13: 'Grape_healthy',
    14: 'Grape_Leaf_blight_Isariopsis_Leaf_Spot',
    15: 'Orange_Haunglongbing_Citrus_greening',
    16: 'Peach_Bacterial_spot',
    17: 'Peach_healthy',
    18: 'Pepper_bell_Bacterial_spot',
    19: 'Pepper_bell_healthy',
    20: 'Potato_Early_blight',
    21: 'Potato_healthy',
    22: 'Potato_Late_blight',
    23: 'Raspberry_healthy',
    24: 'Soybean_healthy',
    25: 'Squash_Powdery_mildew',
    26: 'Strawberry_healthy',
    27: 'Strawberry_Leaf_scorch',
    28: 'Tomato_Bacterial_spot',
    29: 'Tomato_Early_blight',
    30: 'Tomato_healthy',
    31: 'Tomato_Late_blight',
    32: 'Tomato_Leaf_Mold',
    33: 'Tomato_Septoria_leaf_spot',
    34: 'Tomato_Spider_mites_Two-spotted_spider_mite',
    35: 'Tomato_Target_Spot',
    36: 'Tomato_Tomato_mosaic_virus',
    37: 'Tomato_Tomato_Yellow_Leaf_Curl_Virus'
}

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
        self.img_size = (256, 256)  # Ajusta según el tamaño usado en tu entrenamiento
    
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
        x = preprocess_input(x)
        return x
    
    def predict(self, img_path):
        """
        Realiza la predicción de enfermedad y especie de planta.
        
        Args:
            img_path: Ruta a la imagen a predecir
            
        Returns:
            dict: Diccionario con las predicciones y probabilidades
        """
        # Preprocesar imagen
        processed_img = self.preprocess_image(img_path)
        
        # Hacer predicción
        predictions = self.model.predict(processed_img)
        
        # El modelo debería devolver dos salidas: [enfermedad, planta]
        disease_probs, plant_probs = predictions[0][0], predictions[1][0]
        
        # Obtener índices de las clases con mayor probabilidad
        disease_idx = np.argmax(disease_probs)
        plant_idx = np.argmax(plant_probs)
        
        # Obtener nombres de las clases
        disease_name = DISEASE_CLASSES.get(disease_idx, "Unknown")
        plant_name = PLANT_CLASSES[plant_idx]
        
        # Extraer solo el nombre de la enfermedad (removiendo el de la planta)
        disease_only = disease_name.split('_', 1)[1] if '_' in disease_name else disease_name
        
        return {
            'plant': {
                'class': plant_name,
                'confidence': float(plant_probs[plant_idx])
            },
            'disease': {
                'class': disease_only,
                'confidence': float(disease_probs[disease_idx]),
                'full_class': disease_name
            },
            'all_disease_probs': {DISEASE_CLASSES[i]: float(p) for i, p in enumerate(disease_probs)},
            'all_plant_probs': {PLANT_CLASSES[i]: float(p) for i, p in enumerate(plant_probs)}
        }

# Función para cargar desde notebook
def load_predictor(model_path='./runs/run2_res_net_50V2/full_model.h5',
                  weights_path='./runs/run2_res_net_50V2/weights.28-0.02.hdf5'):
    """
    Carga el predictor para usar desde notebook.
    
    Returns:
        Objeto PlantDiseasePredictor
    """
    return PlantDiseasePredictor(model_path, weights_path)