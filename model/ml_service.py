import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

from predict import predict_image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
try:
    db = redis.Redis(
        host=settings.REDIS_IP,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB_ID
    )
    # Check the connection
    db.ping()
    print("Connected to Redis successfully.")
except redis.ConnectionError as e:
    print("Redis connection failed:", e)
    db = None


# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")
# Parámetros fijos
WEIGHTS_PATH = "./base_line.pth"  # Ruta de los pesos del modelo

# Lista de clases actualizada
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf_Mold",
    "Powdery_mildew",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus"
]

def predict(image_name):
    """
    Carga una imagen desde la carpeta de subida y predice su clase con el modelo entrenado.

    Parámetros:
    ----------
    image_name : str
        Nombre del archivo de imagen.

    Retorna:
    -------
    class_name, pred_probability : tuple(str, float)
        Clase predicha y su confianza (0 a 1).
    """
    # Construir la ruta completa de la imagen
    img_path = os.path.join(settings.UPLOAD_FOLDER, image_name)

    # Llamar a predict_image con los parámetros predefinidos
    class_name, pred_probability = predict_image(
        image_path=img_path,
        weights_path=WEIGHTS_PATH,
        class_names=CLASS_NAMES
    )

    return class_name, round(pred_probability, 4)


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions brpop() and set().
        # TODO
        # 1.  Take a new job from Redis
        job_data = db.brpop(settings.REDIS_QUEUE,timeout=settings.SERVER_SLEEP)

        if job_data is not None:


            # Decode the JSON data for the given job
            _, job_info = job_data
            job_info = json.loads(job_info)

            # Important! Get and keep the original job ID
            job_id = job_info["id"]
            image_name = job_info["image_name"]

            # Get the file name without directories
            image_name = image_name.split("/")[-1]
            # #2. Run the loaded ml model (use the predict() function)
            class_name, pred_probability = predict(image_name)

            # 3.  Prepare a new JSON with the results
            output = {
                "prediction": class_name,
                "score": pred_probability
            }

            # 4.  Store the job results on Redis using the original
            # job ID as the key
            db.set(job_id, json.dumps(output))
            print(f"Processed job {job_id} with prediction {class_name} and score {pred_probability}")
        else:
            print("No jobs found, retrying...")

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
