import json
import os
import time

import numpy as np
import redis
import settings

from predict import load_predictor

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

model_path='./full_model.h5'
weights_path = "./weights.28-0.02.hdf5"
model_type = 'resnet'

#model_type = 'EfficentNet'
#model_path=''
#weights_path = "./weights.26-0.02.hdf5"


# Cargar el predictor
predictor = load_predictor(model_type = model_type, model_path = model_path, weights_path = weights_path)

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
    print('definitiva ',img_path)
    predicted_class_enfermedad,predicted_prob_enfermedad,predicted_class_especie,predicted_prob_especie = predictor.predict(img_path)

    return predicted_class_enfermedad, predicted_prob_enfermedad,predicted_class_especie,predicted_prob_especie


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

            predicted_class_enfermedad, predicted_prob_enfermedad, predicted_class_especie, predicted_prob_especie = predict(image_name)

            # 3.  Prepare a new JSON with the results
            output = {
                'predicted_class_enfermedad': predicted_class_enfermedad,
                'predicted_prob_enfermedad':  predicted_prob_enfermedad,
                'predicted_class_especie':    predicted_class_especie,
                'predicted_prob_especie':     predicted_prob_especie
            }

            # 4.  Store the job results on Redis using the original
            # job ID as the key
            db.set(job_id, json.dumps(output))
            print(f"Processed job {job_id} with prediction {predicted_class_enfermedad} and score {predicted_prob_enfermedad}; {predicted_class_especie} and score {predicted_prob_especie}")
        else:
            print("No jobs found, retrying...")

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
