import json
import time
from uuid import uuid4

import redis

from .. import settings

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
try:
    db = redis.StrictRedis(
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


async def model_predict(image_name):
    print(f"Processing image {image_name}...")
    """
    Receives an image name and queues the job into Redis.
    Will loop until getting the answer from our ML service.

    Parameters
    ----------
    image_name : str
        Name for the image uploaded by the user.

    Returns
    -------
    predicted_class_enfermedad, predicted_prob_enfermedad : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        predicted_prob_enfermedad as a number.
    """
    predicted_class_enfermedad = None
    predicted_prob_enfermedad = None
    predicted_class_especie = None
    predicted_prob_especie = None

    # Assign an unique ID for this job and add it to the queue.
    # We need to assing this ID because we must be able to keep track
    # of this particular job across all the services
    #1.- Add an unique ID
    job_id = str(uuid4())

    #2.- Dictionary
    job_data = {
        "id": job_id,
        "image_name": image_name
    }

    # Send the job to the model service using Redis
    # Hint: Using Redis `lpush()` function should be enough to accomplish this.

    #3.- Send the job to the model
    db.lpush(settings.REDIS_QUEUE, json.dumps(job_data))

    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predicted_class_enfermedads using job_id
        # Hint: Investigate how can we get a value using a key from Redis

        output = db.get(job_id)  # Obtener el valor desde Redis usando el ID del trabajo


        # Check if the text was correctly processed by our ML model
        # Don't modify the code below, it should work as expected
        if output is not None:
            output = json.loads(output.decode("utf-8"))
            predicted_class_enfermedad = output["predicted_class_enfermedad"]
            predicted_prob_enfermedad = output["predicted_prob_enfermedad"]
            predicted_class_especie = output["predicted_class_especie"]
            predicted_prob_especie = output["predicted_prob_especie"]

            db.delete(job_id)     # Eliminar el trabajo de Redis una vez procesado
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    return predicted_class_enfermedad, predicted_prob_enfermedad, predicted_class_especie,predicted_prob_especie
