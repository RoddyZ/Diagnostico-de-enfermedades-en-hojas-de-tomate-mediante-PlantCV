from pydantic import BaseModel


class PredictRequest(BaseModel):
    file: str


class PredictResponse(BaseModel):
    success: bool
    predicted_class_enfermedad: str
    predicted_prob_enfermedad: float
    predicted_class_especie: str
    predicted_prob_especie: float
    image_file_name: str
