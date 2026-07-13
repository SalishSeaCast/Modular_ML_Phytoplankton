
"""
Pydantic schemas for API requests/responses - Defyning what JSON the API accepts and validating inputs - output.
"""

from pydantic import create_model, BaseModel
from .model_info import load_model_info

outputs = load_model_info()

inputs = (outputs["drivers"]  + outputs["spatial"] + outputs["day_input"])

PredictionRequest = create_model('PredictionRequest', **{feature: (float, ...) for feature in inputs}) # pyright: ignore[reportCallIssue]        

class PredictionResponse(BaseModel):

    predictions: list[float]