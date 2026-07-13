
"""
FastAPI application.
"""

from fastapi import FastAPI
from .schemas import PredictionRequest, PredictionResponse
from .predictor import predict
from .model_info import load_model_info
import logging
import os
from fastapi import HTTPException

os.makedirs('outputs/logs/', exist_ok=True)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.FileHandler('outputs/logs/api.log'), logging.StreamHandler()]) 

logger = logging.getLogger(__name__)

app = FastAPI( title = 'Phytoplankton Prediction API', description = """ Machine learning API for predicting phytoplankton primary production in the Salish Sea ecosystem. """,
    version = '1.0.0', contact={'name': 'Ilias Bougoudis'})

@app.get('/') # The primary entry point.
def root():

        return {'name': 'Phytoplankton Prediction API', 'version': '1.0.0', 'docs': '/docs', 'redoc': '/redoc'}

@app.get('/health') # Monitoring service availability.
def health():

    return {'status': 'healthy', 'model_loaded': True}

@app.post('/predict', response_model = PredictionResponse) # Where predictions are produced.
def predict_endpoint(request: PredictionRequest): # type: ignore

    logger.info(f'Prediction request received ' f'({len(request.model_dump())} features)')

    try:
        predictions = predict(request.model_dump())

        logger.info('Prediction completed successfully')

        return {'predictions': predictions}

    except ValueError as e:

        logger.exception('Prediction failed')

        raise HTTPException(status_code=400, detail=str(e))

    except Exception:

        logger.exception('Unexpected server error')
        raise HTTPException(status_code=500, detail='Internal server error.')

@app.get('/model-info') # The model info.
def model_info():

    return load_model_info()
