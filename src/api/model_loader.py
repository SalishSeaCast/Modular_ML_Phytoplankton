
"""
Load the model.
"""

import joblib
from pathlib import Path

MODEL_PATH = (Path(__file__).parents[2] / 'outputs'/'model'/'model_diat_pr.joblib')

if MODEL_PATH.exists():

    model = joblib.load(MODEL_PATH)

else:

    model = None
