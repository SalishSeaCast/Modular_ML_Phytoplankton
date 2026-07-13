
"""
Prediction helper functions - converting the JSON to numpy and calling sklearn.
"""

from .model_info import load_model_info
from .model_loader import model

if model is None:

    raise RuntimeError(
        "No trained model available."
    )

else:
    info = load_model_info()

def predict(features):

    expected = (info['drivers'] + info['spatial'] + info['day_input'])

    X = [[features[name] for name in expected]]

    predictions = model.predict(X)

    return predictions.tolist() # for proper FastAPI serialization 