
"""
Testing for the service availability of the API. 
"""

from fastapi.testclient import TestClient
from src.api.main import app
from src.api.model_info import load_model_info

client = TestClient(app)

def test_health():

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["name"] == "Phytoplankton Prediction API"
    assert body["version"] == "1.0.0"

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert "drivers" in body
    assert "spatial" in body
    assert "day_input" in body
    assert "n_features" in body

def test_predict():

    info = load_model_info()
    expected = (info['drivers'] + info['spatial'] + info['day_input'])

    payload = {feature: 0.0 for feature in expected}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert isinstance(body["predictions"][0], float)

def test_predict_missing_feature():

    info = load_model_info()

    expected = (
        info["drivers"]
        + info["spatial"]
        + info["day_input"]
    )

    payload = {feature: 0.0 for feature in expected[:-1]} # intentionally skipping the last feature

    response = client.post("/predict", json=payload)

    assert response.status_code == 422