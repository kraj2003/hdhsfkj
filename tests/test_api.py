# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"hours_back": 2})
    assert response.status_code in [200, 503]  # 503 if model not loaded