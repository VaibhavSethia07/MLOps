from unittest.mock import MagicMock

import pytest
from app import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(app)


def test_home_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'summary': app.summary}


def test_get_prediction_endpoint(client, monkeypatch):
    # Mocking predictor.predict method
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [
        {"label": "unacceptable", "score": 0.25},
        {"label": "acceptable", "score": 0.75}
    ]
    monkeypatch.setattr('app.predictor', mock_predictor)

    # Testing with valid input
    response = client.get('/predict?text=This is a test sentence.')
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["label"] == "unacceptable"
    assert data[1]["label"] == "acceptable"
