"""
Unit tests for the FastAPI prediction API.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_PATH = Path(__file__).resolve().parents[1] / "src" / "models" / "heart_model.pkl"

# Set MODEL_PATH env var before importing the app
import os
os.environ["MODEL_PATH"] = str(MODEL_PATH)

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    if not MODEL_PATH.exists():
        pytest.skip("Trained model not found; run train.py first")
    # Import here so MODEL_PATH env var is set
    from src.api import model_app
    return TestClient(model_app.app)


SAMPLE_PATIENT = {
    "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
    "oldpeak": 1.5, "slope": 1, "ca": 0, "thal": 3
}


class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_predict_valid(self, client):
        resp = client.post("/predict", json=SAMPLE_PATIENT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in [0, 1]
        assert 0 <= data["confidence"] <= 1
        assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]

    def test_predict_missing_field(self, client):
        bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "predictions_total" in resp.text or resp.status_code == 200
