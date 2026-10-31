"""
EarlyAlert: Predicting Student Dropout Risk
Module: tests/test_api.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Unit and integration tests for FastAPI REST API (src/app.py).
    Verifies endpoints:
    - /api/health
    - /api/analytics
    - /api/students
    - /api/predict
    - /api/predictions
    - /api/metrics
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app import app


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient instance with startup events executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies that the /api/health endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["database"]["connected"] is True
    assert data["model"]["loaded"] is True


def test_analytics_endpoint(client):
    """Verifies /api/analytics returns population metrics and target distributions."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_students_enrolled" in data
    assert "historical_target_distribution" in data
    assert data["total_students_enrolled"] >= 1000


def test_students_list_endpoint(client):
    """Verifies pagination and querying on /api/students."""
    response = client.get("/api/students?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert data["total"] >= 1000
    assert len(data["students"]) == 10
    assert "student_id" in data["students"][0]


def test_predict_endpoint_high_risk(client):
    """Verifies that high-risk student JSON triggers proper inference and audit logging."""
    high_risk_path = os.path.join(PROJECT_ROOT, "data", "sample_students", "high_risk_student.json")
    with open(high_risk_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "Dropout"
    assert data["probabilities"]["Dropout"] > 0.5
    assert "HIGH RISK" in data["dropout_risk_tier"]
    assert "prediction_audit_id" in data


def test_predict_endpoint_invalid_input(client):
    """Verifies that malformed student payload returns 422 Unprocessable Entity."""
    invalid_path = os.path.join(PROJECT_ROOT, "data", "sample_students", "invalid_student.json")
    with open(invalid_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predictions_history_endpoint(client):
    """Verifies that /api/predictions logs and returns past inference calls."""
    response = client.get("/api/predictions?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) >= 1


def test_model_metrics_endpoint(client):
    """Verifies /api/metrics loads holdout comparison and subgroup data."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_comparison" in data
    assert len(data["model_comparison"]) >= 4
