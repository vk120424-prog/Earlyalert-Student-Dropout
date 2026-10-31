"""
Tests for EarlyAlert Inference CLI & Input Validation (FR-06)
Author: Vasanthkumar A | Knowledge Institute of Technology
"""

import os
import sys
import json
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_student, validate_student_record, ValidationError
from src.validate import ALL_FEATURE_COLUMNS


@pytest.fixture
def high_risk_json_path():
    return os.path.join(PROJECT_ROOT, "data", "sample_students", "high_risk_student.json")


@pytest.fixture
def low_risk_json_path():
    return os.path.join(PROJECT_ROOT, "data", "sample_students", "low_risk_student.json")


@pytest.fixture
def invalid_json_path():
    return os.path.join(PROJECT_ROOT, "data", "sample_students", "invalid_student.json")


def test_predict_valid_high_risk_record(high_risk_json_path):
    """Verifies that high-risk student input is correctly predicted as Dropout with high risk tier."""
    res = predict_student(input_data=high_risk_json_path)
    assert res["status"] == "SUCCESS"
    assert res["predicted_class"] == "Dropout"
    assert "HIGH RISK" in res["dropout_risk_tier"]
    assert res["probabilities"]["Dropout"] > 0.50
    assert res["key_risk_factors"]["tuition_fees_overdue"] is True


def test_predict_valid_low_risk_record(low_risk_json_path):
    """Verifies that low-risk student input is correctly predicted as Graduate with low risk tier."""
    res = predict_student(input_data=low_risk_json_path)
    assert res["status"] == "SUCCESS"
    assert res["predicted_class"] == "Graduate"
    assert "LOW RISK" in res["dropout_risk_tier"]
    assert res["probabilities"]["Graduate"] > 0.50


def test_predict_rejects_missing_field(high_risk_json_path):
    """Verifies that missing any of the 36 fields raises ValidationError."""
    with open(high_risk_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    del data["Admission grade"]  # Remove a required field

    with pytest.raises(ValidationError) as exc_info:
        predict_student(input_data=data)
    assert "Missing required field: 'Admission grade'" in str(exc_info.value)


def test_predict_rejects_out_of_range_field(low_risk_json_path):
    """Verifies that out-of-range field values raise ValidationError."""
    with open(low_risk_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["Age at enrollment"] = -1  # Impossible negative age

    with pytest.raises(ValidationError) as exc_info:
        predict_student(input_data=data)
    assert "out of allowable range" in str(exc_info.value)


def test_predict_rejects_invalid_json_file(invalid_json_path):
    """Verifies that predict_student fails when pointed at an invalid JSON file."""
    with pytest.raises(ValidationError):
        predict_student(input_data=invalid_json_path)


def test_predict_rejects_nonexistent_file():
    """Verifies that nonexistent input file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        predict_student(input_data="data/sample_students/does_not_exist.json")
