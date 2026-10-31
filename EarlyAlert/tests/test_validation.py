"""
Tests for EarlyAlert Schema & Data Validation (FR-01, FR-02)
Author: Vasanthkumar A | Knowledge Institute of Technology
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.validate import validate_schema, ValidationError, ALL_FEATURE_COLUMNS, LOG_FILE


@pytest.fixture
def sample_valid_df():
    """Loads a small valid slice of the raw dataset."""
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    df = pd.read_csv(raw_path)
    return df.head(50).copy()


def test_valid_schema_passes(sample_valid_df):
    """Verifies that an authentic, compliant dataset passes validation cleanly."""
    assert validate_schema(sample_valid_df, is_inference=False) is True


def test_schema_fails_on_missing_column(sample_valid_df):
    """FR-01: Verifies that dropping an expected column raises ValidationError."""
    corrupted_df = sample_valid_df.drop(columns=["Course"])
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "Missing required column(s)" in str(exc_info.value)
    assert "Course" in str(exc_info.value)


def test_schema_fails_on_missing_target(sample_valid_df):
    """Verifies that missing Target column fails training validation."""
    corrupted_df = sample_valid_df.drop(columns=["Target"])
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "Target" in str(exc_info.value)


def test_schema_fails_on_bad_dtype(sample_valid_df):
    """FR-01: Verifies that non-numeric dtypes in numeric columns raise ValidationError."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df["Age at enrollment"] = "invalid_string_age"
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "non-numeric dtype" in str(exc_info.value)


def test_schema_fails_on_out_of_bound_values(sample_valid_df):
    """FR-01: Verifies that values exceeding allowable bounds raise ValidationError."""
    corrupted_df = sample_valid_df.copy()
    # Age cannot be negative or absurdly high
    corrupted_df.loc[0, "Age at enrollment"] = 150
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "out of bounds" in str(exc_info.value)


def test_schema_fails_on_invalid_target_class(sample_valid_df):
    """Verifies that unauthorized target categories raise ValidationError."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, "Target"] = "Expelled"  # Unauthorized class
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "invalid classes" in str(exc_info.value)


def test_schema_fails_on_binary_bound_violation(sample_valid_df):
    """Verifies that binary indicator fields reject values other than 0 or 1."""
    corrupted_df = sample_valid_df.copy()
    corrupted_df.loc[0, "Tuition fees up to date"] = 5  # Outside {0, 1}
    with pytest.raises(ValidationError) as exc_info:
        validate_schema(corrupted_df, is_inference=False)
    assert "outside {0, 1}" in str(exc_info.value) or "out of bounds" in str(exc_info.value)


def test_validation_errors_are_logged_to_run_log(sample_valid_df):
    """FR-01: Verifies that schema validation failures are logged to logs/run.log, never silently ignored."""
    corrupted_df = sample_valid_df.drop(columns=["Displaced"])
    log_size_before = os.path.getsize(LOG_FILE) if os.path.exists(LOG_FILE) else 0

    with pytest.raises(ValidationError):
        validate_schema(corrupted_df, is_inference=False)

    log_size_after = os.path.getsize(LOG_FILE)
    assert log_size_after > log_size_before, "Validation failure was not logged to logs/run.log!"

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_content = f.read()
    assert "Missing required column(s)" in log_content
    assert "Displaced" in log_content
