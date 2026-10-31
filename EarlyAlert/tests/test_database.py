"""
EarlyAlert: Predicting Student Dropout Risk
Module: tests/test_database.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Unit tests for SQLite database layer (src/database.py).
    Verifies table creation, seeding, record insertion, querying, and audit logging.
"""

import os
import sys
import tempfile
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import (
    get_db,
    init_db,
    close_db,
    seed_database,
    save_student_record,
    get_student_by_id,
    log_prediction,
    get_prediction_history,
    get_analytics_summary,
    Student,
    Prediction,
    AuditLog
)


@pytest.fixture
def temp_db_path():
    """Creates a temporary SQLite database for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    close_db(path)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def test_init_and_seed_database(temp_db_path):
    """Verifies database schema initialization and initial seeding."""
    init_db(temp_db_path)
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    count = seed_database(csv_path=csv_path, db_path=temp_db_path, max_records=20)
    assert count == 20

    session = get_db(temp_db_path)
    students_in_db = session.query(Student).count()
    assert students_in_db == 20

    # Ensure seeding logs an audit record
    audit_count = session.query(AuditLog).filter_by(event_type="SEEDING").count()
    assert audit_count >= 1


def test_save_and_retrieve_student(temp_db_path):
    """Verifies that custom student records are inserted and retrieved accurately."""
    init_db(temp_db_path)
    sample_student = {
        "student_id": "STU-TEST-001",
        "Marital Status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 9500,
        "Daytime/evening attendance": 1,
        "Previous qualification": 1,
        "Previous qualification (grade)": 140.0,
        "Nacionality": 1,
        "Mother's qualification": 1,
        "Father's qualification": 1,
        "Mother's occupation": 1,
        "Father's occupation": 1,
        "Admission grade": 135.0,
        "Displaced": 0,
        "Educational special needs": 0,
        "Debtor": 0,
        "Tuition fees up to date": 1,
        "Gender": 0,
        "Scholarship holder": 1,
        "Age at enrollment": 19,
        "International": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 14.5,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 15.0,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10.5,
        "Inflation rate": 1.2,
        "GDP": 1.5,
        "actual_target": "Graduate"
    }

    saved_id = save_student_record(sample_student, db_path=temp_db_path)
    assert saved_id == "STU-TEST-001"

    retrieved = get_student_by_id("STU-TEST-001", db_path=temp_db_path)
    assert retrieved is not None
    assert retrieved["student_id"] == "STU-TEST-001"
    assert retrieved["actual_target"] == "Graduate"
    assert retrieved["Age at enrollment"] == 19
    assert retrieved["Tuition fees up to date"] == 1


def test_log_prediction_and_history(temp_db_path):
    """Verifies that model predictions are logged and retrieved in historical order."""
    init_db(temp_db_path)
    # Seed one student first
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    seed_database(csv_path=csv_path, db_path=temp_db_path, max_records=5)

    prediction_result = {
        "predicted_class": "Dropout",
        "probabilities": {"Dropout": 0.88, "Enrolled": 0.08, "Graduate": 0.04},
        "dropout_risk_tier": "HIGH RISK (Immediate Intervention Required)",
        "key_risk_factors": {"tuition_fees_overdue": True, "debtor_status": True}
    }

    pred_id = log_prediction(
        student_id="STU-00001",
        prediction_result=prediction_result,
        model_version="v1.0-RandomForest",
        db_path=temp_db_path
    )
    assert pred_id > 0

    history = get_prediction_history(limit=10, db_path=temp_db_path)
    assert len(history) == 1
    assert history[0]["student_id"] == "STU-00001"
    assert history[0]["predicted_class"] == "Dropout"
    assert history[0]["probabilities"]["Dropout"] == 0.88


def test_analytics_summary(temp_db_path):
    """Verifies that population analytics calculate correct aggregations."""
    init_db(temp_db_path)
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    seed_database(csv_path=csv_path, db_path=temp_db_path, max_records=30)

    summary = get_analytics_summary(db_path=temp_db_path)
    assert summary["total_students_enrolled"] == 30
    assert "historical_target_distribution" in summary
    assert sum(summary["historical_target_distribution"].values()) == 30
