"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/database.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Database access layer using SQLite and SQLAlchemy:
    - Defines schemas for 'students', 'predictions', and 'audit_logs'.
    - Handles database initialization, seeding from raw CSV data,
      saving new student records, and recording model inference audits.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    ForeignKey,
    desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

from src.validate import ALL_FEATURE_COLUMNS, CATEGORICAL_CODE_COLUMNS, CONTINUOUS_NUMERIC_COLUMNS

logger = logging.getLogger("database")

DB_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DB_DIR, exist_ok=True)
DEFAULT_DB_PATH = os.path.join(DB_DIR, "earlyalert.db")

Base = declarative_base()


class Student(Base):
    """Stores student demographic, socioeconomic, and curricular unit records."""
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Actual outcome if known (Dropout, Enrolled, Graduate, or None for prospective)
    actual_target = Column(String(32), nullable=True)

    # Core Features (stored in standardized snake_case and mapped from raw CSV)
    marital_status = Column(Integer, nullable=False)
    application_mode = Column(Integer, nullable=False)
    application_order = Column(Integer, nullable=False)
    course = Column(Integer, nullable=False)
    daytime_evening_attendance = Column(Integer, nullable=False)
    previous_qualification = Column(Integer, nullable=False)
    previous_qualification_grade = Column(Float, nullable=False)
    nacionality = Column(Integer, nullable=False)
    mother_qualification = Column(Integer, nullable=False)
    father_qualification = Column(Integer, nullable=False)
    mother_occupation = Column(Integer, nullable=False)
    father_occupation = Column(Integer, nullable=False)
    admission_grade = Column(Float, nullable=False)
    displaced = Column(Integer, nullable=False)
    educational_special_needs = Column(Integer, nullable=False)
    debtor = Column(Integer, nullable=False)
    tuition_fees_up_to_date = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    scholarship_holder = Column(Integer, nullable=False)
    age_at_enrollment = Column(Integer, nullable=False)
    international = Column(Integer, nullable=False)

    # Curricular Units 1st Semester
    curricular_units_1st_sem_credited = Column(Integer, nullable=False)
    curricular_units_1st_sem_enrolled = Column(Integer, nullable=False)
    curricular_units_1st_sem_evaluations = Column(Integer, nullable=False)
    curricular_units_1st_sem_approved = Column(Integer, nullable=False)
    curricular_units_1st_sem_grade = Column(Float, nullable=False)
    curricular_units_1st_sem_without_evaluations = Column(Integer, nullable=False)

    # Curricular Units 2nd Semester
    curricular_units_2nd_sem_credited = Column(Integer, nullable=False)
    curricular_units_2nd_sem_enrolled = Column(Integer, nullable=False)
    curricular_units_2nd_sem_evaluations = Column(Integer, nullable=False)
    curricular_units_2nd_sem_approved = Column(Integer, nullable=False)
    curricular_units_2nd_sem_grade = Column(Float, nullable=False)
    curricular_units_2nd_sem_without_evaluations = Column(Integer, nullable=False)

    # Macroeconomic Indicators
    unemployment_rate = Column(Float, nullable=False)
    inflation_rate = Column(Float, nullable=False)
    gdp = Column(Float, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert database entity to dictionary with original dataset column names."""
        return {
            "student_id": self.student_id,
            "actual_target": self.actual_target,
            "Marital Status": self.marital_status,
            "Application mode": self.application_mode,
            "Application order": self.application_order,
            "Course": self.course,
            "Daytime/evening attendance": self.daytime_evening_attendance,
            "Previous qualification": self.previous_qualification,
            "Previous qualification (grade)": self.previous_qualification_grade,
            "Nacionality": self.nacionality,
            "Mother's qualification": self.mother_qualification,
            "Father's qualification": self.father_qualification,
            "Mother's occupation": self.mother_occupation,
            "Father's occupation": self.father_occupation,
            "Admission grade": self.admission_grade,
            "Displaced": self.displaced,
            "Educational special needs": self.educational_special_needs,
            "Debtor": self.debtor,
            "Tuition fees up to date": self.tuition_fees_up_to_date,
            "Gender": self.gender,
            "Scholarship holder": self.scholarship_holder,
            "Age at enrollment": self.age_at_enrollment,
            "International": self.international,
            "Curricular units 1st sem (credited)": self.curricular_units_1st_sem_credited,
            "Curricular units 1st sem (enrolled)": self.curricular_units_1st_sem_enrolled,
            "Curricular units 1st sem (evaluations)": self.curricular_units_1st_sem_evaluations,
            "Curricular units 1st sem (approved)": self.curricular_units_1st_sem_approved,
            "Curricular units 1st sem (grade)": self.curricular_units_1st_sem_grade,
            "Curricular units 1st sem (without evaluations)": self.curricular_units_1st_sem_without_evaluations,
            "Curricular units 2nd sem (credited)": self.curricular_units_2nd_sem_credited,
            "Curricular units 2nd sem (enrolled)": self.curricular_units_2nd_sem_enrolled,
            "Curricular units 2nd sem (evaluations)": self.curricular_units_2nd_sem_evaluations,
            "Curricular units 2nd sem (approved)": self.curricular_units_2nd_sem_approved,
            "Curricular units 2nd sem (grade)": self.curricular_units_2nd_sem_grade,
            "Curricular units 2nd sem (without evaluations)": self.curricular_units_2nd_sem_without_evaluations,
            "Unemployment rate": self.unemployment_rate,
            "Inflation rate": self.inflation_rate,
            "GDP": self.gdp,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Prediction(Base):
    """Audit table logging inference predictions and associated risk diagnostics."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(64), ForeignKey("students.student_id"), index=True, nullable=False)
    predicted_class = Column(String(32), nullable=False)
    dropout_probability = Column(Float, nullable=False)
    enrolled_probability = Column(Float, nullable=False)
    graduate_probability = Column(Float, nullable=False)
    dropout_risk_tier = Column(String(64), nullable=False)
    key_risk_factors = Column(Text, nullable=True)  # JSON formatted
    model_version = Column(String(32), default="v1.0-RandomForest")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "predicted_class": self.predicted_class,
            "probabilities": {
                "Dropout": self.dropout_probability,
                "Enrolled": self.enrolled_probability,
                "Graduate": self.graduate_probability
            },
            "dropout_risk_tier": self.dropout_risk_tier,
            "key_risk_factors": json.loads(self.key_risk_factors) if self.key_risk_factors else {},
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AuditLog(Base):
    """System event and validation failure audit trail."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False)  # INGESTION, VALIDATION_FAILURE, INFERENCE, SEEDING
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# Database session and engine registry
_engines: Dict[str, Any] = {}
_session_factories: Dict[str, Any] = {}


def get_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes and returns a thread-safe database session for the specified db_path."""
    global _engines, _session_factories
    abs_path = os.path.abspath(db_path)
    if abs_path not in _engines:
        db_url = f"sqlite:///{abs_path}"
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
        Base.metadata.create_all(engine)
        _engines[abs_path] = engine
        _session_factories[abs_path] = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return _session_factories[abs_path]()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Creates database schema and tables."""
    get_db(db_path)
    logger.info("Database schema initialized at %s", db_path)


def close_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Closes all scoped sessions and disposes engine connection pool."""
    global _engines, _session_factories
    abs_path = os.path.abspath(db_path)
    if abs_path in _session_factories:
        _session_factories[abs_path].remove()
        del _session_factories[abs_path]
    if abs_path in _engines:
        _engines[abs_path].dispose()
        del _engines[abs_path]


def seed_database(
    csv_path: str = "data/raw/student_dropout.csv",
    db_path: str = DEFAULT_DB_PATH,
    max_records: int = 1000
) -> int:
    """
    Populates SQLite database with historical student records from the raw dataset.
    """
    session = get_db(db_path)
    existing_count = session.query(Student).count()
    if existing_count > 0:
        logger.info("Database already contains %d student records. Skipping seeding.", existing_count)
        return existing_count

    if not os.path.exists(csv_path):
        logger.warning("Dataset CSV not found at %s. Cannot seed database.", csv_path)
        return 0

    logger.info("Seeding database with up to %d records from %s...", max_records, csv_path)
    df = pd.read_csv(csv_path).head(max_records)

    records = []
    for idx, row in df.iterrows():
        stu_id = f"STU-{idx+1:05d}"
        student = Student(
            student_id=stu_id,
            actual_target=str(row["Target"]),
            marital_status=int(row["Marital Status"]),
            application_mode=int(row["Application mode"]),
            application_order=int(row["Application order"]),
            course=int(row["Course"]),
            daytime_evening_attendance=int(row["Daytime/evening attendance"]),
            previous_qualification=int(row["Previous qualification"]),
            previous_qualification_grade=float(row["Previous qualification (grade)"]),
            nacionality=int(row["Nacionality"]),
            mother_qualification=int(row["Mother's qualification"]),
            father_qualification=int(row["Father's qualification"]),
            mother_occupation=int(row["Mother's occupation"]),
            father_occupation=int(row["Father's occupation"]),
            admission_grade=float(row["Admission grade"]),
            displaced=int(row["Displaced"]),
            educational_special_needs=int(row["Educational special needs"]),
            debtor=int(row["Debtor"]),
            tuition_fees_up_to_date=int(row["Tuition fees up to date"]),
            gender=int(row["Gender"]),
            scholarship_holder=int(row["Scholarship holder"]),
            age_at_enrollment=int(row["Age at enrollment"]),
            international=int(row["International"]),
            curricular_units_1st_sem_credited=int(row["Curricular units 1st sem (credited)"]),
            curricular_units_1st_sem_enrolled=int(row["Curricular units 1st sem (enrolled)"]),
            curricular_units_1st_sem_evaluations=int(row["Curricular units 1st sem (evaluations)"]),
            curricular_units_1st_sem_approved=int(row["Curricular units 1st sem (approved)"]),
            curricular_units_1st_sem_grade=float(row["Curricular units 1st sem (grade)"]),
            curricular_units_1st_sem_without_evaluations=int(row["Curricular units 1st sem (without evaluations)"]),
            curricular_units_2nd_sem_credited=int(row["Curricular units 2nd sem (credited)"]),
            curricular_units_2nd_sem_enrolled=int(row["Curricular units 2nd sem (enrolled)"]),
            curricular_units_2nd_sem_evaluations=int(row["Curricular units 2nd sem (evaluations)"]),
            curricular_units_2nd_sem_approved=int(row["Curricular units 2nd sem (approved)"]),
            curricular_units_2nd_sem_grade=float(row["Curricular units 2nd sem (grade)"]),
            curricular_units_2nd_sem_without_evaluations=int(row["Curricular units 2nd sem (without evaluations)"]),
            unemployment_rate=float(row["Unemployment rate"]),
            inflation_rate=float(row["Inflation rate"]),
            gdp=float(row["GDP"])
        )
        records.append(student)

    session.bulk_save_objects(records)
    session.commit()

    # Log audit event
    audit = AuditLog(
        event_type="SEEDING",
        message=f"Seeded {len(records)} historical student records from {csv_path}."
    )
    session.add(audit)
    session.commit()
    logger.info("Successfully seeded %d students into %s.", len(records), db_path)
    return len(records)


def save_student_record(data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> str:
    """Saves or updates a student profile in the database."""
    session = get_db(db_path)
    stu_id = data.get("student_id")
    if not stu_id:
        count = session.query(Student).count()
        stu_id = f"STU-NEW-{count+1:04d}"

    existing = session.query(Student).filter_by(student_id=stu_id).first()
    if existing is None:
        existing = Student(student_id=stu_id)
        session.add(existing)

    existing.marital_status = int(data["Marital Status"])
    existing.application_mode = int(data["Application mode"])
    existing.application_order = int(data["Application order"])
    existing.course = int(data["Course"])
    existing.daytime_evening_attendance = int(data["Daytime/evening attendance"])
    existing.previous_qualification = int(data["Previous qualification"])
    existing.previous_qualification_grade = float(data["Previous qualification (grade)"])
    existing.nacionality = int(data["Nacionality"])
    existing.mother_qualification = int(data["Mother's qualification"])
    existing.father_qualification = int(data["Father's qualification"])
    existing.mother_occupation = int(data["Mother's occupation"])
    existing.father_occupation = int(data["Father's occupation"])
    existing.admission_grade = float(data["Admission grade"])
    existing.displaced = int(data["Displaced"])
    existing.educational_special_needs = int(data["Educational special needs"])
    existing.debtor = int(data["Debtor"])
    existing.tuition_fees_up_to_date = int(data["Tuition fees up to date"])
    existing.gender = int(data["Gender"])
    existing.scholarship_holder = int(data["Scholarship holder"])
    existing.age_at_enrollment = int(data["Age at enrollment"])
    existing.international = int(data["International"])
    existing.curricular_units_1st_sem_credited = int(data["Curricular units 1st sem (credited)"])
    existing.curricular_units_1st_sem_enrolled = int(data["Curricular units 1st sem (enrolled)"])
    existing.curricular_units_1st_sem_evaluations = int(data["Curricular units 1st sem (evaluations)"])
    existing.curricular_units_1st_sem_approved = int(data["Curricular units 1st sem (approved)"])
    existing.curricular_units_1st_sem_grade = float(data["Curricular units 1st sem (grade)"])
    existing.curricular_units_1st_sem_without_evaluations = int(data["Curricular units 1st sem (without evaluations)"])
    existing.curricular_units_2nd_sem_credited = int(data["Curricular units 2nd sem (credited)"])
    existing.curricular_units_2nd_sem_enrolled = int(data["Curricular units 2nd sem (enrolled)"])
    existing.curricular_units_2nd_sem_evaluations = int(data["Curricular units 2nd sem (evaluations)"])
    existing.curricular_units_2nd_sem_approved = int(data["Curricular units 2nd sem (approved)"])
    existing.curricular_units_2nd_sem_grade = float(data["Curricular units 2nd sem (grade)"])
    existing.curricular_units_2nd_sem_without_evaluations = int(data["Curricular units 2nd sem (without evaluations)"])
    existing.unemployment_rate = float(data["Unemployment rate"])
    existing.inflation_rate = float(data["Inflation rate"])
    existing.gdp = float(data["GDP"])

    if "actual_target" in data:
        existing.actual_target = data["actual_target"]
    elif "Target" in data:
        existing.actual_target = data["Target"]

    session.commit()
    return stu_id


def log_prediction(
    student_id: str,
    prediction_result: Dict[str, Any],
    model_version: str = "v1.0-RandomForest",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Persists model prediction record and probabilities into the database."""
    session = get_db(db_path)
    probs = prediction_result.get("probabilities", {})
    record = Prediction(
        student_id=student_id,
        predicted_class=prediction_result.get("predicted_class", "Unknown"),
        dropout_probability=float(probs.get("Dropout", 0.0)),
        enrolled_probability=float(probs.get("Enrolled", 0.0)),
        graduate_probability=float(probs.get("Graduate", 0.0)),
        dropout_risk_tier=prediction_result.get("dropout_risk_tier", "Unknown"),
        key_risk_factors=json.dumps(prediction_result.get("key_risk_factors", {})),
        model_version=model_version
    )
    session.add(record)
    session.commit()
    return record.id


def get_students(
    limit: int = 50,
    offset: int = 0,
    target_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieves paginated student profiles with total count."""
    session = get_db(db_path)
    query = session.query(Student)
    if target_filter:
        query = query.filter(Student.actual_target == target_filter)
    total = query.count()
    students = query.order_by(Student.id).offset(offset).limit(limit).all()
    return [s.to_dict() for s in students], total


def get_student_by_id(student_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves single student record by student_id."""
    session = get_db(db_path)
    student = session.query(Student).filter_by(student_id=student_id).first()
    return student.to_dict() if student else None


def get_prediction_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent prediction audit logs."""
    session = get_db(db_path)
    preds = session.query(Prediction).order_by(desc(Prediction.created_at)).limit(limit).all()
    return [p.to_dict() for p in preds]


def get_analytics_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Computes high-level student population and prediction telemetry."""
    session = get_db(db_path)
    total_students = session.query(Student).count()
    total_predictions = session.query(Prediction).count()

    # Target class breakdown
    targets = {}
    for target in ["Graduate", "Dropout", "Enrolled"]:
        count = session.query(Student).filter(Student.actual_target == target).count()
        targets[target] = count

    # Recent high-risk count
    high_risk_count = session.query(Prediction).filter(
        Prediction.dropout_risk_tier.like("%HIGH RISK%")
    ).count()

    return {
        "total_students_enrolled": total_students,
        "total_predictions_logged": total_predictions,
        "historical_target_distribution": targets,
        "active_high_risk_alerts": high_risk_count,
        "champion_model": "Random Forest Classifier (Macro-F1: 0.7056, Dropout Recall: 72.89%)"
    }


if __name__ == "__main__":
    init_db()
    count = seed_database()
    print(f"Database ready at {DEFAULT_DB_PATH} with {count} records.")
