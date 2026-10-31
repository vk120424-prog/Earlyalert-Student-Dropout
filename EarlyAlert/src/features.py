"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/features.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Implements FR-03 Feature Engineering & Preprocessing Pipelines:
    - Encapsulates all transformations (imputation, scaling, one-hot encoding)
      inside scikit-learn Pipeline and ColumnTransformer.
    - Treats integer code columns (e.g., Course, Application mode, Mother/Father occupation)
      as categorical using OneHotEncoder(handle_unknown='ignore').
    - Provides stratified train/test split functions ensuring NO data leakage.
    - Provides feature subsets for the Leakage Experiment (Enrolment + Sem-1 vs + Sem-2).
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.validate import (
    validate_schema,
    CATEGORICAL_CODE_COLUMNS,
    CONTINUOUS_NUMERIC_COLUMNS,
    ALL_FEATURE_COLUMNS,
    ALLOWED_TARGETS
)

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "run.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("features")


# Semester 2 curricular feature names (to be excluded in Leakage Experiment Run 1)
SEM_2_COLUMNS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)"
]

# Semester 1 curricular features
SEM_1_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)"
]

# Enrolment + Background + Macroeconomic features
ENROLMENT_BACKGROUND_COLUMNS = [
    col for col in ALL_FEATURE_COLUMNS if col not in SEM_1_COLUMNS and col not in SEM_2_COLUMNS
]

# Feature subset for early-warning model: Enrolment + Sem 1 features only
EARLY_WARNING_COLUMNS = ENROLMENT_BACKGROUND_COLUMNS + SEM_1_COLUMNS


def get_feature_subsets() -> Dict[str, List[str]]:
    """
    Returns feature column groupings for standard modeling and leakage experiment.
    """
    return {
        "all_features": list(ALL_FEATURE_COLUMNS),
        "early_warning_sem1": list(EARLY_WARNING_COLUMNS),
        "sem2_columns": list(SEM_2_COLUMNS),
        "categorical_code_columns": list(CATEGORICAL_CODE_COLUMNS),
        "continuous_numeric_columns": list(CONTINUOUS_NUMERIC_COLUMNS)
    }


def build_preprocessor(
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer that:
    1. One-hot encodes categorical integer codes with most_frequent imputation and handle_unknown='ignore'.
    2. Scales continuous numeric features with median imputation and StandardScaler.

    Strictly encapsulates all preprocessing so it fits ONLY on training data.
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_CODE_COLUMNS
    if numeric_cols is None:
        numeric_cols = CONTINUOUS_NUMERIC_COLUMNS

    # Filter columns to only those actually present in the given list
    transformers = []

    if categorical_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if numeric_cols:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor


def load_and_split_data(
    raw_csv_path: str = "data/raw/student_dropout.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    feature_subset: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Loads raw CSV, runs schema validation, and creates stratified train/test split.
    Fits NOTHING. Returns X_train, X_test, y_train, y_test, preprocessor.

    Args:
        raw_csv_path: Path to student_dropout.csv
        test_size: Holdout fraction (default 0.2 as per FR-03)
        random_state: Seed for reproducibility (default 42 as per FR-03)
        feature_subset: Optional list of columns to restrict X to (e.g. for leakage experiment)

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("Loading dataset from %s for stratified split...", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    # FR-01: Schema validation before analysis
    validate_schema(df, is_inference=False)

    y = df["Target"]
    if feature_subset is not None:
        X = df[feature_subset].copy()
        active_cat = [c for c in CATEGORICAL_CODE_COLUMNS if c in feature_subset]
        active_num = [c for c in CONTINUOUS_NUMERIC_COLUMNS if c in feature_subset]
    else:
        X = df[ALL_FEATURE_COLUMNS].copy()
        active_cat = CATEGORICAL_CODE_COLUMNS
        active_num = CONTINUOUS_NUMERIC_COLUMNS

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        "Stratified split completed (seed=%d, test_ratio=%.2f):",
        random_state, test_size
    )
    logger.info("  Training set size: %d samples", len(X_train))
    logger.info("  Holdout test set size: %d samples", len(X_test))
    logger.info("  Train class distribution: %s", y_train.value_counts().to_dict())
    logger.info("  Test class distribution: %s", y_test.value_counts().to_dict())

    preprocessor = build_preprocessor(active_cat, active_num)
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep = load_and_split_data()
    print("Features module verified successfully.")
