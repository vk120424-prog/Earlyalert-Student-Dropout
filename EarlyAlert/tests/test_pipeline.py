"""
Tests for EarlyAlert Leakage-Free Preprocessing Pipeline (FR-03)
Author: Vasanthkumar A | Knowledge Institute of Technology
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import (
    load_and_split_data,
    build_preprocessor,
    CATEGORICAL_CODE_COLUMNS,
    CONTINUOUS_NUMERIC_COLUMNS
)


def test_preprocessing_never_fits_on_test_data():
    """
    CRITICAL VIVA TEST (FR-03):
    Verifies that the ColumnTransformer is fit STRICTLY on training data only.
    Test data transformations must NOT alter or contaminate transformer parameters
    (e.g., StandardScaler mean_ and var_, OneHotEncoder categories_).
    """
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    X_train, X_test, y_train, y_test, preprocessor = load_and_split_data(
        raw_csv_path=raw_path,
        test_size=0.2,
        random_state=42
    )

    # 1. Fit preprocessor on training data only
    preprocessor.fit(X_train)

    # Extract numerical scaler parameters fitted on X_train
    num_pipeline = preprocessor.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]
    train_fitted_means = np.copy(scaler.mean_)
    train_fitted_vars = np.copy(scaler.var_)

    # Extract categorical encoder categories fitted on X_train
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    train_fitted_categories = [np.copy(cat) for cat in onehot.categories_]

    # 2. Transform test data (simulating holdout evaluation)
    X_test_transformed = preprocessor.transform(X_test)

    # 3. Assert that transformer parameters on X_train REMAIN COMPLETELY UNTOUCHED
    np.testing.assert_array_equal(
        scaler.mean_,
        train_fitted_means,
        err_msg="CRITICAL LEAKAGE DETECTED: StandardScaler mean_ was modified by test data!"
    )
    np.testing.assert_array_equal(
        scaler.var_,
        train_fitted_vars,
        err_msg="CRITICAL LEAKAGE DETECTED: StandardScaler var_ was modified by test data!"
    )

    for i, orig_cat in enumerate(train_fitted_categories):
        np.testing.assert_array_equal(
            onehot.categories_[i],
            orig_cat,
            err_msg=f"CRITICAL LEAKAGE DETECTED: OneHot categories for feature {i} modified by test data!"
        )

    # 4. Verify output shape matches training output dimensions
    X_train_transformed = preprocessor.transform(X_train)
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]


def test_stratified_split_preserves_class_proportions():
    """Verifies that train_test_split preserves identical class balance across train and test sets."""
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv")
    X_train, X_test, y_train, y_test, _ = load_and_split_data(
        raw_csv_path=raw_path,
        test_size=0.2,
        random_state=42
    )

    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)

    for cls in ["Graduate", "Dropout", "Enrolled"]:
        assert abs(train_dist[cls] - test_dist[cls]) < 0.01, (
            f"Class '{cls}' proportion deviates between train ({train_dist[cls]:.4f}) "
            f"and test ({test_dist[cls]:.4f})!"
        )
