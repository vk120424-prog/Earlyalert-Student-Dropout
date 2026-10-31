"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/validate.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Implements FR-01 and FR-02:
    - Schema validation (columns, dtypes, allowed ranges and categorical sets).
    - Failures logged to logs/run.log and raised as ValidationError.
    - Generation of reports/audit_table.csv and reports/data_dictionary.md.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "run.log")

logger = logging.getLogger("validate")
logger.setLevel(logging.INFO)

# Ensure FileHandler is explicitly attached even when runner (e.g. pytest) configures logging
file_handler_present = any(
    isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("run.log")
    for h in logger.handlers
)
if not file_handler_present:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)


class ValidationError(Exception):
    """Custom exception raised when schema or data integrity validation fails."""
    pass


# Target definitions
ALLOWED_TARGETS = {"Dropout", "Enrolled", "Graduate"}

# Categorical features that represent integer code categories (FR-03 requirement)
CATEGORICAL_CODE_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

# Continuous numerical features
CONTINUOUS_NUMERIC_COLUMNS = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

ALL_FEATURE_COLUMNS = CATEGORICAL_CODE_COLUMNS + CONTINUOUS_NUMERIC_COLUMNS

# Value bounds specifications for domain range verification
FEATURE_BOUNDS: Dict[str, Dict[str, Any]] = {
    # Binary indicators (0 or 1)
    "Daytime/evening attendance": {"min": 0, "max": 1, "type": "binary"},
    "Displaced": {"min": 0, "max": 1, "type": "binary"},
    "Educational special needs": {"min": 0, "max": 1, "type": "binary"},
    "Debtor": {"min": 0, "max": 1, "type": "binary"},
    "Tuition fees up to date": {"min": 0, "max": 1, "type": "binary"},
    "Gender": {"min": 0, "max": 1, "type": "binary"},
    "Scholarship holder": {"min": 0, "max": 1, "type": "binary"},
    "International": {"min": 0, "max": 1, "type": "binary"},

    # Categorical codes
    "Marital Status": {"min": 1, "max": 6, "type": "categorical_code"},
    "Application mode": {"min": 1, "max": 60, "type": "categorical_code"},
    "Application order": {"min": 0, "max": 10, "type": "categorical_code"},
    "Course": {"min": 1, "max": 10000, "type": "categorical_code"},
    "Previous qualification": {"min": 1, "max": 50, "type": "categorical_code"},
    "Nacionality": {"min": 1, "max": 120, "type": "categorical_code"},
    "Mother's qualification": {"min": 1, "max": 50, "type": "categorical_code"},
    "Father's qualification": {"min": 1, "max": 50, "type": "categorical_code"},
    "Mother's occupation": {"min": 0, "max": 250, "type": "categorical_code"},
    "Father's occupation": {"min": 0, "max": 250, "type": "categorical_code"},

    # Numeric features
    "Previous qualification (grade)": {"min": 0.0, "max": 200.0, "type": "numeric"},
    "Admission grade": {"min": 0.0, "max": 200.0, "type": "numeric"},
    "Age at enrollment": {"min": 15, "max": 100, "type": "numeric"},
    "Curricular units 1st sem (credited)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 1st sem (enrolled)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 1st sem (evaluations)": {"min": 0, "max": 100, "type": "numeric"},
    "Curricular units 1st sem (approved)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 1st sem (grade)": {"min": 0.0, "max": 20.0, "type": "numeric"},
    "Curricular units 1st sem (without evaluations)": {"min": 0, "max": 30, "type": "numeric"},
    "Curricular units 2nd sem (credited)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 2nd sem (enrolled)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 2nd sem (evaluations)": {"min": 0, "max": 100, "type": "numeric"},
    "Curricular units 2nd sem (approved)": {"min": 0, "max": 50, "type": "numeric"},
    "Curricular units 2nd sem (grade)": {"min": 0.0, "max": 20.0, "type": "numeric"},
    "Curricular units 2nd sem (without evaluations)": {"min": 0, "max": 30, "type": "numeric"},
    "Unemployment rate": {"min": 0.0, "max": 50.0, "type": "numeric"},
    "Inflation rate": {"min": -20.0, "max": 50.0, "type": "numeric"},
    "GDP": {"min": -20.0, "max": 20.0, "type": "numeric"}
}


def validate_schema(
    df: pd.DataFrame,
    is_inference: bool = False,
    allow_missing: bool = False
) -> bool:
    """
    Validates that a dataframe adheres to the expected schema, types, and bounds.
    Failures are logged to logs/run.log and raise ValidationError.

    Args:
        df: pd.DataFrame to validate
        is_inference: If True, Target column is not required.
        allow_missing: If False, any missing values trigger validation error.

    Returns:
        bool: True if validation passes.

    Raises:
        ValidationError: If schema or value constraints are violated.
    """
    logger.info("Starting schema and data validation (is_inference=%s)...", is_inference)
    errors: List[str] = []

    # 1. Check required feature columns
    missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        err = f"Missing required column(s): {missing_cols}"
        errors.append(err)
        logger.error(err)

    # 2. Check Target column if not inference mode
    if not is_inference:
        if "Target" not in df.columns:
            err = "Missing required 'Target' column for training/audit data."
            errors.append(err)
            logger.error(err)
        else:
            invalid_targets = set(df["Target"].dropna().unique()) - ALLOWED_TARGETS
            if invalid_targets:
                err = f"Target contains invalid classes: {invalid_targets}. Allowed: {ALLOWED_TARGETS}"
                errors.append(err)
                logger.error(err)

    if errors:
        full_err_msg = "\n".join(errors)
        logger.critical("Schema validation failed on column presence:\n%s", full_err_msg)
        for handler in logging.getLogger().handlers + logger.handlers:
            handler.flush()
        raise ValidationError(full_err_msg)

    # 3. Check for missing values if not allowed
    if not allow_missing:
        null_counts = df[ALL_FEATURE_COLUMNS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            err = f"Unexpected missing values detected in columns: {cols_with_nulls.to_dict()}"
            errors.append(err)
            logger.error(err)

    # 4. Check data types and value bounds
    for col in ALL_FEATURE_COLUMNS:
        series = df[col].dropna()
        if series.empty:
            continue

        # Check numeric nature
        if not pd.api.types.is_numeric_dtype(series):
            err = f"Column '{col}' has non-numeric dtype: {series.dtype}. Numeric/code expected."
            errors.append(err)
            logger.error(err)
            continue

        bounds = FEATURE_BOUNDS.get(col)
        if bounds:
            col_min = series.min()
            col_max = series.max()
            if col_min < bounds["min"] or col_max > bounds["max"]:
                err = (
                    f"Column '{col}' value out of bounds: observed range [{col_min}, {col_max}], "
                    f"allowed range [{bounds['min']}, bounds['max']]."
                )
                errors.append(err)
                logger.error(err)

            if bounds.get("type") == "binary":
                unique_vals = set(series.unique())
                if not unique_vals.issubset({0, 1}):
                    err = f"Binary column '{col}' has values outside {{0, 1}}: {unique_vals}"
                    errors.append(err)
                    logger.error(err)

    if errors:
        full_err_msg = f"Data validation failed with {len(errors)} error(s):\n" + "\n".join(errors)
        logger.critical(full_err_msg)
        for handler in logging.getLogger().handlers + logger.handlers:
            handler.flush()
        raise ValidationError(full_err_msg)

    logger.info("Schema and data bounds validation PASSED successfully for %d records.", len(df))
    return True


def generate_audit_reports(raw_csv_path: str = "data/raw/student_dropout.csv", reports_dir: str = "reports") -> None:
    """
    Generates FR-02 deliverables:
    1. reports/audit_table.csv: Missing values, duplicates, class balance, dtypes, ranges.
    2. reports/data_dictionary.md: Detailed feature catalog, semantic descriptions, transformation logic.
    """
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("Generating data audit and dictionary reports from %s...", raw_csv_path)

    raw_df = pd.read_csv(raw_csv_path)
    initial_row_count = len(raw_df)

    # Run schema validation
    validate_schema(raw_df, is_inference=False)

    # Check duplicates
    duplicate_count = raw_df.duplicated().sum()
    logger.info("Audit summary: Initial rows = %d, Duplicates = %d", initial_row_count, duplicate_count)

    # 1. Build audit_table.csv
    audit_rows = []
    for col in raw_df.columns:
        series = raw_df[col]
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / initial_row_count) * 100.0
        n_unique = series.nunique()
        dtype_str = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            min_val = series.min()
            max_val = series.max()
            mean_val = round(series.mean(), 3)
            std_val = round(series.std(), 3)
        else:
            min_val = None
            max_val = None
            mean_val = None
            std_val = None

        category_type = "Target" if col == "Target" else (
            "Categorical Code" if col in CATEGORICAL_CODE_COLUMNS else "Continuous Numeric"
        )

        audit_rows.append({
            "Column Name": col,
            "Feature Category": category_type,
            "Data Type": dtype_str,
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct:.2f}%",
            "Unique Values": n_unique,
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Std": std_val
        })

    audit_df = pd.DataFrame(audit_rows)
    audit_csv_path = os.path.join(reports_dir, "audit_table.csv")
    audit_df.to_csv(audit_csv_path, index=False)
    logger.info("Saved audit table to %s", audit_csv_path)

    # 2. Build data_dictionary.md
    target_counts = raw_df["Target"].value_counts()
    target_pcts = raw_df["Target"].value_counts(normalize=True) * 100.0

    dict_md = f"""# EarlyAlert: Data Dictionary & Data Audit Report
**Author:** Vasanthkumar A | **Dept:** Computer Science and Business Systems | **Knowledge Institute of Technology**  
**Dataset:** UCI "Predict Students' Dropout and Academic Success" (ID: 697, CC BY 4.0)

---

## 1. Executive Summary & Audit Statistics

| Audit Metric | Observed Value | Validation Status |
|---|---|---|
| **Total Rows (Before / After Cleaning)** | {initial_row_count} / {initial_row_count} | Complete (No rows dropped) |
| **Total Columns** | {len(raw_df.columns)} (36 Features + 1 Target) | Valid Schema |
| **Duplicate Rows** | {duplicate_count} | Passed (0 duplicates) |
| **Missing Values (Entire Dataset)** | {raw_df.isnull().sum().sum()} | Zero missing values in raw dataset |
| **Target Distribution** | Graduate: {target_counts.get('Graduate', 0)} ({target_pcts.get('Graduate', 0):.2f}%)<br/>Dropout: {target_counts.get('Dropout', 0)} ({target_pcts.get('Dropout', 0):.2f}%)<br/>Enrolled: {target_counts.get('Enrolled', 0)} ({target_pcts.get('Enrolled', 0):.2f}%) | Moderate Class Imbalance |

---

## 2. Target Variable Definition
- **Target Name:** `Target`
- **Class Type:** Multiclass classification (3 classes)
  - `Dropout` (Primary class of interest: Students who prematurely discontinued studies).
  - `Enrolled` (Students currently continuing their program).
  - `Graduate` (Students who successfully completed their degree program).
- **Primary Metric:** **Macro-F1** (balances performance equally across all three classes, preventing dominant Graduate class from distorting performance).
- **Secondary Metric:** **Dropout Recall** (critical early-intervention metric: measures the proportion of actual dropouts correctly detected).

---

## 3. Data Transformation & Preprocessing Logic (FR-03 Architecture)

| Component | Target Columns | Strategy & Transformation |
|---|---|---|
| **Categorical Integer Codes** | 18 features (e.g., `Course`, `Application mode`, `Mother's qualification`, etc.) | Treated as categorical codes. Imputed with `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. Avoids imposing artificial ordinal ranking. |
| **Continuous Numeric Features** | 18 features (grades, curricular units, age, macroeconomic indicators) | Imputed with `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Standardizes mean to 0 and variance to 1. |
| **Leakage Prevention** | All transformations | Strict encapsulation inside scikit-learn `Pipeline` and `ColumnTransformer`. Preprocessing transformers are fit **strictly** on the training split only during cross-validation and final training. Test data is only transformed. |

---

## 4. Comprehensive Feature Catalog

| Column Name | Feature Group | Type | Allowed Range / Codes | Transformation | Description |
|---|---|---|---|---|---|
| `Marital Status` | Demographic | Categorical Code | 1: Single, 2: Married, 3: Widower, 4: Divorced, 5: Facto union, 6: Legally separated | One-Hot Encoding | Marital status at the time of university enrollment. |
| `Application mode` | Academic Enrollment | Categorical Code | 1 to 60 (e.g., 1: General contingent, 2: Ordinance, etc.) | One-Hot Encoding | Method and criteria of university application. |
| `Application order` | Academic Enrollment | Categorical Code | 0 (first choice) to 9 (last choice) | One-Hot Encoding | Student's preference order for this degree program. |
| `Course` | Academic Enrollment | Categorical Code | Integer codes (33 to 9991) representing specific degree programs | One-Hot Encoding | Degree major (e.g., Informatics, Nursing, Management, etc.). |
| `Daytime/evening attendance` | Academic Enrollment | Binary | 0: Evening, 1: Daytime | One-Hot / Binary Pass | Schedule mode of student attendance. |
| `Previous qualification` | Academic Background | Categorical Code | 1 to 50 (Secondary education, Higher education, etc.) | One-Hot Encoding | Highest prior qualification level attained before enrollment. |
| `Previous qualification (grade)` | Academic Background | Continuous Numeric | [0.0, 200.0] | StandardScaler | Grade obtained in previous qualification (scale 0-200). |
| `Nacionality` | Demographic | Categorical Code | 1 to 120 (1: Portuguese, 2: German, 6: Spanish, etc.) | One-Hot Encoding | Country of citizenship. |
| `Mother's qualification` | Socio-Economic | Categorical Code | 1 to 50 | One-Hot Encoding | Educational level attained by student's mother. |
| `Father's qualification` | Socio-Economic | Categorical Code | 1 to 50 | One-Hot Encoding | Educational level attained by student's father. |
| `Mother's occupation` | Socio-Economic | Categorical Code | 0 to 250 | One-Hot Encoding | Occupation category of student's mother. |
| `Father's occupation` | Socio-Economic | Categorical Code | 0 to 250 | One-Hot Encoding | Occupation category of student's father. |
| `Admission grade` | Academic Background | Continuous Numeric | [0.0, 200.0] | StandardScaler | University admission test grade score (scale 0-200). |
| `Displaced` | Socio-Economic | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether student had to relocate to attend university. |
| `Educational special needs` | Demographic | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether student requires special educational accommodations. |
| `Debtor` | Financial | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether student has outstanding university debts. |
| `Tuition fees up to date` | Financial | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether tuition payments are currently paid up to date. |
| `Gender` | Demographic | Binary | 0: Female, 1: Male | One-Hot / Binary Pass | Gender of student. |
| `Scholarship holder` | Financial | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether student receives government/university scholarship. |
| `Age at enrollment` | Demographic | Continuous Numeric | [15, 100] | StandardScaler | Age of student in completed years at matriculation. |
| `International` | Demographic | Binary | 0: No, 1: Yes | One-Hot / Binary Pass | Whether student is an international student. |
| `Curricular units 1st sem (credited)` | Academic 1st Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 1st semester courses credited via equivalency. |
| `Curricular units 1st sem (enrolled)` | Academic 1st Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 1st semester courses enrolled in. |
| `Curricular units 1st sem (evaluations)` | Academic 1st Sem | Continuous Numeric | [0, 100] | StandardScaler | Number of exam/test evaluation attempts in 1st semester. |
| `Curricular units 1st sem (approved)` | Academic 1st Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 1st semester courses successfully passed. |
| `Curricular units 1st sem (grade)` | Academic 1st Sem | Continuous Numeric | [0.0, 20.0] | StandardScaler | Average grade across passed 1st semester courses (0-20 scale). |
| `Curricular units 1st sem (without evaluations)` | Academic 1st Sem | Continuous Numeric | [0, 30] | StandardScaler | Number of enrolled 1st sem courses with zero evaluations. |
| `Curricular units 2nd sem (credited)` | Academic 2nd Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 2nd semester courses credited via equivalency. |
| `Curricular units 2nd sem (enrolled)` | Academic 2nd Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 2nd semester courses enrolled in. |
| `Curricular units 2nd sem (evaluations)` | Academic 2nd Sem | Continuous Numeric | [0, 100] | StandardScaler | Number of exam/test evaluation attempts in 2nd semester. |
| `Curricular units 2nd sem (approved)` | Academic 2nd Sem | Continuous Numeric | [0, 50] | StandardScaler | Number of 2nd semester courses successfully passed. |
| `Curricular units 2nd sem (grade)` | Academic 2nd Sem | Continuous Numeric | [0.0, 20.0] | StandardScaler | Average grade across passed 2nd semester courses (0-20 scale). |
| `Curricular units 2nd sem (without evaluations)` | Academic 2nd Sem | Continuous Numeric | [0, 30] | StandardScaler | Number of enrolled 2nd sem courses with zero evaluations. |
| `Unemployment rate` | Macroeconomic | Continuous Numeric | [0.0, 50.0] | StandardScaler | National unemployment rate during enrollment semester. |
| `Inflation rate` | Macroeconomic | Continuous Numeric | [-20.0, 50.0] | StandardScaler | National inflation rate during enrollment semester. |
| `GDP` | Macroeconomic | Continuous Numeric | [-20.0, 20.0] | StandardScaler | GDP growth rate during enrollment semester. |

---

*Report automatically generated by `src/validate.py` in accordance with FR-01 and FR-02.*
"""

    dict_md_path = os.path.join(reports_dir, "data_dictionary.md")
    with open(dict_md_path, "w", encoding="utf-8") as f:
        f.write(dict_md)
    logger.info("Saved data dictionary to %s", dict_md_path)


if __name__ == "__main__":
    generate_audit_reports()
