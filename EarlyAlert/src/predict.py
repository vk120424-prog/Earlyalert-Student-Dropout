"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/predict.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Implements FR-06:
    - Standalone inference CLI accepting a JSON file with a new student record.
    - Validates every field (type, range, missing values) strictly before inference.
    - Returns predicted class with full probability distribution and risk tier.
    - On validation error, logs failure to logs/run.log and displays clear diagnostic message.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import logging
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.validate import (
    ALL_FEATURE_COLUMNS,
    FEATURE_BOUNDS,
    ValidationError
)

# Configure logging
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
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
logger = logging.getLogger("predict")


def validate_student_record(record: Dict[str, Any]) -> Tuple[bool, list[str]]:
    """
    Validates every field of a single student record:
    1. Check presence of all 36 required features.
    2. Check for unexpected None/missing values.
    3. Check numeric data type.
    4. Check domain range/bounds from FEATURE_BOUNDS.

    Returns:
        Tuple[bool, list[str]]: (is_valid, list_of_error_messages)
    """
    errors = []

    # 1. Missing fields check
    for col in ALL_FEATURE_COLUMNS:
        if col not in record:
            errors.append(f"Missing required field: '{col}'")
        elif record[col] is None:
            errors.append(f"Field '{col}' cannot be null or empty.")

    # 2. Type and bounds checks
    for col, val in record.items():
        if col not in ALL_FEATURE_COLUMNS:
            continue  # Extra/metadata fields are tolerated or ignored

        if val is None:
            continue

        # Check numeric type
        if not isinstance(val, (int, float, np.number)):
            errors.append(f"Field '{col}' has invalid type '{type(val).__name__}'. Expected numeric (int or float).")
            continue

        # Check domain bounds
        bounds = FEATURE_BOUNDS.get(col)
        if bounds:
            min_val = bounds["min"]
            max_val = bounds["max"]
            if val < min_val or val > max_val:
                errors.append(
                    f"Field '{col}' value {val} is out of allowable range [{min_val}, {max_val}]."
                )

            if bounds.get("type") == "binary" and val not in (0, 1, 0.0, 1.0):
                errors.append(f"Binary field '{col}' must be 0 or 1, got {val}.")

    is_valid = len(errors) == 0
    return is_valid, errors


def predict_student(
    input_data: Dict[str, Any] | str,
    model_path: str = "models/model.joblib"
) -> Dict[str, Any]:
    """
    Loads model, validates input student record, and outputs prediction.

    Args:
        input_data: Dict containing student features, or filepath to JSON file.
        model_path: Path to serialized model.joblib.

    Returns:
        Dict[str, Any]: Prediction result containing predicted_class, probabilities, and risk_tier.

    Raises:
        ValidationError: If student record fails validation.
        FileNotFoundError: If model or input file is not found.
    """
    # 1. Load input data if path string provided
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            err_msg = f"Input file not found at: {input_data}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        try:
            with open(input_data, "r", encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError as exc:
            err_msg = f"Failed to parse JSON file '{input_data}': {exc}"
            logger.error(err_msg)
            raise ValidationError(err_msg)
    elif isinstance(input_data, dict):
        record = input_data
    else:
        err_msg = f"Invalid input format: {type(input_data)}. Expected dict or JSON filepath."
        logger.error(err_msg)
        raise ValidationError(err_msg)

    # 2. Strict field validation
    is_valid, errors = validate_student_record(record)
    if not is_valid:
        error_summary = f"Student record validation failed with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error("Inference validation rejected record:\n%s", error_summary)
        raise ValidationError(error_summary)

    # 3. Load model pipeline
    if not os.path.exists(model_path):
        err_msg = f"Trained model artifact not found at: {model_path}. Run 'python src/train.py' first."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        err_msg = f"Failed to deserialize model pipeline from {model_path}: {exc}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # 4. Construct single-row DataFrame with exact schema
    student_df = pd.DataFrame([{col: record[col] for col in ALL_FEATURE_COLUMNS}])

    # 5. Execute prediction
    try:
        predicted_class = str(pipeline.predict(student_df)[0])
        probabilities = {}
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(student_df)[0]
            classes = list(pipeline.classes_)
            probabilities = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
    except Exception as exc:
        err_msg = f"Error during model pipeline inference: {exc}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # 6. Assess Dropout Risk Tier
    dropout_prob = probabilities.get("Dropout", 0.0)
    if dropout_prob >= 0.60:
        risk_tier = "HIGH RISK (Immediate Intervention Required)"
    elif dropout_prob >= 0.35:
        risk_tier = "MODERATE RISK (Proactive Advising Recommended)"
    else:
        risk_tier = "LOW RISK (Normal Academic Progress)"

    result = {
        "status": "SUCCESS",
        "predicted_class": predicted_class,
        "dropout_risk_tier": risk_tier,
        "probabilities": probabilities,
        "key_risk_factors": {
            "tuition_fees_overdue": bool(record.get("Tuition fees up to date") == 0),
            "debtor_status": bool(record.get("Debtor") == 1),
            "sem1_approved_ratio": f"{record.get('Curricular units 1st sem (approved)', 0)} / {record.get('Curricular units 1st sem (enrolled)', 1)}",
            "sem2_approved_ratio": f"{record.get('Curricular units 2nd sem (approved)', 0)} / {record.get('Curricular units 2nd sem (enrolled)', 1)}"
        }
    }

    logger.info("Inference completed successfully: Predicted='%s', Dropout Prob=%.2f%%, Tier='%s'",
                predicted_class, dropout_prob * 100, risk_tier)
    return result


def main():
    parser = argparse.ArgumentParser(description="EarlyAlert Student Dropout Prediction CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input student JSON file")
    parser.add_argument("--model", "-m", type=str, default="models/model.joblib", help="Path to model artifact (default: models/model.joblib)")
    args = parser.parse_args()

    try:
        result = predict_student(input_data=args.input, model_path=args.model)
        print("\n" + json.dumps(result, indent=2))
        sys.exit(0)
    except (ValidationError, FileNotFoundError, RuntimeError) as err:
        sys.stderr.write(f"\n[ERROR] EarlyAlert Prediction Failed:\n{err}\n\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
