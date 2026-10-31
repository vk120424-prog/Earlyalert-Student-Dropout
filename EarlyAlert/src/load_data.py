"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/load_data.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Downloads the UCI 'Predict Students' Dropout and Academic Success' dataset (id 697, CC BY 4.0),
    saves the raw data to data/raw/student_dropout.csv, logs the dataset version,
    download date, and metadata to logs/run.log.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
import pandas as pd

# Configure logging to write to both logs/run.log and console
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
logger = logging.getLogger("load_data")


def download_uci_dataset(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Download UCI dataset ID 697 using ucimlrepo with direct fallback.
    Saves student_dropout.csv and dataset_metadata.json in raw_dir.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]: (features_df, targets_df, metadata_dict)
    """
    os.makedirs(raw_dir, exist_ok=True)
    logger.info("Initiating download for UCI Dataset ID 697 ('Predict Students' Dropout and Academic Success')...")

    download_timestamp = datetime.now(timezone.utc).isoformat()
    raw_csv_path = os.path.join(raw_dir, "student_dropout.csv")
    metadata_path = os.path.join(raw_dir, "dataset_metadata.json")

    metadata = {
        "dataset_id": 697,
        "dataset_name": "Predict Students' Dropout and Academic Success",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "download_date_utc": download_timestamp,
        "source": "UCI Machine Learning Repository",
        "reference_url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
    }

    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=697)
        features = dataset.data.features
        targets = dataset.data.targets

        metadata["version"] = dataset.metadata.get("version", "1.0")
        metadata["uci_metadata"] = {
            "num_instances": dataset.metadata.get("num_instances", len(features)),
            "num_features": dataset.metadata.get("num_features", features.shape[1]),
            "description": dataset.metadata.get("abstract", "")
        }
        logger.info("Dataset successfully retrieved via ucimlrepo package.")

    except Exception as exc:
        logger.warning("ucimlrepo fetch encountered an issue: %s. Attempting fallback URL download...", exc)
        fallback_url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
        import urllib.request
        import zipfile
        import io

        req = urllib.request.Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            zip_bytes = io.BytesIO(response.read())
            with zipfile.ZipFile(zip_bytes) as z:
                # Inside the zip there is data.csv
                csv_filename = [f for f in z.namelist() if f.endswith(".csv")][0]
                with z.open(csv_filename) as f:
                    full_df = pd.read_csv(f, sep=";")

        target_col = "Target"
        features = full_df.drop(columns=[target_col])
        targets = full_df[[target_col]]
        metadata["version"] = "1.0-fallback"
        logger.info("Dataset successfully downloaded from UCI fallback URL.")

    # Combine features and target for a clean single raw CSV
    full_df = features.copy()
    target_col_name = "Target" if "Target" in targets.columns else targets.columns[0]
    full_df["Target"] = targets[target_col_name].values

    full_df.to_csv(raw_csv_path, index=False)
    logger.info("Raw dataset saved to: %s (Shape: %d rows, %d columns)", raw_csv_path, full_df.shape[0], full_df.shape[1])

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Dataset metadata saved to: %s", metadata_path)

    # Log class balance
    class_dist = full_df["Target"].value_counts().to_dict()
    logger.info("Target class distribution: %s", class_dist)

    return features, targets, metadata


if __name__ == "__main__":
    download_uci_dataset()
