#!/usr/bin/env bash
# ==============================================================================
# EarlyAlert: Predicting Student Dropout Risk
# End-to-End Execution Pipeline
# Author: Vasanthkumar A | Knowledge Institute of Technology
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " Starting EarlyAlert Reproducible Pipeline (Seed: 42)"
echo " Author: Vasanthkumar A | Knowledge Institute of Technology"
echo "======================================================================"

PYTHON="${PYTHON:-python}"

# 1. Ingestion
echo -e "\n[STEP 1/5] Ingesting UCI Dataset ID 697..."
$PYTHON src/load_data.py

# 2. Validation & Auditing
echo -e "\n[STEP 2/5] Running Schema Validation & Generating Audit Reports..."
$PYTHON src/validate.py

# 3. Model Arena & Tuning
echo -e "\n[STEP 3/5] Running Model Arena, Tuning & Leakage Experiment..."
$PYTHON src/train.py --seed 42

# 4. Pytest Test Suite
echo -e "\n[STEP 4/6] Executing Pytest Test Suite (27 tests)..."
$PYTHON -m pytest tests/ -v

# 5. Sample Inference Check
echo -e "\n[STEP 5/6] Running Sample High-Risk Student Prediction..."
$PYTHON src/predict.py --input data/sample_students/high_risk_student.json

# 6. Backend & Database Connectivity Check
echo -e "\n[STEP 6/6] Verifying Live Backend & Database Connection..."
$PYTHON scripts/client_query.py

echo -e "\n======================================================================"
echo " EarlyAlert Pipeline Execution Complete! All artifacts generated."
echo "======================================================================"
