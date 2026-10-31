# ==============================================================================
# EarlyAlert: Predicting Student Dropout Risk
# PowerShell End-to-End Execution Script
# Author: Vasanthkumar A | Knowledge Institute of Technology
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Starting EarlyAlert Reproducible Pipeline (Seed: 42)" -ForegroundColor Cyan
Write-Host " Author: Vasanthkumar A | Knowledge Institute of Technology" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pyExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

# Step 1: Ingestion
Write-Host "`n[STEP 1/5] Ingesting UCI Dataset ID 697..." -ForegroundColor Yellow
& $pyExe src/load_data.py

# Step 2: Validation & Auditing
Write-Host "`n[STEP 2/5] Running Schema Validation & Generating Audit Reports..." -ForegroundColor Yellow
& $pyExe src/validate.py

# Step 3: Model Arena & Tuning
Write-Host "`n[STEP 3/5] Running Model Arena, Tuning & Leakage Experiment..." -ForegroundColor Yellow
& $pyExe src/train.py --seed 42

# Step 4: Pytest Test Suite
Write-Host "`n[STEP 4/6] Executing Pytest Test Suite (27 tests)..." -ForegroundColor Yellow
& $pyExe -m pytest tests/ -v

# Step 5: Sample Inference
Write-Host "`n[STEP 5/6] Testing Inference CLI..." -ForegroundColor Yellow
& $pyExe src/predict.py --input data/sample_students/high_risk_student.json

# Step 6: Backend & Database Connectivity
Write-Host "`n[STEP 6/6] Verifying Live Backend & Database Connection..." -ForegroundColor Yellow
& $pyExe scripts/client_query.py

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " EarlyAlert Pipeline Execution Complete! All artifacts verified." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
