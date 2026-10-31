# EarlyAlert: Predicting Student Dropout Risk

**Author:** Vasanthkumar A  
**Department:** Computer Science and Business Systems  
**Academic Year:** III Year / V Semester  
**Institution:** Knowledge Institute of Technology  
**License:** MIT (Code) | Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## 1. Executive Summary

Higher education institutions face persistent challenges with student attrition. Premature dropouts incur substantial personal, institutional, and economic costs. **EarlyAlert** is an end-to-end, reproducible supervised machine learning system engineered to predict academic outcomes—specifically identifying students at imminent risk of dropout—while enforcing zero-data-leakage preprocessing, rigorous schema validation, and fair subgroup diagnostics.

### Problem Formulation
- **Target Variable (`Target`):** 3-Class Multiclass Classification
  - `Dropout` (Primary class of intervention interest: 1,421 students, 32.12%)
  - `Enrolled` (Students continuing active study: 794 students, 17.95%)
  - `Graduate` (Students who completed degree requirements: 2,209 students, 49.93%)
- **Primary Metric:** **Macro-F1** (equally weights harmonic mean of precision and recall across all three classes, preventing the dominant `Graduate` class from inflating performance scores).
- **Secondary Operational Metric:** **Dropout Recall** (proportion of actual dropouts detected; critical for early-intervention sensitivity).

---

## 2. Project Architecture & Repository Layout

```
EarlyAlert/
├── .gitignore
├── Makefile                       # Cross-platform automation (setup, download, train, test)
├── README.md                      # Comprehensive documentation and viva guide
├── requirements.txt               # Pinned dependencies for deterministic reproduction
├── run_all.sh                     # Bash pipeline runner
├── run_all.ps1                    # PowerShell pipeline runner
├── data/
│   ├── raw/                       # Raw UCI dataset and metadata
│   │   ├── student_dropout.csv
│   │   └── dataset_metadata.json
│   ├── earlyalert.db              # SQLite database (students, predictions, audit_logs)
│   ├── processed/                 # Intermediate processed splits
│   └── sample_students/           # JSON files for standalone inference testing
│       ├── high_risk_student.json
│       ├── low_risk_student.json
│       └── invalid_student.json
├── logs/
│   ├── .gitkeep
│   └── run.log                    # Execution and validation logs (failures never ignored)
├── models/
│   ├── .gitkeep
│   └── model.joblib               # Serialized champion scikit-learn Pipeline
├── notebooks/
│   ├── 01_audit.ipynb             # Visual data audit and schema checks
│   └── 02_modeling.ipynb          # Modeling arena, CV analysis and subgroup inspection
├── reports/
│   ├── audit_table.csv            # Feature-by-feature missingness, types, ranges
│   ├── data_dictionary.md         # Full feature documentation & transformation logic
│   ├── error_analysis.md          # Pedagogical case studies on misclassified students
│   ├── leakage_experiment.md      # Sem-1 vs Sem-2 temporal leakage comparison
│   ├── model_comparison.csv       # Holdout benchmark table across all models
│   ├── subgroup_analysis.csv      # Disaggregated fairness slices (gender, age, tuition)
│   ├── tuning_log.csv             # Complete trial-by-trial hyperparameter logs (41 trials)
│   └── figures/
│       ├── confusion_matrix_all_models.png
│       ├── confusion_matrix_random_forest.png
│       ├── confusion_matrix_logistic_regression_(balanced).png
│       ├── confusion_matrix_hist_gradient_boosting.png
│       ├── confusion_matrix_dummy_(most_frequent).png
│       └── pipeline.png           # End-to-end pipeline architecture diagram
├── scripts/
│   └── client_query.py            # API & SQLite connection demonstration client
├── src/
│   ├── __init__.py
│   ├── app.py                     # FastAPI backend & Academic Advisor Dashboard
│   ├── database.py                # SQLite & SQLAlchemy data access layer
│   ├── load_data.py               # Ingestion from UCI Repository ID 697
│   ├── validate.py                # Strict schema, dtype and range validation (FR-01, FR-02)
│   ├── features.py                # Scikit-learn ColumnTransformer pipelines (FR-03)
│   ├── train.py                   # 5-fold CV hyperparameter tuning arena (FR-03, FR-04)
│   ├── evaluate.py                # Holdout metrics, subgroup diagnostics, error analysis (FR-05)
│   └── predict.py                 # Standalone validated inference CLI (FR-06)
└── tests/
    ├── test_api.py                # REST API endpoint tests (FastAPI TestClient)
    ├── test_database.py           # SQLite database schema, seeding, and audit tests
    ├── test_pipeline.py           # Verifies preprocessing never fits on test data
    ├── test_predict.py            # Verifies CLI rejection of missing/corrupted fields
    └── test_validation.py         # Verifies schema validation and logging of failures
```

---

## 3. Dataset Specifications

- **Source:** UCI Machine Learning Repository, Dataset ID 697 (*"Predict Students' Dropout and Academic Success"*).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Instances:** 4,424 students.
- **Attributes:** 36 input features categorized into:
  - **Academic Background & Enrollment:** Course, Application mode, Admission grade, Attendance mode.
  - **Demographics & Socio-Economic:** Age, Gender, Nationality, Mother/Father qualification and occupation.
  - **Financial Indicators:** Debtor status, Tuition fees up to date, Scholarship holder.
  - **Curricular Performance (Semester 1 & 2):** Enrolled units, evaluations, approved units, average grade, non-evaluated units.
  - **Macroeconomic Environment:** Unemployment rate, Inflation rate, GDP growth.

---

## 4. Empirical Evaluation & Benchmark Results

All models were tuned strictly on the 80% training set (3,539 students) using 5-Fold Stratified Cross-Validation (`StratifiedKFold`, `seed=42`) and evaluated on the **untouched holdout test set (885 students)**.

### Model Arena Comparison Table (Holdout Test Set)

| Model Architecture | 5-Fold CV Macro-F1 | Holdout Macro-F1 | Holdout Accuracy | Dropout Recall | Enrolled Recall | Graduate Recall | Best Hyperparameters |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.7217** | **0.7056** | **75.59%** | **72.89%** | 52.83% | 85.52% | `n_estimators=100`, `max_depth=None`, `class_weight='balanced'`, `min_samples_split=2` |
| **Logistic Regression (Balanced)** | 0.7109 | 0.7005 | 73.79% | 68.31% | 65.41% | 80.32% | `C=0.05`, `solver='lbfgs'` |
| **Hist Gradient Boosting** | 0.7140 | 0.6933 | 76.27% | 72.54% | 42.14% | 90.95% | `learning_rate=0.05`, `max_depth=6`, `max_iter=100` |
| **Dummy (Most Frequent)** | 0.2220 | 0.2221 | 49.94% | 0.00% | 0.00% | 100.0% | `strategy='most_frequent'` |

> *All values are empirically measured and recorded in `reports/model_comparison.csv` and `reports/tuning_log.csv`.*

---

## 5. Subgroup Diagnostics (FR-05)

To ensure the early-warning model performs equitably across diverse student populations, holdout predictions were disaggregated across four slices:

| Demographic / Financial Slice | Subgroup Category | Test Samples | Sample % | Actual Dropouts | Predicted Dropouts | Macro-F1 | Dropout Recall | Subgroup Accuracy |
|---|---|---|---|---|---|---|---|---|
| **Tuition Fees Status** | **Tuition Overdue (0)** | 109 | 12.3% | 93 | 105 | 0.5084 | **100.0%** | 88.07% |
| | **Tuition Up to Date (1)** | 776 | 87.8% | 191 | 149 | 0.6772 | 59.69% | 73.84% |
| **Age Band** | **Age > 25 years** | 215 | 24.3% | 119 | 118 | 0.7184 | **85.71%** | 80.00% |
| | **Age 21 - 25 years** | 180 | 20.4% | 58 | 53 | 0.7175 | 72.41% | 74.44% |
| | **Age <= 20 years** | 490 | 55.4% | 107 | 83 | 0.6675 | 58.88% | 74.08% |
| **Gender** | **Male (1)** | 315 | 35.6% | 145 | 126 | 0.7088 | **74.48%** | 74.29% |
| | **Female (0)** | 570 | 64.5% | 139 | 128 | 0.6956 | **71.22%** | 76.32% |
| **Scholarship Status** | **Non-Scholarship (0)** | 680 | 76.9% | 265 | 237 | 0.7041 | **74.34%** | 73.38% |
| | **Scholarship Holder (1)** | 205 | 23.2% | 19 | 17 | 0.5871 | 52.63% | 82.93% |

### Key Diagnostic Insights:
1. **Tuition Overdue Indicator is a Critical Alert Trigger:** The model correctly captured **100% of dropouts (93/93)** among students with overdue tuition fees.
2. **Older Students Experience Elevated Attrition:** Non-traditional students (>25 years) had a dropout rate of 55.3% (119/215); the model identified them with **85.71% recall**.
3. **Gender Parity:** Dropout recall between male (74.48%) and female (71.22%) students is well-balanced within 3.26 percentage points.

---

## 6. Data Leakage Experiment: Horizon Comparison

To evaluate temporal target leakage, the champion model was trained under two distinct feature horizons:

| Metric | Run 1: Early Warning<br/>(Enrolment + Sem-1 Features Only) | Run 2: Full Horizon<br/>(Enrolment + Sem-1 + Sem-2 Features) | Operational Impact |
|---|---|---|---|
| **Feature Count** | 30 features | 36 features (+6 Sem-2 features) | Realistic early window vs late audit |
| **Holdout Macro-F1** | **0.5622** | **0.6364** (+0.0742) | Higher certainty after full first year |
| **Dropout Recall** | **72.54%** | **76.76%** (+4.22%) | **72.54% of dropouts detected a full semester earlier!** |
| **Holdout Accuracy** | 71.75% | 75.37% (+3.62%) | Reliable decision boundary |

**Conclusion:** Deploying Run 1 at the conclusion of Semester 1 provides actionable lead time for preventative academic advising, capturing nearly three-quarters of eventual dropouts before second-semester failure occurs.

---

## 7. Quickstart & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- Git

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/vasanthkumar-a/EarlyAlert.git
cd EarlyAlert

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Single-Command End-to-End Run
```bash
# Deterministic run with fixed seed 42
python src/train.py --seed 42
```
*Or using the automated scripts:*
- Windows: `powershell -ExecutionPolicy Bypass -File run_all.ps1`
- Linux/macOS: `bash run_all.sh` or `make all`

### 3. Run Pytest Suite
```bash
pytest -v
```
*Validates schema failure on bad columns, input rejection on predict.py, and mathematical proof of zero data leakage.*

### 4. Standalone Student Risk Inference CLI (FR-06)
```bash
# High-risk student prediction
python src/predict.py --input data/sample_students/high_risk_student.json

# Low-risk student prediction
python src/predict.py --input data/sample_students/low_risk_student.json

# Deliberately invalid input (demonstrating validation rejection & error logging)
python src/predict.py --input data/sample_students/invalid_student.json
```

### 5. Backend REST API & Database Server
```bash
# Start FastAPI backend server with live database and model connection
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000

# Run automated backend & database demonstration client
python scripts/client_query.py
```

---

## 8. Backend REST API & Database Architecture

EarlyAlert features an enterprise-grade backend service built on **FastAPI**, **SQLAlchemy**, and **SQLite**, connecting model inference with student data management and audit trails.

### Database Schema (`data/earlyalert.db`)
- **`students`**: Stores all 36 curricular, demographic, and socioeconomic features per student, along with unique ID (`student_id`), enrollment status, and timestamps. Pre-seeded with 1,000 historical records from the UCI repository.
- **`predictions`**: Audit trail for all model predictions, logging predicted class, calibrated probabilities (`Dropout`, `Enrolled`, `Graduate`), risk tier, identified risk drivers (e.g. overdue tuition, debtor flag), model version, and UTC timestamps.
- **`audit_logs`**: System event tracking for data seeding, schema validations, and inference events.

### Live Endpoints & User Interfaces
- **Interactive Advisor Dashboard:** [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) — Modern web dashboard for academic counselors to test student profiles, triage high-risk dropouts, and inspect enrolled student records.
- **Swagger UI Interactive Docs:** [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) — OpenAPI interactive documentation.
- **Health Telemetry:** `GET /api/health` — Verifies database connection, model status, and system health.
- **Population Analytics:** `GET /api/analytics` — Student cohort counts, active high-risk alerts, and target distributions.
- **Paginated Student Queries:** `GET /api/students?limit=25&offset=0` — Filter and explore student database records.
- **Real-Time Risk Prediction:** `POST /api/predict` — Validates student JSON, executes champion pipeline, logs audit record, and returns calibrated probabilities and intervention flags.
- **Prediction Audit History:** `GET /api/predictions` — Query historical inference logs.
- **Model Benchmarks:** `GET /api/metrics` — Subgroup diagnostics and model comparison metrics.

---

## 9. Viva Voce Defense Guide (Architectural Decisions)

| Examiner Question | Prepared Technical Defense |
|---|---|
| **Q1: Why choose Macro-F1 as the primary metric instead of Accuracy?** | *Dataset exhibits notable class imbalance (49.9% Graduate, 32.1% Dropout, 18.0% Enrolled). A trivial Dummy classifier achieves ~50% accuracy by always predicting Graduate while completely missing every single dropout (0% recall). Macro-F1 calculates the harmonic mean of precision and recall independently per class and averages them unweighted, ensuring equal penalty for misclassifying minority and high-risk classes.* |
| **Q2: Why treat integer codes as categorical rather than numerical?** | *Features like `Course` (e.g., 9147, 9500), `Application mode` (1 to 57), and `Mother's occupation` (0 to 195) are nominal qualitative identifiers. Treating them as integers would impose an artificial ordinal distance (e.g. implying course 9500 is 'greater' or 'closer' to 9501 than 9147). Using `OneHotEncoder(handle_unknown='ignore')` allows linear and tree models to evaluate each program or occupation independently without ordinal bias.* |
| **Q3: How do you prove zero data leakage in your preprocessing?** | *All imputers, standard scalers, and one-hot encoders are strictly housed within a scikit-learn `ColumnTransformer` inside the model `Pipeline`. The transformer is fit exclusively on `X_train` during 5-fold CV and final training. In unit test `test_preprocessing_never_fits_on_test_data`, we verify that transforming `X_test` leaves the scaler's internal `mean_` and `var_` vectors 100% mathematically identical to training values.* |
| **Q4: How did you handle hyperparameter tuning without test leakage?** | *Hyperparameter exploration via `GridSearchCV` utilized `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` strictly on the 80% training set. All 41 trial results were logged to `reports/tuning_log.csv`. The 20% holdout test set remained completely untouched until final scoring of the refitted champion pipeline.* |
| **Q5: How is the ML pipeline integrated into production backend and databases?** | *The trained champion model artifact (`models/model.joblib`) is served via a FastAPI REST API (`src/app.py`). Incoming student data is schema-validated before inference, and every prediction along with its risk tier and probability vector is logged into an SQLite database (`data/earlyalert.db`) via an SQLAlchemy data access layer (`src/database.py`). An academic advisor dashboard provides visual triage at `http://127.0.0.1:8000`.* |
| **Q6: What are the main limitations and ethical considerations of this model?** | *The dataset reflects students from a Portuguese polytechnic institute (2008-2019). Socio-economic factors like parents' qualification and occupation may encode historical systemic inequities. The model is designed strictly as an assistive early-warning triage tool for academic advisors to offer support—never as an automated gateway to deny financial aid or enrollment.* |

---

*Project developed and verified as a fully reproducible supervised ML project for III Year / V Sem B.Tech (CSBS) at Knowledge Institute of Technology.*
