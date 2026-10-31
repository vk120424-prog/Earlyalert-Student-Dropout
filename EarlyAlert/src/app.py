"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/app.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    FastAPI backend and REST API service connected to SQLite database (data/earlyalert.db)
    and champion ML model (models/model.joblib).
    Provides endpoints for:
    - Real-time student dropout risk prediction (persisted to DB)
    - Querying student profiles and past prediction audit trails
    - System health and population analytics
    - Interactive EarlyAlert Academic Advisor Dashboard
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict_student, validate_student_record, ValidationError
from src.database import (
    init_db,
    seed_database,
    get_students,
    get_student_by_id,
    save_student_record,
    log_prediction,
    get_prediction_history,
    get_analytics_summary,
    DEFAULT_DB_PATH
)

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: initializes and seeds database on application startup."""
    init_db(DEFAULT_DB_PATH)
    seed_database(
        csv_path=os.path.join(PROJECT_ROOT, "data", "raw", "student_dropout.csv"),
        db_path=DEFAULT_DB_PATH,
        max_records=1000
    )
    logger.info("EarlyAlert FastAPI backend initialized and connected to database.")
    yield


# Initialize FastAPI application
app = FastAPI(
    title="EarlyAlert: Student Dropout Risk API",
    description="Backend ML inference and database service for higher education academic retention triage.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", summary="System Health & Status")
def health_check():
    """Returns system status, active model path, and database health."""
    model_path = os.path.join(PROJECT_ROOT, "models", "model.joblib")
    model_exists = os.path.exists(model_path)
    db_exists = os.path.exists(DEFAULT_DB_PATH)

    return {
        "status": "HEALTHY",
        "service": "EarlyAlert Academic Retention Backend",
        "author": "Vasanthkumar A (Knowledge Institute of Technology)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "database": {
            "connected": db_exists,
            "engine": "SQLite",
            "path": DEFAULT_DB_PATH
        },
        "model": {
            "loaded": model_exists,
            "architecture": "Random Forest Classifier (Tuned)",
            "primary_metric": "Macro-F1 (Holdout: 0.7056)",
            "artifact_path": model_path
        }
    }


@app.get("/api/analytics", summary="Student Population Analytics")
def get_analytics():
    """Returns database telemetry, dropout counts, and alert counts."""
    return get_analytics_summary(DEFAULT_DB_PATH)


@app.get("/api/students", summary="List Enrolled Students from Database")
def list_students(
    limit: int = Query(25, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    target: Optional[str] = Query(None, description="Filter by Target: Dropout, Enrolled, Graduate")
):
    """Retrieves paginated student profiles from the database."""
    students, total = get_students(limit=limit, offset=offset, target_filter=target, db_path=DEFAULT_DB_PATH)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(students),
        "students": students
    }


@app.get("/api/students/{student_id}", summary="Get Student by ID")
def get_student(student_id: str):
    """Retrieves a single student's complete feature record."""
    student = get_student_by_id(student_id, db_path=DEFAULT_DB_PATH)
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student ID '{student_id}' not found in database."
        )
    return student


@app.post("/api/predict", summary="Predict Dropout Risk for a Student Record")
def predict_risk(record: Dict[str, Any]):
    """
    Validates student record against all 36 features, runs through champion model pipeline,
    records prediction into database audit trail, and returns risk analysis.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "model.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model artifact not found. Please run 'python src/train.py --seed 42' first."
        )

    # 1. Run inference & validation
    try:
        prediction_result = predict_student(input_data=record, model_path=model_path)
    except ValidationError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}"
        )

    # 2. Persist student and prediction into database
    stu_id = record.get("student_id")
    if not stu_id:
        stu_id = save_student_record(record, db_path=DEFAULT_DB_PATH)
    else:
        # Check if student exists or create
        existing = get_student_by_id(stu_id, db_path=DEFAULT_DB_PATH)
        if not existing:
            save_student_record(record, db_path=DEFAULT_DB_PATH)

    pred_id = log_prediction(
        student_id=stu_id,
        prediction_result=prediction_result,
        model_version="v1.0-RandomForest",
        db_path=DEFAULT_DB_PATH
    )

    prediction_result["student_id"] = stu_id
    prediction_result["prediction_audit_id"] = pred_id
    return prediction_result


@app.get("/api/predictions", summary="View Prediction Audit Trail")
def list_predictions(limit: int = Query(25, ge=1, le=100)):
    """Retrieves recent prediction audit logs from database."""
    history = get_prediction_history(limit=limit, db_path=DEFAULT_DB_PATH)
    return {
        "count": len(history),
        "predictions": history
    }


@app.get("/api/metrics", summary="Holdout Model Benchmarks & Subgroup Diagnostics")
def get_model_metrics():
    """Returns holdout model comparison and fairness slice metrics."""
    comp_path = os.path.join(PROJECT_ROOT, "reports", "model_comparison.csv")
    sub_path = os.path.join(PROJECT_ROOT, "reports", "subgroup_analysis.csv")

    comparison = pd.read_csv(comp_path).to_dict(orient="records") if os.path.exists(comp_path) else []
    subgroups = pd.read_csv(sub_path).to_dict(orient="records") if os.path.exists(sub_path) else []

    return {
        "model_comparison": comparison,
        "subgroup_diagnostics": subgroups
    }


# ---------------------------------------------------------------------------
# Interactive Web Dashboard UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Academic Advisor Dashboard")
def serve_dashboard():
    """Serves an advisor dashboard for real-time triage and database exploration."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EarlyAlert - Student Dropout Risk Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 28, 45, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent-danger: #ef4444;
      --accent-warning: #f59e0b;
      --accent-success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: 'Plus Jakarta Sans', sans-serif;
      min-height: 100vh;
      padding: 2rem 1.5rem;
      background-image: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                        radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.08) 0px, transparent 50%);
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--card-border);
    }
    .brand h1 { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.025em; }
    .brand p { color: var(--text-muted); font-size: 0.875rem; margin-top: 0.25rem; }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2rem;
    }
    .stat-card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.25rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .stat-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }
    .stat-val { font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }
    @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 1rem;
      padding: 1.5rem;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
    }
    .card-title { font-size: 1.15rem; font-weight: 700; }
    .btn-group { display: flex; gap: 0.5rem; }
    button {
      cursor: pointer;
      font-family: inherit;
      font-weight: 600;
      border: none;
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      font-size: 0.85rem;
      transition: all 0.2s;
    }
    .btn-primary { background: var(--primary); color: white; }
    .btn-primary:hover { background: var(--primary-hover); transform: translateY(-1px); }
    .btn-outline { background: transparent; border: 1px solid var(--card-border); color: var(--text); }
    .btn-outline:hover { background: rgba(255,255,255,0.05); }
    .btn-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .btn-success { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    textarea {
      width: 100%;
      height: 320px;
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      color: #38bdf8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.825rem;
      padding: 0.75rem;
      resize: vertical;
    }
    .result-box {
      margin-top: 1rem;
      padding: 1.25rem;
      border-radius: 0.75rem;
      background: #0f172a;
      border: 1px solid var(--card-border);
      display: none;
    }
    .risk-badge {
      display: inline-block;
      padding: 0.35rem 0.85rem;
      border-radius: 0.5rem;
      font-weight: 800;
      font-size: 0.85rem;
      margin-bottom: 0.75rem;
    }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .prob-bar {
      margin-top: 0.5rem;
      display: flex;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(255,255,255,0.1);
    }
    .prob-seg { height: 100%; transition: width 0.4s ease; }
    .seg-dropout { background: #ef4444; }
    .seg-enrolled { background: #f59e0b; }
    .seg-graduate { background: #10b981; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 0.75rem 0.5rem; text-align: left; border-bottom: 1px solid var(--card-border); }
    th { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; font-weight: 700; }
    tr:hover { background: rgba(255,255,255,0.02); }
    .links a { color: #60a5fa; text-decoration: none; font-size: 0.85rem; margin-right: 1rem; }
    .links a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>EarlyAlert Advisor Portal</h1>
        <p>Predicting Student Dropout Risk | Author: Vasanthkumar A (KIOT CSBS)</p>
      </div>
      <div>
        <span class="badge" id="model-status">Model: Random Forest (70.6% Macro-F1)</span>
      </div>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-title">Database Records</div>
        <div class="stat-val" id="stat-students">1,000</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Holdout Macro-F1</div>
        <div class="stat-val" style="color: #60a5fa;">0.7056</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Dropout Recall</div>
        <div class="stat-val" style="color: #34d399;">72.89%</div>
      </div>
      <div class="stat-card">
        <div class="stat-title">Tuition Overdue Detection</div>
        <div class="stat-val" style="color: #f87171;">100.0%</div>
      </div>
    </div>

    <div class="main-grid">
      <!-- Live Inference Card -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Real-Time Risk Prediction</div>
          <div class="btn-group">
            <button class="btn-danger" onclick="loadHighRisk()">Load High Risk</button>
            <button class="btn-success" onclick="loadLowRisk()">Load Low Risk</button>
          </div>
        </div>
        <textarea id="json-input" spellcheck="false"></textarea>
        <div style="margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
          <button class="btn-primary" onclick="runPrediction()">Analyze Dropout Risk</button>
          <span id="predict-spinner" style="display:none; color: var(--text-muted); font-size: 0.8rem;">Evaluating model pipeline...</span>
        </div>

        <div id="result-box" class="result-box">
          <div id="risk-badge" class="risk-badge">HIGH RISK</div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-top: 0.5rem;">
            <span>Dropout: <strong id="prob-dropout">0%</strong></span>
            <span>Enrolled: <strong id="prob-enrolled">0%</strong></span>
            <span>Graduate: <strong id="prob-graduate">0%</strong></span>
          </div>
          <div class="prob-bar">
            <div id="bar-dropout" class="prob-seg seg-dropout" style="width: 0%;"></div>
            <div id="bar-enrolled" class="prob-seg seg-enrolled" style="width: 0%;"></div>
            <div id="bar-graduate" class="prob-seg seg-graduate" style="width: 0%;"></div>
          </div>
          <div id="risk-factors" style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted);"></div>
        </div>
      </div>

      <!-- Student Records Table Card -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">Enrolled Students Database</div>
          <button class="btn-outline" onclick="fetchStudents()">Refresh DB</button>
        </div>
        <div style="overflow-x: auto; max-height: 420px;">
          <table id="students-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Age</th>
                <th>Tuition</th>
                <th>Sem1 Grade</th>
                <th>Target</th>
              </tr>
            </thead>
            <tbody id="students-tbody">
              <tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Loading records...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <footer style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center;">
      <div class="links">
        <a href="/docs" target="_blank">Interactive Swagger Docs (/docs)</a>
        <a href="/api/health" target="_blank">Health Telemetry (/api/health)</a>
        <a href="/api/metrics" target="_blank">Model Benchmarks (/api/metrics)</a>
      </div>
      <div style="color: var(--text-muted); font-size: 0.8rem;">EarlyAlert v1.0 | Knowledge Institute of Technology</div>
    </footer>
  </div>

  <script>
    const sampleHighRisk = {
      "student_id": "STU-LIVE-HIGH-RISK",
      "Marital Status": 1,
      "Application mode": 17,
      "Application order": 2,
      "Course": 9147,
      "Daytime/evening attendance": 1,
      "Previous qualification": 1,
      "Previous qualification (grade)": 115.0,
      "Nacionality": 1,
      "Mother's qualification": 19,
      "Father's qualification": 19,
      "Mother's occupation": 5,
      "Father's occupation": 7,
      "Admission grade": 110.0,
      "Displaced": 1,
      "Educational special needs": 0,
      "Debtor": 1,
      "Tuition fees up to date": 0,
      "Gender": 1,
      "Scholarship holder": 0,
      "Age at enrollment": 29,
      "International": 0,
      "Curricular units 1st sem (credited)": 0,
      "Curricular units 1st sem (enrolled)": 6,
      "Curricular units 1st sem (evaluations)": 4,
      "Curricular units 1st sem (approved)": 1,
      "Curricular units 1st sem (grade)": 10.0,
      "Curricular units 1st sem (without evaluations)": 2,
      "Curricular units 2nd sem (credited)": 0,
      "Curricular units 2nd sem (enrolled)": 6,
      "Curricular units 2nd sem (evaluations)": 2,
      "Curricular units 2nd sem (approved)": 0,
      "Curricular units 2nd sem (grade)": 0.0,
      "Curricular units 2nd sem (without evaluations)": 4,
      "Unemployment rate": 12.7,
      "Inflation rate": 2.8,
      "GDP": -1.5
    };

    const sampleLowRisk = {
      "student_id": "STU-LIVE-LOW-RISK",
      "Marital Status": 1,
      "Application mode": 1,
      "Application order": 1,
      "Course": 9500,
      "Daytime/evening attendance": 1,
      "Previous qualification": 1,
      "Previous qualification (grade)": 160.0,
      "Nacionality": 1,
      "Mother's qualification": 1,
      "Father's qualification": 1,
      "Mother's occupation": 2,
      "Father's occupation": 2,
      "Admission grade": 155.0,
      "Displaced": 0,
      "Educational special needs": 0,
      "Debtor": 0,
      "Tuition fees up to date": 1,
      "Gender": 0,
      "Scholarship holder": 1,
      "Age at enrollment": 18,
      "International": 0,
      "Curricular units 1st sem (credited)": 0,
      "Curricular units 1st sem (enrolled)": 6,
      "Curricular units 1st sem (evaluations)": 8,
      "Curricular units 1st sem (approved)": 6,
      "Curricular units 1st sem (grade)": 15.5,
      "Curricular units 1st sem (without evaluations)": 0,
      "Curricular units 2nd sem (credited)": 0,
      "Curricular units 2nd sem (enrolled)": 6,
      "Curricular units 2nd sem (evaluations)": 8,
      "Curricular units 2nd sem (approved)": 6,
      "Curricular units 2nd sem (grade)": 16.2,
      "Curricular units 2nd sem (without evaluations)": 0,
      "Unemployment rate": 9.4,
      "Inflation rate": 0.5,
      "GDP": 2.1
    };

    function loadHighRisk() {
      document.getElementById('json-input').value = JSON.stringify(sampleHighRisk, null, 2);
    }

    function loadLowRisk() {
      document.getElementById('json-input').value = JSON.stringify(sampleLowRisk, null, 2);
    }

    async function runPrediction() {
      const inputVal = document.getElementById('json-input').value;
      const spinner = document.getElementById('predict-spinner');
      const resBox = document.getElementById('result-box');

      try {
        const payload = JSON.parse(inputVal);
        spinner.style.display = 'inline';

        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        spinner.style.display = 'none';

        if (!res.ok) {
          alert('Validation Error:\n' + data.detail);
          return;
        }

        resBox.style.display = 'block';
        const badge = document.getElementById('risk-badge');
        badge.innerText = data.dropout_risk_tier;
        badge.className = 'risk-badge ' + (data.probabilities.Dropout >= 0.4 ? 'risk-high' : 'risk-low');

        const dPct = (data.probabilities.Dropout * 100).toFixed(1);
        const ePct = (data.probabilities.Enrolled * 100).toFixed(1);
        const gPct = (data.probabilities.Graduate * 100).toFixed(1);

        document.getElementById('prob-dropout').innerText = dPct + '%';
        document.getElementById('prob-enrolled').innerText = ePct + '%';
        document.getElementById('prob-graduate').innerText = gPct + '%';

        document.getElementById('bar-dropout').style.width = dPct + '%';
        document.getElementById('bar-enrolled').style.width = ePct + '%';
        document.getElementById('bar-graduate').style.width = gPct + '%';

        let factorsHtml = '<strong>Primary Factors:</strong> ';
        factorsHtml += `Tuition Overdue: ${data.key_risk_factors.tuition_fees_overdue ? '<span style="color:#ef4444">YES</span>' : '<span style="color:#10b981">NO</span>'} | `;
        factorsHtml += `Debtor: ${data.key_risk_factors.debtor_status ? '<span style="color:#ef4444">YES</span>' : '<span style="color:#10b981">NO</span>'} | `;
        factorsHtml += `Sem1 Passed: ${data.key_risk_factors.sem1_approved_ratio}`;
        document.getElementById('risk-factors').innerHTML = factorsHtml;

      } catch (err) {
        spinner.style.display = 'none';
        alert('Invalid JSON input format:\n' + err.message);
      }
    }

    async function fetchStudents() {
      try {
        const res = await fetch('/api/students?limit=15');
        const data = await res.json();
        const tbody = document.getElementById('students-tbody');
        tbody.innerHTML = '';
        data.students.forEach(s => {
          const tr = document.createElement('tr');
          const targetColor = s.actual_target === 'Dropout' ? '#ef4444' : (s.actual_target === 'Graduate' ? '#10b981' : '#f59e0b');
          tr.innerHTML = `
            <td><code>${s.student_id}</code></td>
            <td>${s['Age at enrollment']}</td>
            <td>${s['Tuition fees up to date'] === 1 ? '<span style="color:#10b981">Current</span>' : '<span style="color:#ef4444">Overdue</span>'}</td>
            <td>${s['Curricular units 1st sem (grade)']}</td>
            <td><strong style="color:${targetColor}">${s.actual_target}</strong></td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error('Error fetching students:', e);
      }
    }

    // Init
    loadHighRisk();
    fetchStudents();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
