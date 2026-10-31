"""
EarlyAlert: Predicting Student Dropout Risk
Module: scripts/client_query.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Demonstration and diagnostic client for EarlyAlert:
    1. Connects to the live FastAPI backend service (http://127.0.0.1:8000).
    2. Queries /api/health and /api/analytics telemetry.
    3. Fetches student records from the connected SQLite database.
    4. Submits a real-time student profile for model inference and persists audit logs.
    5. Directly queries the SQLite database (data/earlyalert.db) to demonstrate dual connectivity.
"""

import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

API_BASE_URL = os.environ.get("EARLYALERT_API_URL", "http://127.0.0.1:8000")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "earlyalert.db")


def make_request(endpoint: str, method: str = "GET", payload: dict = None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as err:
        return {"error": f"Failed to connect to backend at {url}: {err}"}


def demonstrate_backend_connection():
    print("=" * 70)
    print(" 1. CONNECTING TO FASTAPI BACKEND SERVICE")
    print(f" Target URL: {API_BASE_URL}")
    print("=" * 70)

    # 1. Health Check
    health = make_request("/api/health")
    if "error" in health:
        print(f"[!] Backend Connection Error: {health['error']}")
        print("    Ensure the backend is running via: python -m uvicorn src.app:app --port 8000")
        return
    print(f"[+] Health Status:    {health.get('status')}")
    print(f"[+] Database Engine:  {health.get('database', {}).get('engine')} (Connected: {health.get('database', {}).get('connected')})")
    print(f"[+] Model Loaded:     {health.get('model', {}).get('architecture')}")
    print(f"[+] Holdout Metric:   {health.get('model', {}).get('primary_metric')}")

    # 2. Analytics
    print("\n--- Telemetry & Population Analytics ---")
    analytics = make_request("/api/analytics")
    print(f"[+] Total Enrolled Students: {analytics.get('total_students_enrolled')}")
    print(f"[+] Predictions Logged:      {analytics.get('total_predictions_logged')}")
    print(f"[+] Target Distribution:     {analytics.get('historical_target_distribution')}")

    # 3. Query Students from API
    print("\n--- Querying Students via API (/api/students) ---")
    students_resp = make_request("/api/students?limit=3")
    for s in students_resp.get("students", []):
        print(f"  • Student: {s['student_id']} | Age: {s['Age at enrollment']} | Tuition OK: {s['Tuition fees up to date']} | Sem1 Grade: {s['Curricular units 1st sem (grade)']} | Outcome: {s['actual_target']}")

    # 4. Run Live Prediction
    print("\n--- Submitting Student for Real-Time Risk Analysis (/api/predict) ---")
    sample_file = os.path.join(PROJECT_ROOT, "data", "sample_students", "high_risk_student.json")
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_payload = json.load(f)
        pred = make_request("/api/predict", method="POST", payload=sample_payload)
        print(f"[+] Predicted Class:   {pred.get('predicted_class')}")
        print(f"[+] Dropout Risk Tier: {pred.get('dropout_risk_tier')}")
        print(f"[+] Probabilities:     {pred.get('probabilities')}")
        print(f"[+] DB Audit ID:       {pred.get('prediction_audit_id')}")


def demonstrate_database_connection():
    print("\n" + "=" * 70)
    print(" 2. DIRECT SQLITE DATABASE CONNECTION")
    print(f" Database Path: {DB_PATH}")
    print("=" * 70)

    if not os.path.exists(DB_PATH):
        print(f"[!] Database file does not exist at {DB_PATH}.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query Table Names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    print(f"[+] Database Tables Found: {tables}")

    # Count rows in each table
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  • Table '{table}': {count} records")

    # Sample predictions audit
    print("\n--- Recent Prediction Audit Entries from DB ---")
    cursor.execute("SELECT id, student_id, predicted_class, dropout_probability, dropout_risk_tier, created_at FROM predictions ORDER BY id DESC LIMIT 3;")
    rows = cursor.fetchall()
    for r in rows:
        print(f"  [Audit #{r[0]}] Student: {r[1]} | Pred: {r[2]} | Dropout Prob: {r[3]*100:.1f}% | Risk: {r[4]}")

    conn.close()
    print("\n[+] Direct database connection verified and closed successfully.")


if __name__ == "__main__":
    demonstrate_backend_connection()
    demonstrate_database_connection()
