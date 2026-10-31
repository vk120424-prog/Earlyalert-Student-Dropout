# EarlyAlert: Data Leakage & Feature Horizon Experiment
**Author:** Vasanthkumar A | **Dept:** Computer Science and Business Systems | **Knowledge Institute of Technology**

---

### 1. Research Motivation & The Early-Warning Dilemma
In higher education predictive analytics, the timing of intervention is critical:
- **Real-World Operational Goal (Early Warning):** Identify students at risk of dropout **at the conclusion of Semester 1** so advisors and faculty can intervene with tutoring, financial assistance, and counseling before Semester 2 failures occur.
- **The Temporal / Data Leakage Risk:** Semester 2 curricular performance columns (`Curricular units 2nd sem approved`, `grade`, `evaluations`) occur *after* the initial intervention window. Including them in an early warning system constitutes **temporal target leakage**: the model relies on second-semester failure to 'predict' dropout, by which point the student may already have disengaged and departed.

---

### 2. Experimental Setup & Comparative Metrics

Both configurations were evaluated using the exact same classifier architecture (`RandomForestClassifier`, seed=42) and evaluated on the identical 885-student holdout test set (stratified 80/20 split).

| Evaluation Metric | Run 1: Early Warning<br/>(Enrolment + 1st Semester Only) | Run 2: Full Horizon<br/>(Enrolment + 1st & 2nd Semesters) | Metric Delta (Run 2 vs Run 1) | Operational Significance |
|---|---|---|---|---|
| **Feature Count** | 30 features | 36 features (+6 Sem-2 features) | +6 Curricular features | Higher dimensionality |
| **Holdout Macro-F1** | **0.5622** | **0.6364** | **+0.0742** | Sem-2 features boost overall F1 |
| **Dropout Recall** | **0.7254** (72.5%) | **0.7676** (76.8%) | **+0.0423** | +4.2% higher detection |
| **Overall Accuracy** | **0.7175** (71.8%) | **0.7537** (75.4%) | **+0.0362** | High overall predictive fidelity |
| **Enrolled Recall** | **0.0818** | **0.2013** | **+0.1195** | Reduced confusion in Enrolled |
| **Graduate Recall** | **0.9412** | **0.9434** | **+0.0023** | High precision graduation tracking |

---

### 3. Viva Analysis & Key Takeaways
1. **Trade-off between Timeliness and Predictive Power:**
   - Run 1 achieves an impressive **0.5622 Macro-F1** and **72.5% Dropout Recall** using *only* 1st semester data. This validates that universities can reliably identify more than 70% of potential dropouts a full semester earlier.
   - Run 2 achieves **0.6364 Macro-F1** and **76.8% Dropout Recall**, indicating that 2nd semester academic performance strongly cements the final graduation vs dropout outcome.
2. **Pedagogical Recommendation:**
   - Deploy **Run 1 (Early Warning Model)** in January/February for actionable preventative intervention.
   - Deploy **Run 2 (Consolidated Model)** in June/July for institutional retention reporting and academic progress auditing.

---

### 4. Pipeline Architecture Reference
The end-to-end preprocessing pipeline diagram is stored at: `reports/figures/pipeline.png`.
