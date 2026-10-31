# EarlyAlert: Data Dictionary & Data Audit Report
**Author:** Vasanthkumar A | **Dept:** Computer Science and Business Systems | **Knowledge Institute of Technology**  
**Dataset:** UCI "Predict Students' Dropout and Academic Success" (ID: 697, CC BY 4.0)

---

## 1. Executive Summary & Audit Statistics

| Audit Metric | Observed Value | Validation Status |
|---|---|---|
| **Total Rows (Before / After Cleaning)** | 4424 / 4424 | Complete (No rows dropped) |
| **Total Columns** | 37 (36 Features + 1 Target) | Valid Schema |
| **Duplicate Rows** | 0 | Passed (0 duplicates) |
| **Missing Values (Entire Dataset)** | 0 | Zero missing values in raw dataset |
| **Target Distribution** | Graduate: 2209 (49.93%)<br/>Dropout: 1421 (32.12%)<br/>Enrolled: 794 (17.95%) | Moderate Class Imbalance |

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
