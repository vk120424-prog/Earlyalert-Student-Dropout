# EarlyAlert: Holdout Error Analysis (FR-05)
**Author:** Vasanthkumar A | **Dept:** Computer Science and Business Systems | **Knowledge Institute of Technology**

---

### Executive Overview of Model Misclassifications
In academic early-warning systems, analyzing false predictions reveals the boundary conditions of machine learning models.
Below are detailed case investigations of misclassified students from the untouched holdout test set.


### Case Study 1: Student Index #36
- **True Status:** `Dropout`
- **Model Prediction:** `Graduate`
- **Predicted Class Probabilities:** **Dropout**: 7.0%, **Enrolled**: 46.0%, **Graduate**: 47.0%
- **Key Feature Signatures:**
  - **Admission Grade:** 108.2
  - **Age at Enrollment:** 20.0
  - **Tuition Fees Up to Date:** Yes (1)
  - **Debtor Status:** No (0)
  - **Scholarship Holder:** No (0)
  - **1st Sem Approved / Enrolled:** 6.0 / 7.0
  - **1st Sem Grade:** 13.26666667
  - **2nd Sem Approved / Enrolled:** 5.0 / 7.0
  - **2nd Sem Grade:** 12.3
- **Pedagogical Diagnostic:**
  **False Negative (High Risk):** The student demonstrated moderate or passing initial grades in Curricular units (1st sem grade: 13.26666667), leading the model to project graduation. However, external non-academic factors (e.g. financial debtor=0.0, tuition up to date=1.0, or age=20.0) precipitated premature discontinuation.

---

### Case Study 2: Student Index #5
- **True Status:** `Graduate`
- **Model Prediction:** `Dropout`
- **Predicted Class Probabilities:** **Dropout**: 36.0%, **Enrolled**: 33.0%, **Graduate**: 31.0%
- **Key Feature Signatures:**
  - **Admission Grade:** 140.0
  - **Age at Enrollment:** 27.0
  - **Tuition Fees Up to Date:** Yes (1)
  - **Debtor Status:** No (0)
  - **Scholarship Holder:** No (0)
  - **1st Sem Approved / Enrolled:** 4.0 / 5.0
  - **1st Sem Grade:** 13.5
  - **2nd Sem Approved / Enrolled:** 4.0 / 5.0
  - **2nd Sem Grade:** 13.75
- **Pedagogical Diagnostic:**
  **False Positive:** The student faced initial academic hurdles (e.g. low 1st sem approved courses: 4.0 out of 5.0), triggering the model's high-sensitivity dropout alert. Nonetheless, the student demonstrated academic resilience, recovered in subsequent terms, and graduated.

---

### Case Study 3: Student Index #2
- **True Status:** `Enrolled`
- **Model Prediction:** `Dropout`
- **Predicted Class Probabilities:** **Dropout**: 57.0%, **Enrolled**: 30.0%, **Graduate**: 13.0%
- **Key Feature Signatures:**
  - **Admission Grade:** 120.7
  - **Age at Enrollment:** 18.0
  - **Tuition Fees Up to Date:** Yes (1)
  - **Debtor Status:** No (0)
  - **Scholarship Holder:** No (0)
  - **1st Sem Approved / Enrolled:** 4.0 / 6.0
  - **1st Sem Grade:** 10.75
  - **2nd Sem Approved / Enrolled:** 0.0 / 6.0
  - **2nd Sem Grade:** 0.0
- **Pedagogical Diagnostic:**
  **Boundary Ambiguity (Enrolled vs Dropout):** The student exhibits overlapping characteristics between borderline enrollment and discontinuation, highlighting the subtle decision boundary between continuous enrollment and eventual dropout.

---
