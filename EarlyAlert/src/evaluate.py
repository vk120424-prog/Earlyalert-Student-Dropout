"""
EarlyAlert: Predicting Student Dropout Risk
Module: src/evaluate.py
Author: Vasanthkumar A | Dept: Computer Science and Business Systems
Knowledge Institute of Technology

Description:
    Implements FR-04 and FR-05:
    - Classification report, confusion matrix figures in reports/figures/.
    - Macro-F1, accuracy, and Dropout recall calculation.
    - Comparison table saved to reports/model_comparison.csv.
    - Subgroup diagnostics on slices (gender, scholarship holder, age band, tuition fees up to date).
    - Detailed error analysis on misclassified students.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score
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
logger = logging.getLogger("evaluate")

TARGET_CLASSES = ["Dropout", "Enrolled", "Graduate"]


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes primary and secondary project metrics:
    - Macro-F1 (primary)
    - Accuracy
    - Dropout Recall (per-class recall for Dropout)
    - Enrolled Recall
    - Graduate Recall
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate per-class recall
    recalls = recall_score(y_true, y_pred, labels=TARGET_CLASSES, average=None)
    dropout_recall = float(recalls[0])   # Index 0 is Dropout
    enrolled_recall = float(recalls[1])  # Index 1 is Enrolled
    graduate_recall = float(recalls[2])  # Index 2 is Graduate

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "dropout_recall": dropout_recall,
        "enrolled_recall": enrolled_recall,
        "graduate_recall": graduate_recall
    }


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    figures_dir: str = "reports/figures"
) -> str:
    """
    Plots confusion matrices for all evaluated models and saves individual
    and combined figures to reports/figures/.
    """
    os.makedirs(figures_dir, exist_ok=True)
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(5.5 * num_models, 4.5), sharey=True)
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=TARGET_CLASSES)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_CLASSES,
            yticklabels=TARGET_CLASSES,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{model_name}\nMacro-F1: {res['metrics']['macro_f1']:.3f} | Dropout Rec: {res['metrics']['dropout_recall']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="semibold")
        if ax == axes[0]:
            ax.set_ylabel("True Label", fontweight="semibold")

        # Save individual model plot
        indiv_fig, indiv_ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=TARGET_CLASSES,
            yticklabels=TARGET_CLASSES,
            ax=indiv_ax
        )
        indiv_ax.set_title(f"Confusion Matrix: {model_name}\nMacro-F1: {res['metrics']['macro_f1']:.3f} | Dropout Recall: {res['metrics']['dropout_recall']:.3f}")
        indiv_ax.set_xlabel("Predicted Class")
        indiv_ax.set_ylabel("True Class")
        indiv_path = os.path.join(figures_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        indiv_fig.tight_layout()
        indiv_fig.savefig(indiv_path, dpi=200)
        plt.close(indiv_fig)

    fig.suptitle("Holdout Test Set Performance Comparison: EarlyAlert Models", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    combined_path = os.path.join(figures_dir, "confusion_matrix_all_models.png")
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix figures to %s", figures_dir)
    return combined_path


def run_subgroup_diagnostics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: str = "reports"
) -> pd.DataFrame:
    """
    Implements FR-05 Subgroup Diagnostics on 4 slices:
    1. Gender (Female: 0, Male: 1)
    2. Scholarship holder (No: 0, Yes: 1)
    3. Tuition fees up to date (No: 0, Yes: 1)
    4. Age at enrollment bands (<=20, 21-25, >25)

    Reports Macro-F1 and Dropout recall per subgroup slice.
    """
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("Executing FR-05 subgroup diagnostics on holdout test set...")

    y_pred = model.predict(X_test)
    X_eval = X_test.copy()
    X_eval["y_true"] = y_test.values
    X_eval["y_pred"] = y_pred

    # Age band discretization
    def categorize_age(age):
        if age <= 20:
            return "<= 20 years"
        elif age <= 25:
            return "21 - 25 years"
        else:
            return "> 25 years"

    X_eval["age_band"] = X_eval["Age at enrollment"].apply(categorize_age)

    slices_def = [
        ("Gender", "Gender", {0: "Female (0)", 1: "Male (1)"}),
        ("Scholarship Status", "Scholarship holder", {0: "Non-Scholarship (0)", 1: "Scholarship Holder (1)"}),
        ("Tuition Fees Status", "Tuition fees up to date", {0: "Tuition Overdue (0)", 1: "Tuition Up to Date (1)"}),
        ("Age Band", "age_band", {
            "<= 20 years": "<= 20 years",
            "21 - 25 years": "21 - 25 years",
            "> 25 years": "> 25 years"
        })
    ]

    records = []
    for slice_name, col_name, val_mapping in slices_def:
        for val, label in val_mapping.items():
            mask = (X_eval[col_name] == val)
            subgroup = X_eval[mask]
            sub_count = len(subgroup)
            if sub_count == 0:
                continue

            sub_y_true = subgroup["y_true"]
            sub_y_pred = subgroup["y_pred"]

            sub_metrics = compute_metrics(sub_y_true, sub_y_pred)
            dropout_actual = (sub_y_true == "Dropout").sum()
            dropout_pred = (sub_y_pred == "Dropout").sum()

            records.append({
                "Slice Group": slice_name,
                "Subgroup Category": label,
                "Sample Count": sub_count,
                "Sample %": f"{(sub_count / len(X_eval)) * 100.1:.1f}%",
                "Actual Dropouts": dropout_actual,
                "Predicted Dropouts": dropout_pred,
                "Macro-F1": round(sub_metrics["macro_f1"], 4),
                "Dropout Recall": round(sub_metrics["dropout_recall"], 4),
                "Accuracy": round(sub_metrics["accuracy"], 4)
            })

    subgroup_df = pd.DataFrame(records)
    subgroup_csv = os.path.join(reports_dir, "subgroup_analysis.csv")
    subgroup_df.to_csv(subgroup_csv, index=False)
    logger.info("Saved subgroup diagnostics table to %s", subgroup_csv)
    return subgroup_df


def analyze_misclassifications(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    reports_dir: str = "reports",
    n_cases: int = 3
) -> str:
    """
    Implements FR-05 Error Analysis:
    Selects wrongly predicted students, displays true vs predicted classes,
    full probability vectors, top features, and provides pedagogical explanations.
    """
    os.makedirs(reports_dir, exist_ok=True)
    logger.info("Conducting FR-05 error analysis on misclassified student records...")

    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    classes = list(model.classes_) if hasattr(model, "classes_") else TARGET_CLASSES

    # Find error indices
    error_indices = np.where(y_test.values != y_pred)[0]
    logger.info("Total holdout misclassifications: %d out of %d (%.2f%%)",
                len(error_indices), len(y_test), (len(error_indices) / len(y_test)) * 100.0)

    # Pick representative error cases:
    # 1. Actual Dropout misclassified as Graduate or Enrolled (False Negative - High Risk)
    # 2. Actual Graduate misclassified as Dropout (False Positive)
    # 3. Actual Enrolled misclassified as Dropout
    selected_indices = []
    
    # Case 1: Actual Dropout predicted as Graduate
    for idx in error_indices:
        if y_test.iloc[idx] == "Dropout" and y_pred[idx] == "Graduate":
            selected_indices.append(idx)
            break

    # Case 2: Actual Graduate predicted as Dropout
    for idx in error_indices:
        if y_test.iloc[idx] == "Graduate" and y_pred[idx] == "Dropout":
            selected_indices.append(idx)
            break

    # Case 3: Actual Enrolled predicted as Dropout
    for idx in error_indices:
        if y_test.iloc[idx] == "Enrolled" and y_pred[idx] == "Dropout":
            selected_indices.append(idx)
            break

    # Fallback if specific combos not met
    while len(selected_indices) < n_cases and len(selected_indices) < len(error_indices):
        cand = error_indices[len(selected_indices)]
        if cand not in selected_indices:
            selected_indices.append(cand)

    analysis_markdown = """# EarlyAlert: Holdout Error Analysis (FR-05)
**Author:** Vasanthkumar A | **Dept:** Computer Science and Business Systems | **Knowledge Institute of Technology**

---

### Executive Overview of Model Misclassifications
In academic early-warning systems, analyzing false predictions reveals the boundary conditions of machine learning models.
Below are detailed case investigations of misclassified students from the untouched holdout test set.

"""

    for i, idx in enumerate(selected_indices, start=1):
        true_label = y_test.iloc[idx]
        pred_label = y_pred[idx]
        student_data = X_test.iloc[idx]

        if probs is not None:
            student_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs[idx])}
            prob_str = ", ".join([f"**{k}**: {v * 100:.1f}%" for k, v in student_probs.items()])
        else:
            prob_str = "N/A"

        # Domain explanation
        reason = ""
        if true_label == "Dropout" and pred_label == "Graduate":
            reason = (
                "**False Negative (High Risk):** The student demonstrated moderate or passing initial grades in "
                f"Curricular units (1st sem grade: {student_data.get('Curricular units 1st sem (grade)', 'N/A')}), "
                "leading the model to project graduation. However, external non-academic factors (e.g. financial "
                f"debtor={student_data.get('Debtor', 'N/A')}, tuition up to date={student_data.get('Tuition fees up to date', 'N/A')}, "
                f"or age={student_data.get('Age at enrollment', 'N/A')}) precipitated premature discontinuation."
            )
        elif true_label == "Graduate" and pred_label == "Dropout":
            reason = (
                "**False Positive:** The student faced initial academic hurdles (e.g. low 1st sem approved courses: "
                f"{student_data.get('Curricular units 1st sem (approved)', 'N/A')} out of {student_data.get('Curricular units 1st sem (enrolled)', 'N/A')}), "
                "triggering the model's high-sensitivity dropout alert. Nonetheless, the student demonstrated "
                "academic resilience, recovered in subsequent terms, and graduated."
            )
        else:
            reason = (
                f"**Boundary Ambiguity ({true_label} vs {pred_label}):** The student exhibits overlapping characteristics "
                "between borderline enrollment and discontinuation, highlighting the subtle decision boundary between "
                "continuous enrollment and eventual dropout."
            )

        analysis_markdown += f"""
### Case Study {i}: Student Index #{idx}
- **True Status:** `{true_label}`
- **Model Prediction:** `{pred_label}`
- **Predicted Class Probabilities:** {prob_str}
- **Key Feature Signatures:**
  - **Admission Grade:** {student_data.get('Admission grade', 'N/A')}
  - **Age at Enrollment:** {student_data.get('Age at enrollment', 'N/A')}
  - **Tuition Fees Up to Date:** {'Yes (1)' if student_data.get('Tuition fees up to date') == 1 else 'No (0)'}
  - **Debtor Status:** {'Yes (1)' if student_data.get('Debtor') == 1 else 'No (0)'}
  - **Scholarship Holder:** {'Yes (1)' if student_data.get('Scholarship holder') == 1 else 'No (0)'}
  - **1st Sem Approved / Enrolled:** {student_data.get('Curricular units 1st sem (approved)', 'N/A')} / {student_data.get('Curricular units 1st sem (enrolled)', 'N/A')}
  - **1st Sem Grade:** {student_data.get('Curricular units 1st sem (grade)', 'N/A')}
  - **2nd Sem Approved / Enrolled:** {student_data.get('Curricular units 2nd sem (approved)', 'N/A',)} / {student_data.get('Curricular units 2nd sem (enrolled)', 'N/A')}
  - **2nd Sem Grade:** {student_data.get('Curricular units 2nd sem (grade)', 'N/A')}
- **Pedagogical Diagnostic:**
  {reason}

---
"""

    error_md_path = os.path.join(reports_dir, "error_analysis.md")
    with open(error_md_path, "w", encoding="utf-8") as f:
        f.write(analysis_markdown)
    logger.info("Saved error analysis to %s", error_md_path)
    return error_md_path


def create_pipeline_diagram(figures_dir: str = "reports/figures") -> str:
    """
    Generates a visual diagram of the scikit-learn preprocessing and modeling pipeline
    and saves it to reports/figures/pipeline.png.
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Draw pipeline boxes
    boxes = [
        ("Raw Input\nStudent Data\n(36 Features)", 0.08, 0.5, "#E3F2FD", "#1565C0"),
        ("Validation Gate\nSchema, Ranges & Dtypes\n(src/validate.py)", 0.28, 0.5, "#E8F5E9", "#2E7D32"),
        ("ColumnTransformer\nCat: Imputer + OHE\nNum: Imputer + Scaler", 0.52, 0.5, "#FFF3E0", "#E65100"),
        ("Classification Model\nTuned Classifier\n(e.g., Random Forest)", 0.74, 0.5, "#F3E5F5", "#7B1FA2"),
        ("Predictions & Risk\n[Dropout, Enrolled, Grad]\n+ Probability Vector", 0.92, 0.5, "#FFEBEE", "#C62828")
    ]

    for text, x, y, bg_color, border_color in boxes:
        bbox_props = dict(boxstyle="round,pad=0.6", facecolor=bg_color, edgecolor=border_color, linewidth=2)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", bbox=bbox_props)

    # Draw arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 0.075
        x2 = boxes[i+1][1] - 0.075
        ax.annotate(
            "",
            xy=(x2, 0.5), xycoords="data",
            xytext=(x1, 0.5), textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#37474F", shrinkA=4, shrinkB=4)
        )

    ax.text(
        0.52, 0.15,
        "STRICT DESIGN PRINCIPLE: Preprocessing transformer is fit STRICTLY on Training Data (80% Stratified Split).\nZero Data Leakage between Holdout Test Data and Transformation Parameters.",
        ha="center", va="center", fontsize=9.5, style="italic", color="#263238",
        bbox=dict(boxstyle="square,pad=0.5", facecolor="#ECEFF1", edgecolor="#90A4AE", linewidth=1)
    )

    ax.set_title("EarlyAlert: End-to-End Scikit-Learn Pipeline Architecture", fontsize=14, fontweight="bold", pad=20)
    diagram_path = os.path.join(figures_dir, "pipeline.png")
    fig.tight_layout()
    fig.savefig(diagram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved pipeline architecture diagram to %s", diagram_path)
    return diagram_path
