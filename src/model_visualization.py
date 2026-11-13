
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"
LOGISTIC_PATH = BASE_DIR / "models" / "churn_logistic_model.pkl"
RF_PATH = BASE_DIR / "models" / "churn_random_forest.pkl"
REPORT_PATH = BASE_DIR / "reports"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

logistic_model = joblib.load(LOGISTIC_PATH)
rf_model = joblib.load(RF_PATH)

# 1. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_estimator(
    logistic_model, X_test, y_test,
    ax=axes[0], cmap="Blues"
)
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_estimator(
    rf_model, X_test, y_test,
    ax=axes[1], cmap="Greens"
)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.savefig(REPORT_PATH / "confusion_matrices.png", dpi=300)
plt.close()

# 2. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(
    logistic_model, X_test, y_test,
    ax=ax, name="Logistic Regression"
)

RocCurveDisplay.from_estimator(
    rf_model, X_test, y_test,
    ax=ax, name="Random Forest"
)

ax.set_title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig(REPORT_PATH / "roc_curve_comparison.png", dpi=300)
plt.close()

print("Visualizations saved successfully!")
print("1. reports/confusion_matrices.png")
print("2. reports/roc_curve_comparison.png")
