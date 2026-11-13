
from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_random_forest.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"

# Load trained model and training column names
model = joblib.load(MODEL_PATH)
training_data = pd.read_csv(DATA_PATH)
feature_columns = training_data.drop(columns=["Churn"]).columns

# Take one real customer record for a demonstration
customer = training_data.drop(columns=["Churn"]).iloc[[0]].copy()

# Predict churn probability
churn_probability = model.predict_proba(customer)[0][1]

# Assign risk category
if churn_probability >= 0.70:
    risk = "HIGH"
    action = "Contact customer immediately and offer a personalized retention plan."
elif churn_probability >= 0.40:
    risk = "MEDIUM"
    action = "Review customer concerns and offer suitable plan recommendations."
else:
    risk = "LOW"
    action = "Continue regular engagement and monitor customer activity."

print("\n========== CUSTOMER CHURN PREDICTION ==========")
print("Customer record: Demonstration customer")
print(f"Churn probability: {churn_probability:.2%}")
print("Risk category:", risk)
print("Recommended action:", action)
print("\nPrediction completed successfully!")
