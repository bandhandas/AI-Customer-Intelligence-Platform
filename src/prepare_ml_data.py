
from pathlib import Path
import pandas as pd

# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ml_ready_data.csv"

# 2. Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 3. Remove customerID because it is only an identifier
df = df.drop(columns=["customerID"])

# 4. Convert target variable into numeric values
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# 5. Convert categorical columns into numeric columns
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include=["object"]).columns,
    drop_first=True,
    dtype=int
)

# 6. Check the prepared dataset
print("\nPrepared dataset shape:", df.shape)

print("\nMissing values:", df.isnull().sum().sum())

print("\nTarget distribution:")
print(df["Churn"].value_counts())

# 7. Save machine-learning-ready dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nMachine learning dataset saved successfully!")
print("Saved at:", OUTPUT_PATH)
