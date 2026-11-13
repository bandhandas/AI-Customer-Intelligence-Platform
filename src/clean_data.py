from pathlib import Path
import pandas as pd

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Dataset paths
RAW_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"

# 3. Load original dataset
df = pd.read_csv(RAW_PATH)

print("Original dataset loaded:", df.shape)

# 4. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# 5. Handle blank values
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 6. Check data types
print("\nUpdated Data Types:")
print(df.dtypes)

# 7. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 8. Check duplicates
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 9. Save cleaned dataset
df.to_csv(PROCESSED_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("Final dataset shape:", df.shape)