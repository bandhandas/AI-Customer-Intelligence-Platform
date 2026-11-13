from pathlib import Path
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Display dataset information
print("\n========== DATASET LOADED SUCCESSFULLY ==========")

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. First 5 Customer Records:")
print(df.head())

print("\n3. Dataset Information:")
df.info()

print("\n4. Column Names:")
print(df.columns.tolist())
# 5. Data Quality Audit

print("\n========== DATA QUALITY AUDIT ==========")

# Check missing values
print("\n1. Missing Values:")
print(df.isnull().sum())

# Check duplicate customer records
print("\n2. Duplicate Records:")
print(df.duplicated().sum())

# Check blank TotalCharges
print("\n3. Blank TotalCharges:")
print(df["TotalCharges"].str.strip().eq("").sum())

# Check customer churn distribution
print("\n4. Churn Distribution:")
print(df["Churn"].value_counts())

# Check churn percentage
print("\n5. Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
# 6. Investigate Blank TotalCharges

print("\n========== INVESTIGATING BLANK RECORDS ==========")

blank_records = df[
    df["TotalCharges"].str.strip() == ""
]

print("\n1. Number of Blank Records:")
print(len(blank_records))

print("\n2. Customer IDs and Tenure:")
print(
    blank_records[
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\n3. Tenure Values of Blank Records:")
print(blank_records["tenure"].value_counts())