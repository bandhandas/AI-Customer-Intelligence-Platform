from pathlib import Path
import pandas as pd

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Cleaned dataset path
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"

# 3. Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("\n========== FINAL DATA VALIDATION ==========")

# 4. Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# 5. Missing values
print("\n2. Total Missing Values:")
print(df.isnull().sum().sum())

# 6. Duplicate records
print("\n3. Duplicate Records:")
print(df.duplicated().sum())

# 7. TotalCharges datatype
print("\n4. TotalCharges Data Type:")
print(df["TotalCharges"].dtype)

# 8. Check zero tenure customers
print("\n5. Zero Tenure Customers:")
print((df["tenure"] == 0).sum())

# 9. Check TotalCharges for zero tenure
print("\n6. TotalCharges for Zero Tenure:")
print(df.loc[df["tenure"] == 0, "TotalCharges"].value_counts())

# 10. Final result
print("\n========== VALIDATION COMPLETED ==========")