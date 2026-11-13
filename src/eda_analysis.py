from pathlib import Path
import pandas as pd

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Load cleaned dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 3. Basic statistical summary
print("\n1. Statistical Summary:")
print(df.describe())

# 4. Customer churn count
print("\n2. Customer Churn Count:")
print(df["Churn"].value_counts())

# 5. Churn percentage
print("\n3. Customer Churn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# 6. Churn by contract type
print("\n4. Churn by Contract Type:")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100)

# 7. Churn by payment method
print("\n5. Churn by Payment Method:")
print(pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100)

# 8. Churn by internet service
print("\n6. Churn by Internet Service:")
print(pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100)

print("\n========== EDA COMPLETED ==========")
# 7. Customer Tenure Analysis

print("\n========== CUSTOMER TENURE ANALYSIS ==========")

# Create customer tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]
)

# Calculate churn percentage for each group
tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\n1. Churn Rate by Tenure Group:")
print(tenure_churn)

# Calculate average monthly charges
print("\n2. Average Monthly Charges by Churn:")
print(df.groupby("Churn", observed=True)["MonthlyCharges"].mean())

# Calculate average tenure
print("\n3. Average Tenure by Churn:")
print(df.groupby("Churn", observed=True)["tenure"].mean())

print("\n========== TENURE ANALYSIS COMPLETED ==========")