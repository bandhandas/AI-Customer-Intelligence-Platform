from pathlib import Path
import pandas as pd

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Load cleaned dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"

df = pd.read_csv(DATA_PATH)

# 3. Create customer segments
df["CustomerSegment"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "New",
        "Growing",
        "Established",
        "Loyal"
    ]
)

# 4. Customer count by segment
print("\n========== CUSTOMER SEGMENTATION ==========")

print("\n1. Customer Count by Segment:")
print(df["CustomerSegment"].value_counts().sort_index())

# 5. Churn percentage by segment
print("\n2. Churn Rate by Segment:")

churn_analysis = pd.crosstab(
    df["CustomerSegment"],
    df["Churn"],
    normalize="index"
) * 100

print(churn_analysis)

# 6. Average monthly charges
print("\n3. Average Monthly Charges by Segment:")

print(
    df.groupby("CustomerSegment", observed=True)["MonthlyCharges"].mean()
)

# 7. Save segmented dataset
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_segments.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nSegmented dataset saved successfully!")

print("\n========== SEGMENTATION COMPLETED ==========")