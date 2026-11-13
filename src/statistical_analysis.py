from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

# 1. Load cleaned dataset
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== STATISTICAL ANALYSIS ==========")

# 2. Chi-square test function
def perform_chi_square(column):

    table = pd.crosstab(df[column], df["Churn"])

    chi2, p_value, dof, expected = chi2_contingency(table)

    print(f"\n{column} vs Churn")
    print("Chi-square statistic:", chi2)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("Result: Statistically significant association")
    else:
        print("Result: No statistically significant association")

# 3. Test categorical variables
perform_chi_square("Contract")
perform_chi_square("PaymentMethod")
perform_chi_square("InternetService")

# 4. Compare monthly charges
stayed = df[df["Churn"] == "No"]["MonthlyCharges"]

churned = df[df["Churn"] == "Yes"]["MonthlyCharges"]

t_stat, p_value = ttest_ind(churned, stayed)

print("\nMonthly Charges vs Churn")
print("Average charges of churned customers:", churned.mean())
print("Average charges of retained customers:", stayed.mean())
print("T-statistic:", t_stat)
print("P-value:", p_value)

print("\n========== ANALYSIS COMPLETED ==========")