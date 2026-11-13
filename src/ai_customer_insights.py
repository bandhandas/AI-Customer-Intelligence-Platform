
from pathlib import Path
import pandas as pd
import requests

# -----------------------------------------
# 1. Project paths and Ollama configuration
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"
REPORT_PATH = BASE_DIR / "reports" / "ai_business_insights.md"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# -----------------------------------------
# 2. Load customer data
# -----------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "Churn",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "tenure",
    "MonthlyCharges"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing dataset columns: {missing_columns}")

total_customers = len(df)
churned = (df["Churn"] == "Yes").sum()
retained = (df["Churn"] == "No").sum()

churn_rate = churned / total_customers * 100

print("Customer dataset loaded successfully.")
print(f"Total customers: {total_customers}")
print(f"Overall churn rate: {churn_rate:.2f}%")

# -----------------------------------------
# 3. Calculate actual churn rates
# -----------------------------------------

def calculate_churn_rate(column):
    """
    Calculate percentage of customers who churned
    within each category.
    """

    return (
        df.groupby(column, observed=False)["Churn"]
        .apply(lambda group: (group == "Yes").mean() * 100)
        .round(2)
    )


contract_churn = calculate_churn_rate("Contract")
payment_churn = calculate_churn_rate("PaymentMethod")
internet_churn = calculate_churn_rate("InternetService")

# -----------------------------------------
# 4. Customer tenure segmentation
# -----------------------------------------

tenure_groups = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

tenure_churn = (
    df.assign(TenureGroup=tenure_groups)
    .groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda group: (group == "Yes").mean() * 100)
    .round(2)
)

# -----------------------------------------
# 5. Calculate average customer metrics
# -----------------------------------------

average_monthly_charges = df.groupby("Churn")["MonthlyCharges"].mean()
average_tenure = df.groupby("Churn")["tenure"].mean()

# -----------------------------------------
# 6. Prepare verified business context
# -----------------------------------------

business_context = f"""
TELECOM CUSTOMER CHURN ANALYSIS

IMPORTANT DEFINITION:
All segment percentages below represent CHURN RATES.
Churn rate = Number of churned customers / Total customers in that segment * 100.

These are NOT retention rates.

OVERALL BUSINESS METRICS:

- Total customers: {total_customers}
- Churned customers: {churned}
- Retained customers: {retained}
- Overall churn rate: {churn_rate:.2f}%

AVERAGE MONTHLY CHARGES:

- Churned customers: ${average_monthly_charges['Yes']:.2f}
- Retained customers: ${average_monthly_charges['No']:.2f}

AVERAGE TENURE:

- Churned customers: {average_tenure['Yes']:.2f} months
- Retained customers: {average_tenure['No']:.2f} months

CHURN RATE BY CONTRACT TYPE:

{contract_churn.to_string()}

CHURN RATE BY PAYMENT METHOD:

{payment_churn.to_string()}

CHURN RATE BY INTERNET SERVICE:

{internet_churn.to_string()}

CHURN RATE BY TENURE GROUP:

{tenure_churn.to_string()}
"""

# -----------------------------------------
# 7. Generate AI business insights using Ollama
# -----------------------------------------

print("\nGenerating AI-powered customer insights using Ollama...")
print("Model:", MODEL_NAME)

prompt = f"""
You are a senior telecom business analyst.

Analyze the following verified telecom customer churn statistics.

{business_context}

Prepare a professional business intelligence report in Markdown.

Use these exact sections:

1. Executive Summary
2. Main Churn Findings
3. High-Risk Customer Segments
4. Possible Business Reasons
5. Customer Retention Recommendations
6. Business Impact and Next Steps

STRICT ANALYSIS RULES:

1. Every segment percentage supplied is an actual CHURN RATE.
2. Never interpret a churn rate as a retention rate.
3. A higher churn rate means more customers are leaving.
4. A lower churn rate means fewer customers are leaving.
5. The highest churn rate in a category indicates the highest-risk segment.
6. Use the supplied percentages exactly. Do not invent or reverse them.
7. Compare the segments accurately.
8. Clearly distinguish observed facts from possible explanations.
9. Do not claim correlation proves causation.
10. Give practical and measurable retention recommendations.
11. Do not invent financial savings or revenue figures.
12. Use professional, clear business English.

Important examples:
- Month-to-month contract churn rate is 42.71%.
- One-year contract churn rate is 11.27%.
- Two-year contract churn rate is 2.83%.
- Electronic check churn rate is 45.29%.
- Fiber optic churn rate is 41.89%.

Do not contradict these figures.
"""

payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1
    }
}

try:
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    ai_report = response.json().get("response", "").strip()

    if not ai_report:
        raise RuntimeError("Ollama returned an empty report.")

except requests.exceptions.RequestException as error:
    raise RuntimeError(
        "Could not connect to Ollama. "
        "Make sure Ollama is running."
    ) from error

# -----------------------------------------
# 8. Save AI business insights report
# -----------------------------------------

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

REPORT_PATH.write_text(
    "# AI-Powered Customer Intelligence Report\n\n"
    + ai_report,
    encoding="utf-8"
)

# -----------------------------------------
# 9. Display final results
# -----------------------------------------

print("\n========== VERIFIED CHURN RATES ==========")

print("\nContract Type:")
print(contract_churn)

print("\nPayment Method:")
print(payment_churn)

print("\nInternet Service:")
print(internet_churn)

print("\n========== AI INSIGHTS GENERATED ==========")
print(ai_report)

print("\nReport saved successfully!")
print("Location:", REPORT_PATH)
