# TeleConnect Customer Churn Analysis

## 1. Business Problem

TeleConnect is experiencing customer churn,
resulting in revenue loss and reduced retention.

## 2. Dataset Overview

- Total Customers: 7,043
- Total Features: 21
- Churned Customers: 1,869
- Retained Customers: 5,174
- Overall Churn Rate: 26.54%

## 3. Key Business Findings

### Contract Analysis
- Month-to-month churn: 42.71%
- One-year churn: 11.27%
- Two-year churn: 2.83%

### Payment Method Analysis
- Electronic check churn: 45.29%
- Mailed check churn: 19.11%
- Bank transfer churn: 16.71%
- Credit card churn: 15.24%

### Internet Service Analysis
- Fiber optic churn: 41.89%
- DSL churn: 18.96%
- No internet churn: 7.40%

### Customer Tenure Analysis
- First year churn: 47.44%
- Second year churn: 28.71%
- 25-48 months churn: 20.39%
- 49-72 months churn: 9.51%

## 4. Recommendations

1. Improve first-year customer retention.
2. Investigate month-to-month contract churn.
3. Review fiber optic customer complaints.
4. Investigate electronic payment customer experience.
5. Evaluate pricing and perceived customer value.

## 5. Next Steps

- Customer segmentation
- Statistical analysis
- Machine learning churn prediction
- Business recommendation automation
## Statistical Analysis

Statistical tests were performed to identify whether customer churn is associated with important customer characteristics.

### Chi-Square Tests

The chi-square tests found statistically significant associations between customer churn and:

- **Contract:** χ² = 1184.60, p < 0.001
- **Payment Method:** χ² = 648.14, p < 0.001
- **Internet Service:** χ² = 732.31, p < 0.001

These results indicate that churn rates differ across the categories of these variables.

### Independent Samples T-Test

Monthly charges were compared between churned and retained customers.

- Average monthly charges for churned customers: $74.44
- Average monthly charges for retained customers: $61.27
- T-statistic: 16.54
- P-value: 2.71 × 10⁻⁶⁰

The difference is statistically significant. Churned customers pay approximately $13.18 more per month on average.

### Business Conclusion

Contract type, payment method, internet service and monthly charges are statistically associated with customer churn. These findings can help identify customer groups that may need targeted retention strategies. Statistical association does not establish causation.