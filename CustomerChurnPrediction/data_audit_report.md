# Customer Churn Data Audit Report

## Source

Kaggle: `blastchar/telco-customer-churn`

## Initial shape

- Rows: 7,043
- Columns: 21

## Target distribution

- Churn = No: 5,174
- Churn = Yes: 1,869
- Positive churn rate: approximately 26.5%

## Data preparation

`TotalCharges` is stored as text in the source file and is converted to numeric. Rows that cannot be converted are excluded because the field is required for modelling. Duplicate customer IDs are removed before analysis.

## Analytical caution

The dataset is a historical sample. Churn predictions should be used for prioritised retention analysis and human review, not automatic denial of service or adverse customer treatment.
