# Project Decisions

## Topic

Customer churn prediction was selected as a supervised machine-learning problem suitable for the internship's Data → Information → Insights → Decision → Action workflow.

## Dataset

The public Kaggle Telco Customer Churn dataset was selected instead of an IBM-hosted source, as requested. The Kaggle URL and download endpoint are documented in the README.

## Model

Logistic Regression was selected as an explainable baseline classifier. The pipeline uses class balancing, numeric imputation and scaling, categorical imputation, and one-hot encoding. ROC-AUC and the confusion matrix are reported alongside accuracy.

## Dashboard

The dashboard uses the same Gradient Able-inspired visual direction as CivicPulse: dark slate sidebar, blue/purple/teal accents, white admin-style cards, and Plotly charts.
