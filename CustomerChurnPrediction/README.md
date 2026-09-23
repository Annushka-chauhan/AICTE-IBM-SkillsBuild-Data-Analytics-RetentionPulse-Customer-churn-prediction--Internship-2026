# RetentionPulse — Customer Churn Prediction

RetentionPulse is an AI/ML data analytics project for the AICTE–IBM SkillsBuild Data Analytics with AI Internship Program 2026. It analyses the public Kaggle Telco Customer Churn dataset and predicts whether a customer is likely to churn.

## Dataset

Source: [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The downloaded source file is stored at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains customer demographics, tenure, services, contract details, billing information, and the target variable `Churn`.

## Methodology

1. Load and validate the source CSV.
2. Convert `TotalCharges` to numeric values.
3. Remove invalid total-charge records and duplicate customer IDs.
4. Create `ChurnFlag` and `TenureBand` features.
5. Explore churn by contract, tenure, charges, and customer attributes.
6. Train a class-balanced Logistic Regression pipeline with imputation, scaling, and one-hot encoding.
7. Evaluate accuracy, ROC-AUC, confusion matrix, and classification report.
8. Present the findings through a Streamlit dashboard.

## Run locally

```bash
cd ~/Downloads/CustomerChurnPrediction
python3 -m pip install -r requirements.txt
python3 AnushkaChauhan_CustomerChurnPrediction.py
python3 -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`.

## Structure

```text
CustomerChurnPrediction/
├── data/raw/
├── data/processed/
├── dashboard/app.py
├── outputs/figures/
├── src/churn_pipeline.py
├── AnushkaChauhan_CustomerChurnPrediction.py
├── AnushkaChauhan_CustomerChurnPrediction.ipynb
├── AnushkaChauhan_CustomerChurnPrediction_ProjectReport.docx
├── requirements.txt
└── README.md
```

## Limitations

The dataset is a historical sample and does not include customer acquisition cost, intervention history, or time-varying behaviour. Model probabilities should support prioritisation, not automatic adverse decisions. The project reports association and predictive performance, not causal claims.

## Author

Anushka Chauhan, University Roll Number 2300321540043, ABES Engineering College.
