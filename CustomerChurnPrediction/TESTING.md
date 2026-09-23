# Testing

## Checks

- Source dataset downloaded from the documented Kaggle URL.
- Dataset schema and target balance audited.
- `TotalCharges` conversion and invalid-record handling executed.
- Duplicate customer IDs checked.
- Model pipeline executed with stratified train/test split.
- Metrics generated: accuracy, ROC-AUC, classification report, and confusion matrix.
- Dashboard syntax checked with `py_compile`.
- Streamlit startup checked with an HTTP response.

## Run

```bash
python3 AnushkaChauhan_CustomerChurnPrediction.py
python3 -m streamlit run dashboard/app.py
```

Do not report a metric as passed unless the command has actually been run in the current environment.
