from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"


def load_and_clean() -> pd.DataFrame:
    data = pd.read_csv(RAW_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).drop_duplicates(subset=["customerID"]).copy()
    data["ChurnFlag"] = (data["Churn"] == "Yes").astype(int)
    data["TenureBand"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    return data


def train_model(data: pd.DataFrame) -> tuple[Pipeline, dict]:
    features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    X = data[features]
    y = data["ChurnFlag"]
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": int(len(y_test)),
    }
    return model, metrics


def generate_figures(data: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    churn = data["Churn"].value_counts()
    plt.figure(figsize=(7, 5))
    churn.plot(kind="bar", color=["#4099ff", "#ff5370"])
    plt.title("Customer Churn Distribution")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_distribution.png", dpi=180)
    plt.close()

    contract = pd.crosstab(data["Contract"], data["Churn"], normalize="index").mul(100)
    contract.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#2ed8b6", "#ff5370"])
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Percentage of customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_by_contract.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="Churn", y="MonthlyCharges", palette=["#4099ff", "#ff5370"])
    plt.title("Monthly Charges by Churn Outcome")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthly_charges_by_churn.png", dpi=180)
    plt.close()

    tenure = data.groupby("TenureBand", observed=False)["ChurnFlag"].mean().mul(100).reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=tenure, x="TenureBand", y="ChurnFlag", color="#7759de")
    plt.title("Churn Rate by Tenure Band")
    plt.ylabel("Churn rate (%)")
    plt.xlabel("Tenure")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_by_tenure.png", dpi=180)
    plt.close()


def run() -> dict:
    data = load_and_clean()
    model, metrics = train_model(data)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(PROCESSED_DIR / "telco_churn_clean.csv", index=False)
    generate_figures(data)
    summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "churn_yes": int(data["ChurnFlag"].sum()),
        "churn_rate": round(float(data["ChurnFlag"].mean()), 4),
        "unique_customers": int(data["customerID"].nunique()),
        "metrics": metrics,
        "top_churn_contract": data.groupby("Contract")["ChurnFlag"].mean().idxmax(),
        "top_churn_contract_rate": round(float(data.groupby("Contract")["ChurnFlag"].mean().max()), 4),
    }
    (PROCESSED_DIR / "churn_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"data": data, "model": model, "summary": summary}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2))
