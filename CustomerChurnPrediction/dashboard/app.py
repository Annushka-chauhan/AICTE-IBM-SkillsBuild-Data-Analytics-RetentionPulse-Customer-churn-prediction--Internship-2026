from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"

st.set_page_config(page_title="Customer Churn Intelligence", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    .stApp { background:#f6f7fb; color:#263238; font-family:'Poppins',sans-serif; }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#3b4650 0%,#263238 100%); }
    [data-testid="stSidebar"] * { color:#e8eef4; }
    div.block-container { padding:2rem 2.5rem 3rem; max-width:1500px; }
    [data-testid="stMetric"] { background:white; border-top:4px solid #4099ff; border-radius:5px; padding:1.2rem; box-shadow:0 1px 3px rgba(4,26,55,.16); }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] { border-top-color:#2ed8b6; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] { border-top-color:#7759de; }
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] { border-top-color:#ff5370; }
    .brand { color:white; font-size:21px; font-weight:700; padding-bottom:1.2rem; border-bottom:1px solid rgba(255,255,255,.15); margin-bottom:1.3rem; }
    .eyebrow { color:#748892; font-size:12px; letter-spacing:.08em; text-transform:uppercase; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
st.sidebar.markdown('<div class="brand">RetentionPulse</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Dashboard filters")
contracts = st.sidebar.multiselect("Contract type", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
senior = st.sidebar.multiselect("Senior citizen", [0, 1], default=[0, 1])

filtered = df[df["Contract"].isin(contracts) & df["InternetService"].isin(internet) & df["SeniorCitizen"].isin(senior)]
if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

st.title("RetentionPulse — Customer Churn Intelligence")
st.caption("Predicting customer churn risk through analytics and explainable machine learning")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(filtered):,}")
col2.metric("Churn rate", f"{filtered['ChurnFlag'].mean():.1%}")
col3.metric("Average tenure", f"{filtered['tenure'].mean():.1f} months")
col4.metric("Average monthly charge", f"${filtered['MonthlyCharges'].mean():.2f}")

trend = filtered.groupby("TenureBand", observed=False)["ChurnFlag"].mean().mul(100).reset_index()
fig_tenure = px.bar(trend, x="TenureBand", y="ChurnFlag", title="Churn rate by tenure band", labels={"ChurnFlag": "Churn rate (%)"})
fig_tenure.update_layout(template="plotly_white", height=350)

contract = filtered.groupby("Contract")["ChurnFlag"].mean().mul(100).reset_index()
fig_contract = px.bar(contract, x="Contract", y="ChurnFlag", title="Churn rate by contract", labels={"ChurnFlag": "Churn rate (%)"}, color="ChurnFlag", color_continuous_scale=["#2ed8b6", "#ff5370"])
fig_contract.update_layout(template="plotly_white", height=350, showlegend=False)

charges = filtered.groupby("Churn", as_index=False)["MonthlyCharges"].mean()
fig_charges = px.bar(charges, x="Churn", y="MonthlyCharges", title="Average monthly charges by churn outcome", labels={"MonthlyCharges": "Average monthly charge"})
fig_charges.update_layout(template="plotly_white", height=350)

st.markdown('<div class="eyebrow">Customer risk overview</div>', unsafe_allow_html=True)
left, right = st.columns(2)
left.plotly_chart(fig_tenure, use_container_width=True)
right.plotly_chart(fig_contract, use_container_width=True)
st.plotly_chart(fig_charges, use_container_width=True)
st.subheader("Filtered customer preview")
st.dataframe(filtered.head(50), use_container_width=True)
