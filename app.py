import streamlit as st
import pandas as pd
import pyreadstat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.set_page_config(page_title="Waist Circumference & Mortality Risk", layout="wide")

# --- Load and Merge NHANES Data ---
@st.cache_data
def load_data():
    # Use pandas to read .XPT files with correct encoding
    demo = pd.read_sas("DEMO_L.XPT", format="xport", encoding="latin1")
    bmx = pd.read_sas("BMX_L.XPT", format="xport", encoding="latin1")

    df = pd.merge(demo, bmx, on="SEQN")
    df = df[df["RIDAGEYR"] >= 20]
    df = df[["SEQN", "RIDAGEYR", "RIAGENDR", "BMXWAIST"]].dropna()
    df.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "gender", "BMXWAIST": "waist"}, inplace=True)
    df["gender"] = df["gender"].map({1: "Male", 2: "Female"})
    return df


# --- Synthetic Mortality Label ---
def generate_mortality(df):
    np.random.seed(42)
    risk_score = 0.03 * df["age"] + 0.02 * df["waist"] + np.random.normal(0, 2, len(df))
    risk_prob = 1 / (1 + np.exp(-0.1 * (risk_score - 20)))
    df["mortality"] = np.random.binomial(1, risk_prob)
    df["risk_prob"] = risk_prob
    return df

# --- Train Model ---
def train_model(df):
    df_model = pd.get_dummies(df, columns=["gender"], drop_first=True)
    X = df_model[["age", "waist", "gender_Male"]]
    y = df_model["mortality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df["pred_prob"] = model.predict_proba(X)[:, 1]
    return model, report, df

# --- App UI ---
st.title("🧬 NHANES Synthetic Mortality Predictor")
st.markdown("This demo uses 2017–2018 NHANES data to simulate mortality risk using age, waist circumference, and gender.")

with st.spinner("Loading and preparing data..."):
    df = load_data()
    df = generate_mortality(df)
    model, report, df = train_model(df)

# --- Classification Report ---
st.header("🔎 Classification Report")
st.json(report)

# --- Risk Plot ---
st.header("📊 Predicted Mortality Risk by Waist Circumference")

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x="waist", y="pred_prob", hue="gender", alpha=0.7, ax=ax)
ax.set_title("Mortality Risk vs. Waist Circumference")
ax.set_ylabel("Predicted Mortality Risk")
st.pyplot(fig)

# --- User Prediction ---
st.header("🎯 Try It Yourself")

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.slider("Age", 20, 90, 50)
with col2:
    waist_input = st.slider("Waist Circumference (cm)", 50, 160, 90)
with col3:
    gender_input = st.selectbox("Gender", ["Male", "Female"])

input_data = pd.DataFrame({
    "age": [age_input],
    "waist": [waist_input],
    "gender_Male": [1 if gender_input == "Male" else 0]
})

risk = model.predict_proba(input_data)[0][1]
st.success(f"🔮 Predicted Mortality Risk: **{risk:.2%}**")

# --- Data Preview ---
st.header("📄 Data Sample")
st.dataframe(df[["SEQN", "age", "waist", "gender", "mortality", "pred_prob"]].head(10))
