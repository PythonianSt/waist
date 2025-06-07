import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Mortality Risk Predictor", layout="centered")

# ----- Data + Model Setup -----
@st.cache_data
def train_model():
    demo = pd.read_sas("DEMO_L.XPT", format="xport", encoding="latin1")
    bmx = pd.read_sas("BMX_L.XPT", format="xport", encoding="latin1")
    df = pd.merge(demo, bmx, on="SEQN")
    df = df[df["RIDAGEYR"] >= 20]
    df = df[["RIDAGEYR", "RIAGENDR", "BMXWAIST"]].dropna()
    df.rename(columns={"RIDAGEYR": "age", "RIAGENDR": "gender", "BMXWAIST": "waist"}, inplace=True)
    df["gender"] = df["gender"].map({1: "Male", 2: "Female"})

    # Create synthetic mortality label
    np.random.seed(42)
    risk_score = 0.03 * df["age"] + 0.02 * df["waist"] + np.random.normal(0, 2, len(df))
    risk_prob = 1 / (1 + np.exp(-0.1 * (risk_score - 20)))
    df["mortality"] = np.random.binomial(1, risk_prob)

    # Prepare features
    df_encoded = pd.get_dummies(df, columns=["gender"], drop_first=True)
    X = df_encoded[["age", "waist", "gender_Male"]]
    y = df_encoded["mortality"]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model()

# ----- UI -----
st.title("🧬 Mortality Risk Predictor (Teaching Demo)")
st.markdown("Estimate synthetic mortality risk based on age, waist circumference, and gender.")

# Input widgets
age_input = st.slider("Age", 20, 90, 50)
waist_input = st.slider("Waist Circumference (cm)", 60, 160, 90)
gender_input = st.selectbox("Gender", ["Male", "Female"])

# Format input
input_df = pd.DataFrame({
    "age": [age_input],
    "waist": [waist_input],
    "gender_Male": [1 if gender_input == "Male" else 0]
})

# Predict
pred_prob = model.predict_proba(input_df)[0][1]
st.success(f"🔮 Predicted Mortality Risk: **{pred_prob:.2%}**")

