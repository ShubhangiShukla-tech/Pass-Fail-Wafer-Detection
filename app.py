import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt

# UI Theme
st.set_page_config(page_title="Wafer Pass/Fail Detection", page_icon="🧠", layout="centered")

# Custom CSS for pastel colors
st.markdown("""
    <style>
    body {
        background-color: #F9F5F0;
    }
    .stApp {
        background-color: #F6F8FC;
        font-family: 'Arial';
        color: #333;
    }
    .css-18e3th9 {
        background-color: #E8F0FE;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Semiconductor Wafer Pass/Fail Detector")
st.subheader("Upload features to check wafer quality")

# Load model and scaler
@st.cache_resource
def load_model():
    df = pd.read_csv("wafer.csv").dropna()
    X = df.drop("Pass/Fail", axis=1)
    y = df["Pass/Fail"].map({"Pass": 1, "Fail": 0}) if df["Pass/Fail"].dtype == object else df["Pass/Fail"]
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier().fit(scaler.transform(X), y)
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = load_model()

# Option to upload CSV
st.sidebar.header("📂 CSV Upload (optional)")
uploaded_file = st.sidebar.file_uploader("Upload wafer data (CSV)", type=["csv"])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(df_input)
    input_data = scaler.transform(df_input)
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)
    df_input["Prediction"] = ["Pass" if p == 1 else "Fail" for p in preds]
    df_input["Confidence"] = [f"{max(p)*100:.2f}%" for p in probs]
    st.write("✅ Prediction Results:")
    st.dataframe(df_input)
else:
    # Sidebar sliders for single prediction
    st.sidebar.title("🛠 Enter Wafer Data Manually")
    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.sidebar.slider(f"{feature}", float(-3), float(3), float(0), 0.1)

    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction]

    if prediction == 1:
        st.success(f"✅ **Wafer Status: PASS** ({probability:.2%} confidence)")
    else:
        st.error(f"❌ **Wafer Status: FAIL** ({probability:.2%} confidence)")

    st.write("📌 Input Features:")
    st.dataframe(input_df)

# Optional: Show SHAP Explanation
if st.checkbox("🧠 Show SHAP Feature Importance"):
    df = pd.read_csv("wafer.csv").dropna()
    X = df.drop("Pass/Fail", axis=1)
    y = df["Pass/Fail"].map({"Pass": 1, "Fail": 0}) if df["Pass/Fail"].dtype == object else df["Pass/Fail"]
    X_scaled = scaler.transform(X)

    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("🔍 SHAP Summary Plot (Feature Importance)")
    shap.summary_plot(shap_values, features=X, feature_names=feature_names)
    st.pyplot()
