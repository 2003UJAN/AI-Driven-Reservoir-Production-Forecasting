import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ Set Page Config ------------------
st.set_page_config(
    page_title="Reservoir Production Forecasting",
    layout="wide",
    page_icon="🛢️",
)

# ------------------ Load Models and Scaler ------------------
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    scaler = joblib.load('data/processed_data_scaler.pkl')
    return xgb_model, lstm_model, scaler

xgb_model, lstm_model, scaler = load_models()

# ------------------ App Title ------------------
st.markdown(
    """
    <div style="background-color:#002b36;padding:15px;border-radius:10px">
    <h2 style="color:white;text-align:center;">🛢️ AI-Driven Reservoir Production Forecasting</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ Upload Data ------------------
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    features = ['pressure', 'flow_rate', 'water_cut']
    df_scaled = df.copy()
    df_scaled[features] = scaler.transform(df[features])

    st.subheader("⚙️ Predictions")

    # XGBoost Prediction
    xgb_preds = xgb_model.predict(df_scaled[features])
    df['XGBoost_Predicted_Rate'] = xgb_preds

    # LSTM Prediction
    lstm_input = np.reshape(df_scaled[features].values, (df_scaled.shape[0], 1, len(features)))
    lstm_preds = lstm_model.predict(lstm_input)
    df['LSTM_Predicted_Rate'] = lstm_preds.flatten()

    st.dataframe(df[['pressure', 'flow_rate', 'water_cut', 'XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']].head())

    # ------------------ Visualization ------------------
    st.subheader("📈 Forecast Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=range(len(df)), y=df['XGBoost_Predicted_Rate'], label='XGBoost')
    sns.lineplot(x=range(len(df)), y=df['LSTM_Predicted_Rate'], label='LSTM')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Rate")
    ax.set_title("Reservoir Production Forecast")
    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV with columns: pressure, flow_rate, water_cut.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
