import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.preprocessing import preprocess_data

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Reservoir Production Forecasting",
    layout="wide",
    page_icon="🛢️",
)

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load('models/xgb_model.pkl')
    except:
        st.error("❌ XGBoost model not found.")
        xgb_model = None

    try:
        lstm_model = load_model('models/lstm_model.h5', compile=False)
    except:
        st.error("❌ LSTM model not found.")
        lstm_model = None

    try:
        scaler = joblib.load('data/processed_data_scaler.pkl')
    except:
        st.warning("⚠️ Scaler not found. Proceeding without scaling.")
        scaler = None

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

    st.subheader("⚙️ Make Predictions")
    features = ['pressure', 'flow_rate', 'water_cut']

    # Optional: Scale data
    if scaler:
        df[features] = scaler.transform(df[features])

    # XGBoost Prediction
    if xgb_model:
        xgb_preds = xgb_model.predict(df[features])
        df['XGBoost_Predicted_Rate'] = xgb_preds

    # LSTM Prediction
    if lstm_model:
        lstm_input = np.reshape(df[features].values, (df.shape[0], 1, len(features)))
        lstm_preds = lstm_model.predict(lstm_input)
        df['LSTM_Predicted_Rate'] = lstm_preds.flatten()

    # ------------------ Results ------------------
    st.subheader("📈 Forecast Results")
    pred_cols = ['XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']
    shown_cols = features + [col for col in pred_cols if col in df.columns]
    st.dataframe(df[shown_cols].head())

    # ------------------ Visualization ------------------
    if 'XGBoost_Predicted_Rate' in df.columns or 'LSTM_Predicted_Rate' in df.columns:
        st.subheader("🔍 Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        if 'XGBoost_Predicted_Rate' in df.columns:
            sns.lineplot(x=range(len(df)), y=df['XGBoost_Predicted_Rate'], label='XGBoost')
        if 'LSTM_Predicted_Rate' in df.columns:
            sns.lineplot(x=range(len(df)), y=df['LSTM_Predicted_Rate'], label='LSTM')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Predicted Production Rate")
        ax.set_title("Reservoir Production Forecast")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No predictions made yet.")

else:
    st.warning("⚠️ Upload a CSV file with columns: pressure, flow_rate, water_cut to proceed.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
