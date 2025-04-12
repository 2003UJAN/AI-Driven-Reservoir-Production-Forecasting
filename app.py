import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data

# Set Streamlit config
st.set_page_config(
    page_title="Reservoir Production Forecasting",
    layout="wide",
    page_icon="🛢️",
)

# Load models
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    return xgb_model, lstm_model

xgb_model, lstm_model = load_models()

# Title
st.markdown("""
    <div style="background-color:#002b36;padding:15px;border-radius:10px">
    <h2 style="color:white;text-align:center;">🛢️ AI-Driven Reservoir Production Forecasting</h2>
    </div>
""", unsafe_allow_html=True)

# File upload
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    features = ['pressure', 'flow_rate', 'water_cut']

    # XGBoost Predictions
    xgb_preds = xgb_model.predict(df[features])
    df['XGBoost_Predicted_Rate'] = xgb_preds

    # LSTM Predictions
    lstm_input = np.expand_dims(df[features].values, axis=0)  # (1, timesteps, features)
    lstm_preds = lstm_model.predict(lstm_input)[0]  # (timesteps, 1)
    
    # Match lengths
    if len(lstm_preds.flatten()) == len(df):
        df['LSTM_Predicted_Rate'] = lstm_preds.flatten()
    else:
        st.error("⚠️ LSTM prediction length mismatch. Please check model input compatibility.")
    
    # Results
    st.subheader("📈 Forecast Results")
    st.dataframe(df[['pressure', 'flow_rate', 'water_cut', 'XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']].head())

    # Plot
    st.subheader("🔍 Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=range(len(df)), y=df['XGBoost_Predicted_Rate'], label='XGBoost')
    if 'LSTM_Predicted_Rate' in df:
        sns.lineplot(x=range(len(df)), y=df['LSTM_Predicted_Rate'], label='LSTM')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Production Rate")
    ax.set_title("Reservoir Production Forecast")
    st.pyplot(fig)

else:
    st.warning("⚠️ Upload a CSV file with columns: pressure, flow_rate, water_cut.")

# Footer
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
