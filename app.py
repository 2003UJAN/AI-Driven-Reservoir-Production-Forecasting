import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import preprocess_data

# ------------------ Set Page Config ------------------
st.set_page_config(
    page_title="Reservoir Production Forecasting",
    layout="wide",
    page_icon="🛢️",
)

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    scaler = joblib.load('models/processed_data_scaler.pkl')
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

    feature_cols = ['pressure', 'flow_rate', 'water_cut']
    
    # Scale features using loaded scaler
    scaled_features = scaler.transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)

    # XGBoost Prediction
    xgb_preds = xgb_model.predict(df_scaled)
    df['XGBoost_Predicted_Rate'] = xgb_preds

    # LSTM Prediction
    lstm_input = np.expand_dims(df_scaled.values, axis=0)  # shape: (1, timesteps, features)
    lstm_preds = lstm_model.predict(lstm_input)[0]

    # Align lengths
    pred_len = lstm_preds.shape[0]
    df_trimmed = df.iloc[:pred_len].copy()
    df_trimmed['LSTM_Predicted_Rate'] = lstm_preds.flatten()

    # ------------------ Results ------------------
    st.subheader("📈 Forecast Results")
    st.dataframe(df_trimmed[['pressure', 'flow_rate', 'water_cut', 'XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']].head())

    # ------------------ Visualization ------------------
    st.subheader("🔍 Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_trimmed, y='XGBoost_Predicted_Rate', x=df_trimmed.index, label='XGBoost')
    sns.lineplot(data=df_trimmed, y='LSTM_Predicted_Rate', x=df_trimmed.index, label='LSTM')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Production Rate")
    ax.set_title("Reservoir Production Forecast")
    st.pyplot(fig)

else:
    st.warning("⚠️ Upload a CSV file with columns: pressure, flow_rate, water_cut to proceed.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
