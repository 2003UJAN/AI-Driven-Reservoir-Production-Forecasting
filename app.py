import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

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
    return xgb_model, lstm_model

xgb_model, lstm_model = load_models()

# ------------------ App Title ------------------
st.markdown(
    """
    <div style="background-color:#002b36;padding:15px;border-radius:10px">
    <h2 style="color:white;text-align:center;">🛢️ AI-Driven Reservoir Production Forecasting</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ Background Image ------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image:('assests/reservoir_graphic.jpg');
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
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

    # ------------------ Prediction ------------------
    st.subheader("⚙️ Make Predictions")

    features = ['pressure', 'flow_rate', 'water_cut']

    # XGBoost Prediction
    xgb_preds = xgb_model.predict(df[features])
    df['XGBoost_Predicted_Rate'] = xgb_preds

    # LSTM Prediction
    lstm_input = np.expand_dims(df[features].values, axis=0)  # shape: (1, timesteps, features)
    lstm_preds = lstm_model.predict(lstm_input)[0]  # shape: (timesteps, 1)
    df['LSTM_Predicted_Rate'] = lstm_preds.flatten()  # match length with df


    # ------------------ Results ------------------
    st.subheader("📈 Forecast Results")
    st.dataframe(df[['pressure', 'flow_rate', 'water_cut', 'XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']].head())

    # ------------------ Visualization ------------------
    st.subheader("🔍 Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x=range(len(df)), y='XGBoost_Predicted_Rate', label='XGBoost')
    sns.lineplot(data=df, x=range(len(df)), y='LSTM_Predicted_Rate', label='LSTM')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Production Rate")
    ax.set_title("Reservoir Production Forecast")
    st.pyplot(fig)

else:
    st.warning("⚠️ Upload a CSV file with columns: pressure, flow_rate, water_cut to proceed.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
