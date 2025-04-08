import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Load models
@st.cache_resource
def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lstm_model = load_model('models/lstm_model.h5')
    return xgb_model, lstm_model

# Load models
xgb_model, lstm_model = load_models()

# App title and banner
st.set_page_config(page_title="Reservoir Forecasting App", layout="wide")
st.title("🛢️ AI-Driven Reservoir Production Forecasting")
st.markdown("Predict future production using historical reservoir parameters!")

# Sidebar inputs
st.sidebar.header("Input Reservoir Parameters")
pressure = st.sidebar.slider("Reservoir Pressure (psi)", 1000, 5000, 3000)
flow_rate = st.sidebar.slider("Initial Flow Rate (STB/day)", 100, 5000, 1500)
water_cut = st.sidebar.slider("Water Cut (%)", 0, 100, 30)

input_data = pd.DataFrame([[pressure, flow_rate, water_cut]], 
                          columns=['pressure', 'flow_rate', 'water_cut'])

# Normalize input
def scale_input(df, scaler_path='data/scaler.pkl'):
    scaler = joblib.load(scaler_path)
    return scaler.transform(df)

# Check if scaler exists
if not os.path.exists('data/scaler.pkl'):
    st.error("Scaler not found! Please run `preprocess_data.py` and save the scaler.")
    st.stop()

scaled_input = scale_input(input_data)

# Reshape for LSTM
scaled_input_lstm = np.reshape(scaled_input, (scaled_input.shape[0], 1, scaled_input.shape[1]))

# Predictions
xgb_pred = xgb_model.predict(scaled_input)[0]
lstm_pred = lstm_model.predict(scaled_input_lstm, verbose=0)[0][0]

# Results
st.subheader("📊 Predicted Production Rates")
st.success(f"🔸 XGBoost Prediction: **{xgb_pred:.2f} STB/day**")
st.info(f"🔹 LSTM Prediction: **{lstm_pred:.2f} STB/day**")

# Cool graphic
st.markdown("---")
st.subheader("🖼️ Reservoir Graphic")
image = Image.open("assets/reservoir_graphic.jpg")
st.image(image, caption="Petroleum Reservoir Schematic", use_column_width=True)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, LSTM, and XGBoost.")
