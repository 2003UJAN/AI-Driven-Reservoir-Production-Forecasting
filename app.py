import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_data

# Load models
lstm_model = load_model('models/lstm_model.h5')
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('models/xgboost_model.json')

# App UI
st.set_page_config(page_title="🛢️ Reservoir Production Forecasting", layout="wide")
st.title("🛢️ AI-Driven Reservoir Production Forecasting")
st.markdown("Predict oil reservoir production using **LSTM** and **XGBoost** based on well parameters.")

st.image("https://i.ibb.co/k2QQV5C/oil-well-graphic.png", use_column_width=True)

# Inputs
st.sidebar.header("Enter Well Parameters:")
pressure = st.sidebar.slider("Pressure (psi)", 2000, 5000, 3000)
flow_rate = st.sidebar.slider("Flow Rate (barrels/day)", 50, 500, 200)
water_cut = st.sidebar.slider("Water Cut (%)", 0.0, 1.0, 0.3)

input_df = pd.DataFrame({
    'pressure': [pressure],
    'flow_rate': [flow_rate],
    'water_cut': [water_cut]
})

# Normalize like training
_, scaler = preprocess_data()
input_scaled = scaler.transform(input_df)

# LSTM
lstm_input = np.reshape(input_scaled, (input_scaled.shape[0], 1, input_scaled.shape[1]))
lstm_pred = lstm_model.predict(lstm_input)

# XGBoost
xgb_pred = xgb_model.predict(input_scaled)

# Output
st.subheader("📈 Forecasted Production Rates:")
st.success(f"LSTM Model Prediction: **{lstm_pred[0][0]:.2f} barrels/day**")
st.info(f"XGBoost Model Prediction: **{xgb_pred[0]:.2f} barrels/day**")

# Plot
st.subheader("📊 Visual Comparison")
fig, ax = plt.subplots()
ax.bar(['LSTM', 'XGBoost'], [lstm_pred[0][0], xgb_pred[0]], color=['#f77f00', '#003049'])
ax.set_ylabel("Predicted Production Rate")
st.pyplot(fig)
