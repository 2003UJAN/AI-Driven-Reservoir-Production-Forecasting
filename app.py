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

# ------------------ Load Models & Scaler ------------------
@st.cache_resource
def load_models_and_scaler():
    xgb_model = joblib.load('models/xgb_model.pkl')
    lstm_model = load_model('models/lstm_model.h5', compile=False)
    scaler_path = 'data/processed_data_scaler.pkl'
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return xgb_model, lstm_model, scaler

xgb_model, lstm_model, scaler = load_models_and_scaler()

# ------------------ Theme Toggle ------------------
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stApp {
            background-color: #1e1e1e;
        }
        </style>
    """, unsafe_allow_html=True)

# ------------------ App Title ------------------
st.markdown(
    """
    <div style="background-color:#002b36;padding:15px;border-radius:10px">
    <h2 style="color:white;text-align:center;">🛢️ AI-Driven Reservoir Production Forecasting</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar: Upload ------------------
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("""
**Instructions**  
1. Upload a CSV with: `pressure`, `flow_rate`, `water_cut`.  
2. Models will predict production rate.  
3. See outputs in the tabs.
""")

# ------------------ Main Logic ------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    required_cols = {'pressure', 'flow_rate', 'water_cut'}
    if not required_cols.issubset(df.columns):
        st.error(f"Uploaded file must contain columns: {required_cols}")
        st.stop()

    features = ['pressure', 'flow_rate', 'water_cut']
    if scaler:
        df[features] = scaler.transform(df[features])
    else:
        st.warning("⚠️ Scaler not found — using raw feature values. Ensure preprocessing consistency.")

    # ------------------ Predictions ------------------
    with st.spinner("🔮 Making predictions..."):
        xgb_preds = xgb_model.predict(df[features])
        df['XGBoost_Predicted_Rate'] = xgb_preds

        lstm_input = np.expand_dims(df[features].values, axis=0)
        lstm_preds = lstm_model.predict(lstm_input)[0]
        pred_len = lstm_preds.shape[0]
        df_trimmed = df.iloc[:pred_len].copy()
        df_trimmed['LSTM_Predicted_Rate'] = lstm_preds.flatten()

    # ------------------ Tabs ------------------
    tab1, tab2, tab3 = st.tabs(["📊 Input Data", "📈 Forecast Table", "📉 Visualization"])

    with tab1:
        st.subheader("📊 Uploaded Input")
        st.dataframe(df[features].head())

    with tab2:
        st.subheader("📈 Forecast Results")
        st.dataframe(df_trimmed[['pressure', 'flow_rate', 'water_cut', 'XGBoost_Predicted_Rate', 'LSTM_Predicted_Rate']].head(20))

        # Download Predictions
        csv_download = df_trimmed.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_download,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("🔍 Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=range(len(df_trimmed)), y=df_trimmed['XGBoost_Predicted_Rate'], label='XGBoost')
        sns.lineplot(x=range(len(df_trimmed)), y=df_trimmed['LSTM_Predicted_Rate'], label='LSTM')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Predicted Production Rate")
        ax.set_title("Reservoir Production Forecast")
        ax.grid(True)
        ax.legend()
        sns.despine()
        st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file with columns: `pressure`, `flow_rate`, `water_cut` to get started.")

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("💡 Developed by [Ujan Pradhan] | Powered by Streamlit, LSTM & XGBoost", unsafe_allow_html=True)
