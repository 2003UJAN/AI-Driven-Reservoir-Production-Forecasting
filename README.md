# 🛢️ AI-Driven Reservoir Production Forecasting

Forecast oil reservoir production rates using cutting-edge machine learning models (LSTM + XGBoost) and visualize predictions with an interactive Streamlit dashboard.

![Reservoir Forecast Banner](https://i.ibb.co/k2QQV5C/oil-well-graphic.png)

## 📌 Project Overview

This project predicts oil reservoir production rates based on:

- Historical production logs
- Well parameters (e.g., pressure, flow rate)
- Pressure and water cut data

We use:
- **LSTM** (Long Short-Term Memory) Neural Networks for sequence modeling.
- **XGBoost** for gradient boosting-based regression.

The results are visualized using a **Streamlit-based web app** with a modern petroleum-themed UI.

---

## 🚀 Features

- 📈 Forecast reservoir output in real-time
- 🧠 Dual model prediction: LSTM and XGBoost
- 🎛️ Interactive input sliders for well parameters
- 🌍 Beautiful Streamlit UI with oil graphics
- 📊 Comparison bar chart for visual analysis

---

## 🗂️ Folder Structure

ai_reservoir_forecasting/ ├── data/ # Contains raw and processed datasets ├── models/ # Saved LSTM and XGBoost models ├── utils/ # Preprocessing scripts ├── generate_data.py # Synthetic data generator ├── train_models.py # Model training script ├── app.py # Streamlit dashboard app ├── requirements.txt # Python dependencies └── README.md # Project documentation

# Clone the repo
git clone https://github.com/your-username/ai_reservoir_forecasting.git
cd ai_reservoir_forecasting

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Train models
python train_models.py

# Launch the Streamlit app
streamlit run app.py
