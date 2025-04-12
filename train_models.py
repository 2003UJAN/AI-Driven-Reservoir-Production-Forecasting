import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocessing import preprocess_data
from utils.preprocessing import preprocess_data

def load_data():
    df, _ = preprocess_data()
    features = ['pressure', 'flow_rate', 'water_cut']
    target = 'flow_rate'  # or 'production_rate' if applicable
    return df[features], df[target]

def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    return model

def train_lstm(X_train, y_train, X_val, y_val):
    X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
              epochs=50, batch_size=8, verbose=1, callbacks=[es])
    return model

def save_models(xgb_model, lstm_model):
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    lstm_model.save('models/lstm_model.h5')
    print("✅ Models saved to models/")

def main():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("🧠 Training LSTM...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val)

    save_models(xgb_model, lstm_model)

if __name__ == "__main__":
    main()
