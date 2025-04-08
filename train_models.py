import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import xgboost as xgb
import joblib

def train_lstm_model(X, y):
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=1)
    model.save('models/lstm_model.h5')

def train_xgboost_model(X, y):
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X, y)
    model.save_model('models/xgboost_model.json')

if __name__ == "__main__":
    df, _ = preprocess_data()
    X = df[['pressure', 'flow_rate', 'water_cut']].values
    y = df['production_rate'].values

    train_lstm_model(X, y)
    train_xgboost_model(X, y)
