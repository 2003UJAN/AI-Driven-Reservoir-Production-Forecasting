import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import os

# Load and prepare data
df = pd.read_csv('data/processed_data.csv')
features = ['pressure', 'flow_rate', 'water_cut']
target = 'flow_rate'

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost (or any regressor like GradientBoosting)
xgb_model = GradientBoostingRegressor()
xgb_model.fit(X_train, y_train)

# Save XGBoost model
os.makedirs('models', exist_ok=True)
joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("✅ XGBoost model saved to models/xgb_model.pkl")

# LSTM requires reshaping
X_lstm = np.expand_dims(X.values, axis=0)   # (1, timesteps, features)
y_lstm = np.expand_dims(y.values, axis=0)   # (1, timesteps)

# Build simple LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=True))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mse', optimizer='adam')

# Train LSTM
lstm_model.fit(X_lstm, y_lstm, epochs=100, verbose=0, callbacks=[EarlyStopping(patience=10)])

# Save LSTM model
lstm_model.save('models/lstm_model.h5')
print("✅ LSTM model saved to models/lstm_model.h5")
