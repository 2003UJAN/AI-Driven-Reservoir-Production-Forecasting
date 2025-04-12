import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(file_path='data/raw_data.csv', output_path='data/processed_data.csv', scaler_path='models/processed_data_scaler.pkl', feature_cols=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    df = pd.read_csv(file_path)

    if feature_cols is None:
        feature_cols = ['pressure', 'flow_rate', 'water_cut']

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv(output_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Data preprocessed and saved to {output_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    return df, scaler

if __name__ == "__main__":
    preprocess_data()
