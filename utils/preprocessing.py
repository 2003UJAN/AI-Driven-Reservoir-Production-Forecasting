import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(file_path='data/raw_data.csv', output_path='data/processed_data.csv', feature_cols=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(file_path)

    if feature_cols is None:
        feature_cols = ['pressure', 'flow_rate', 'water_cut']

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv(output_path, index=False)
    joblib.dump(scaler, 'data/processed_data_scaler.pkl')

    print(f"✅ Data preprocessed and saved at {output_path}")
    return df, scaler

if __name__ == "__main__":
    import numpy as np

    sample_data = pd.DataFrame({
        'pressure': np.random.uniform(2000, 5000, 100),
        'flow_rate': np.random.uniform(50, 500, 100),
        'water_cut': np.random.uniform(0.0, 1.0, 100)
    })

    raw_data_path = 'data/raw_data.csv'
    processed_data_path = 'data/processed_data.csv'

    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    sample_data.to_csv(raw_data_path, index=False)

    preprocess_data(file_path=raw_data_path, output_path=processed_data_path)
