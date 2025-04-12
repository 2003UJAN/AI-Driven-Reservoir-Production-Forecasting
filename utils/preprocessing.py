import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path='data/raw_data.csv', output_path='data/processed_data.csv', feature_cols=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(file_path)

    if feature_cols is None:
        feature_cols = ['pressure', 'flow_rate', 'water_cut']

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df.to_csv(output_path, index=False)

    scaler_path = output_path.replace('.csv', '_scaler.pkl')
    joblib.dump(scaler, scaler_path)

    print(f"✅ Data preprocessed and saved at {output_path}")
    print(f"✅ Scaler saved at {scaler_path}")
    return df, scaler
