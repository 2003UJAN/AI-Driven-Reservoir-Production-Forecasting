import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def preprocess_data(file_path='data/raw_data.csv', output_path='data/processed_data.csv', feature_cols=None):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load raw data
    df = pd.read_csv(file_path)

    # Default feature columns
    if feature_cols is None:
        feature_cols = ['pressure', 'flow_rate', 'water_cut']

    # Scale selected features
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save processed data
    df.to_csv(output_path, index=False)
    
    print(f"✅ Data preprocessed and saved at {output_path}")
    return df, scaler

# Example usage when running directly
if __name__ == "__main__":
    import numpy as np

    # Simulate raw data
    sample_data = pd.DataFrame({
        'pressure': np.random.uniform(2000, 5000, 100),
        'flow_rate': np.random.uniform(50, 500, 100),
        'water_cut': np.random.uniform(0.0, 1.0, 100)
    })

    raw_data_path = 'data/raw_data.csv'
    processed_data_path = 'data/processed_data.csv'

    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    sample_data.to_csv(raw_data_path, index=False)

    # Run preprocessing
    df, scaler = preprocess_data(file_path=raw_data_path, output_path=processed_data_path)
