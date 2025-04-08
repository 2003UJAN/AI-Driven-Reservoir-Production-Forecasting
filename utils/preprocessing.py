import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path='/content/data/raw_data.csv'):
    # Create output directory if it doesn't exist
    os.makedirs('/content/data', exist_ok=True)

    # Load raw data
    df = pd.read_csv(file_path)

    # Scale selected features
    scaler = MinMaxScaler()
    features = ['pressure', 'flow_rate', 'water_cut']
    df[features] = scaler.fit_transform(df[features])

    # Save processed data
    df.to_csv('/content/data/processed_data.csv', index=False)
    
    return df, scaler

# Example usage (only if running this script interactively in Colab)
if __name__ == "__main__":
    # Simulate raw data creation if needed
    import numpy as np

    sample_data = pd.DataFrame({
        'pressure': np.random.uniform(2000, 5000, 100),
        'flow_rate': np.random.uniform(50, 500, 100),
        'water_cut': np.random.uniform(0.0, 1.0, 100)
    })
    os.makedirs('/content/data', exist_ok=True)
    sample_data.to_csv('/content/data/raw_data.csv', index=False)

    # Run preprocessing
    df, scaler = preprocess_data()
    print("✅ Data preprocessed and saved at /content/data/processed_data.csv")
