import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path='data/raw_data.csv'):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    features = ['pressure', 'flow_rate', 'water_cut']
    df[features] = scaler.fit_transform(df[features])
    df.to_csv('data/processed_data.csv', index=False)
    return df, scaler
