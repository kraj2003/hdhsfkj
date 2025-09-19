import pandas as pd
import joblib
import json
from src import config
from tensorflow.keras.models import load_model
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future
from src.data_manager import DataManager
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)
import os

if not os.path.exists('models/lstm_model_latest.h5'):
    print("❌ No trained model found")

# Load recent data
data_manager = DataManager()
df = data_manager.load_data()

# Get last few hours of data
df['timestamp'] = pd.to_datetime(df['timestamp'])
recent_cutoff = df['timestamp'].max() - timedelta(hours=3)
recent_df = df[df['timestamp'] >= recent_cutoff].copy()

if len(recent_df) < 10:
    print("❌ Not enough recent data")

# Simple preprocessing
processed_df = preprocessing_pipeline(recent_df)
print(processed_df)

model = load_model("./models/lstm_model_latest.h5",  compile=False)
scaler = joblib.load("./models/scaler_latest.pkl")

features= [
            'CPU', 'RAM', 'response_time', 'is_error',
            'cpu_lag_1', 'ram_lag_1', 'response_lag_1',
            'cpu_rolling_mean', 'ram_rolling_mean', 'response_rolling_mean',
            'high_cpu', 'high_ram', 'is_peak_hour', 'hour'
        ]

print("✅ Features type:", type(features))
print("✅ Features content:", features)



# Make prediction
prediction, probability = predict_future(model,scaler,processed_df, features, config.WINDOW_SIZE)

# Save result
result = pd.DataFrame({
    'timestamp': [datetime.now()],
    'prediction': [prediction],
    'probability': [probability]
})
data_manager.save_results(result)

# Simple alert
if prediction == 1:
    print(f"🚨 DELAY PREDICTED - Probability: {probability:.3f}")
else:
    print(f"✅ No delay expected - Probability: {probability:.3f}")