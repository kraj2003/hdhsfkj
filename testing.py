
# main.py - Corrected Version
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

def make_prediction():
    """Simple prediction function"""
    print("🔮 Making delay prediction...")
    
    try:
        # Check if model exists
        if not os.path.exists('models/lstm_model_latest.h5'):
            print("❌ No trained model found")
            return "no_model"
        
        # Load recent data
        data_manager = DataManager()
        df = data_manager.load_data()
        
        # Get last few hours of data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_cutoff = df['timestamp'].max() - timedelta(hours=3)
        recent_df = df[df['timestamp'] >= recent_cutoff].copy()
        
        if len(recent_df) < 10:
            print("❌ Not enough recent data")
            return "insufficient_data"
        
        # Simple preprocessing
        processed_df = preprocessing_pipeline(recent_df)
        print(processed_df)

        model = load_model("./models/lstm_model_latest.h5")
        scaler = joblib.load("./models/scaler_latest.pkl")
        
        with open("feature_names.json", "r") as f:
            features = json.load(f)

        print("✅ Features type:", type(features))
        print("✅ Features content:", features)

        # features=processed_df.columns

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
        
        return f"prediction_{prediction}_prob_{probability:.3f}"
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return "prediction_failed"

with open("feature_names.json", "r") as f:
            features = json.load(f)

# Load recent data
data_manager = DataManager()
df = data_manager.load_data()

print("✅ Features type:", type(features))
print("✅ Features content:", features)
# Get last few hours of data
df['timestamp'] = pd.to_datetime(df['timestamp'])
recent_cutoff = df['timestamp'].max() - timedelta(hours=3)
recent_df = df[df['timestamp'] >= recent_cutoff].copy()
print(recent_df)
processed_df = preprocessing_pipeline(recent_df)
print(processed_df)
print(type(processed_df))
print(processed_df.head())

if __name__ == "__main__":
    make_prediction()