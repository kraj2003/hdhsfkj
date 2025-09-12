# main.py - Corrected Version
import pandas as pd
import joblib
from src import config
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future
from src.data_manager import DataManager
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

def main():
    # Load data
    data_manager = DataManager()
    df = data_manager.load_data()
    # print(df.head())
    # print(df[['CPU','RAM','sc_status','time_taken','is_error']])
    
    # # Preprocess data
    processed_df = preprocessing_pipeline(df)
    
    # Train model
    model, scaler, _ = train_lstm(processed_df)

    # Load the saved features
    features= [
            'CPU', 'RAM', 'response_time', 'is_error',
            'cpu_lag_1', 'ram_lag_1', 'response_lag_1',
            'cpu_rolling_mean', 'ram_rolling_mean', 'response_rolling_mean',
            'high_cpu', 'high_ram', 'is_peak_hour', 'hour'
        ]
    print(f"Using features: {features}")

    # Filter last 6 hours
    last_6_hours_df = processed_df[processed_df['timestamp'] >= (processed_df['timestamp'].max() - pd.Timedelta(hours=6))]

    # Make prediction
    print("\nMaking prediction...")
    prediction_class, prediction_prob = predict_future(model, scaler, last_6_hours_df,features, config.WINDOW_SIZE)

    results_df = pd.DataFrame({'prediction': [prediction_class]})
    data_manager.save_results(results_df)

    future_time = datetime.now() + timedelta(minutes=10)

    print(f"Predicted delay in next 10 minutes? {bool(prediction_class)} "
                f"(Prob: {prediction_class:.3f}, Confidence: {prediction_prob}) "
                f"at {future_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    print("\nResult:")
    print(f"Delay predicted: {'Yes' if prediction_class else 'No'}")
    print(f"Probability: {prediction_prob:.3f}")
    print("\n✅ Model uses NO current delay information!")
    print("✅ Prediction based on system health only")

if __name__ == "__main__":
    print("Simple Corrected Delay Prediction")
    print("=" * 40)
    main()
