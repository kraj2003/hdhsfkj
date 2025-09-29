# main.py - Corrected Version
import pandas as pd
import joblib
from src import config
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future
from src.data_manager import DataManager
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import logging
from src import config
logger = logging.getLogger(__name__)
import json
from src.data_validation import validate_raw_data
def main():
    # Load data
    data_manager = DataManager()
    df = data_manager.load_data()
    # print(df.head())
    # print(df[['CPU','RAM','sc_status','time_taken','is_error']])
    
    # Preprocess data
    processed_df = preprocessing_pipeline(df)

    # total_records,issues,is_valid,null_percentage = validate_raw_data(processed_df)
    
    # Train model
    model, scaler, _ = train_lstm(processed_df)

    with open("./models/feature_names.json", "r") as f:
        features = json.load(f)
    # print(f"Using features: {features}")
    model = load_model("./models/lstm_model_latest.h5")
    scaler= joblib.load("./models/scaler_latest.pkl")

    # Filter last 6 hours
    # last_6_hours_df = processed_df[processed_df['response_time'] >= (processed_df['response_time'].max() - pd.Timedelta(hours=6))]
    data=pd.read_csv('./data/final_dataset_3.csv')
    processed_df_2 = preprocessing_pipeline(data)
    # total_records,issues,is_valid,null_percentage = validate_raw_data(processed_df_2)
    # print(f"Total records: {total_records}")
    # print(f"Data validation issues: {issues}")
    # print(f"Is data valid: {is_valid}")
    # print(f"Percentage of NaN values: {null_percentage:.2f}%")
    print(f"NaN values remaining: {processed_df_2.isnull().sum().sum()}")

    # Make prediction
    print("\nMaking prediction...")
    prediction_class, prediction_prob = predict_future(model, scaler, processed_df_2,features, config.WINDOW_SIZE)

    results_df = pd.DataFrame({'prediction': [prediction_class]})
    data_manager.save_results(results_df)

    print("\nResult:")
    print(f"Delay predicted: {'Yes' if prediction_class else 'No'}")
    print(f"Probability: {prediction_prob:.3f}")
    print("\n✅ Model uses NO current delay information!")
    print("✅ Prediction based on system health only")

if __name__ == "__main__":
    print("Simple Corrected Delay Prediction")
    print("=" * 40)
    main()
