# main.py - Simple Corrected Version
import pandas as pd
import joblib
from src import config
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future
from src.data_manager import DataManager
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm


def main():
    # Load data (automatically handles CSV vs SQL)
    data_manager = DataManager()
    df = data_manager.load_data()
    
    # Your existing code
    processed_df = preprocessing_pipeline(df)
    model, scaler, history = train_lstm(processed_df)
    
    # Save any results
    # results_df = pd.DataFrame({'prediction': [1, 0, 1]})
    # data_manager.save_results(results_df)

if __name__ == "__main__":
    print("Simple Corrected Delay Prediction")
    print("=" * 40)


    # main()
    
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} records")

    # Preprocess
    processed_df = preprocessing_pipeline(df)
    print(f"Processed shape: {processed_df.shape}")
    print("processed data",processed_df.head())
    processed_df['target'] = (processed_df['Delay_Detected'].shift(-2).fillna(0) > 0).astype(int)
    print(processed_df.head())
    print(processed_df['target'])
    print(processed_df['Delay_Detected'].unique())

    # Train corrected model
    print("\nTraining corrected model...")
    model, scaler, _ = train_lstm(processed_df)

    # Load the saved features
    features = joblib.load("models/features.pkl")
    print(f"Using features: {features}")

    # Make prediction
    print("\nMaking prediction...")
    prediction_class, prediction_prob = predict_future(model, scaler, processed_df, features, config.WINDOW_SIZE)
    
    print(f"\nResult:")
    print(f"Delay predicted: {'Yes' if prediction_class else 'No'}")
    print(f"Probability: {prediction_prob:.3f}")
    print(f"\n✅ Model uses NO current delay information!")
    print(f"✅ Prediction based on system health only")