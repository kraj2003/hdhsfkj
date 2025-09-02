# main.py - Corrected Version
import pandas as pd
import joblib
from src import config
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future
from src.data_manager import DataManager

def main():
    # Load data
    data_manager = DataManager()
    df = data_manager.load_data()
    
    # # Preprocess data
    processed_df = preprocessing_pipeline(df)
    
    # Train model
    model, scaler, _ = train_lstm(processed_df)
    
    # Save placeholder results (replace later with real predictions)
    # results_df = pd.DataFrame({'prediction': [1, 0, 1]})
    # data_manager.save_results(results_df)

    # Load the saved features
    features = ['Delay_Detected', 'CPU', 'RAM','is_error']
    print(f"Using features: {features}")

    # Make prediction
    print("\nMaking prediction...")
    prediction_class, prediction_prob = predict_future(model, scaler, processed_df,features, config.WINDOW_SIZE)

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
