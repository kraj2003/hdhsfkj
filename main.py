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
    model, scaler, history = train_lstm()
    
    # Save any results
    # results_df = pd.DataFrame({'prediction': [1, 0, 1]})
    # data_manager.save_results(results_df)

if __name__ == "__main__":
    print("Simple Corrected Delay Prediction")
    print("=" * 40)
    main()
    