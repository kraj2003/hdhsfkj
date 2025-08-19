# main.py
import pandas as pd
from src import config
from src.preprocessing import preprocessing_pipeline
from src.train import train_lstm
from src.evaluate import predict_future

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(config.DATA_PATH)

    # Preprocess
    processed_df = preprocessing_pipeline(df)

    # Train
    model, scaler, _ = train_lstm(processed_df)

    # Evaluate latest
    predict_future(model, scaler, processed_df, ['Delay_Detected', 'CPU', 'RAM'], config.WINDOW_SIZE)
