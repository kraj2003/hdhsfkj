# src/config.py

DATA_PATH = "./data/raw_data.csv"
MODEL_PATH = "models/my_lstm_model.h5"

# Preprocessing
RESAMPLE_FREQ = "10T"   # 5 minutes
LAGS = [2, 4, 6, 12]   # past delays
ROLLING_AVG_WINDOW = 12
ROLLING_STD_WINDOW = 36

# Model
WINDOW_SIZE = 24   # Last 1 hour of data (12 x 5 min)
EPOCHS = 20
BATCH_SIZE = 32
