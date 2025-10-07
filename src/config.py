# Data paths
DATA_PATH = "./data/raw_data.csv"
MODEL_PATH = "models/lstm_model_latest.h5"
SCALER_PATH = "models/scaler_latest.pkl"

# Preprocessing settings
RESAMPLE_FREQ = "5T"   # 5 minutes
LAGS = [1, 2, 3]       # lag periods
ROLLING_AVG_WINDOW = 6
ROLLING_STD_WINDOW = 12
RESPONSE_TIME_THRESHOLD = 2000  # milliseconds

# Model settings
WINDOW_SIZE = 24   # Last 2 hours of data (24 x 5 min)
EPOCHS = 20
BATCH_SIZE = 32

# 🔥 CRITICAL FIX: Standardized features list
# This is the SINGLE source of truth for all features
FEATURES = [
    'CPU', 
    'RAM', 
    'response_time', 
    'cpu_rolling_mean', 
    'ram_rolling_mean', 
    'response_rolling_mean',
    'high_cpu', 
    'high_ram', 
    'is_peak_hour', 

]


def get_features_for_training():
    """Get features to use for training"""
    return FEATURES 

def get_features_for_prediction():
    """Get features to use for prediction (must match training)"""
    return FEATURES 