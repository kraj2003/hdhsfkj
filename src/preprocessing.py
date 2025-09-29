# src/preprocessing.py - FIXED VERSION
import pandas as pd
import numpy as np
from src import config

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    
    # Create delay detection based on response time threshold
    df['Delay_Detected'] = (df['response_time'] > config.RESPONSE_TIME_THRESHOLD).astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Original data shape: {df.shape}")
    print(f"Delay instances: {df['Delay_Detected'].sum()}/{len(df)} ({df['Delay_Detected'].mean()*100:.1f}%)")
    
    # Resample to fixed intervals with proper aggregation
    df_5min = (
        df.set_index('timestamp')
        .resample(config.RESAMPLE_FREQ)
        .agg({
            'Delay_Detected': 'max',  # Changed from 'sum' to 'max' - any delay in window = 1
            'response_time': 'mean',
            'CPU': 'mean',
            'RAM': 'mean', 
            # 'status_code': 'mean',
            'is_error': 'max'  # Changed from 'sum' to 'max'
        })
    )
    
    print(f"After resampling: {df_5min.shape}")
    
    # PROPER NaN handling - use modern pandas methods
    # Forward fill for system metrics (they change slowly)
    df_5min['RAM'] = df_5min['RAM'].ffill().fillna(df_5min['RAM'].median())
    # df_5min['CPU'] = df_5min['CPU'].ffill().fillna(df_5min['CPU'].median())
    
    # # For response time, use forward fill then median
    # df_5min['response_time'] = df_5min['response_time'].ffill().fillna(df_5min['response_time'].median())

    df_5min["CPU"]=df_5min['CPU'].fillna(0)
    df_5min["response_time"]=df_5min['response_time'].fillna(0)
    
    # For categorical variables, use mode or 0
    # df_5min['status_code'] = df_5min['status_code'].ffill().fillna(200)
    df_5min['is_error'] = df_5min['is_error'].fillna(0)
    df_5min['Delay_Detected'] = df_5min['Delay_Detected'].fillna(0)
    
    # Create time-based features
    df_5min['hour'] = df_5min.index.hour
    df_5min['day_of_week'] = df_5min.index.dayofweek
    df_5min['is_weekend'] = df_5min['day_of_week'].isin([5, 6]).astype(int)
    # df_5min['is_peak_hour'] = df_5min['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)

    # Define peak hours (example: 9 AM - 6 PM)
    df_5min["is_peak_hour"] = df_5min["hour"].apply(lambda h: 1 if 9 <= h < 18 else 0)
    
    # Create lag features for system health (NOT delay detection)
    for lag in [1, 2, 3]:
        df_5min[f'cpu_lag_{lag}'] = df_5min['CPU'].shift(lag)
        df_5min[f'ram_lag_{lag}'] = df_5min['RAM'].shift(lag)
        df_5min[f'response_lag_{lag}'] = df_5min['response_time'].shift(lag)
    
    # Rolling statistics for system health
    df_5min['cpu_rolling_mean'] = df_5min['CPU'].rolling(window=6, min_periods=1).mean()
    df_5min['cpu_rolling_std'] = df_5min['CPU'].rolling(window=6, min_periods=1).std()
    df_5min['ram_rolling_mean'] = df_5min['RAM'].rolling(window=6, min_periods=1).mean()
    df_5min['response_rolling_mean'] = df_5min['response_time'].rolling(window=6, min_periods=1).mean()
    
    # System health indicators
    df_5min['high_cpu'] = (df_5min['CPU'] > df_5min['CPU'].quantile(0.8)).astype(int)
    df_5min['high_ram'] = (df_5min['RAM'] > df_5min['RAM'].quantile(0.8)).astype(int)
    df_5min['high_error_period'] = (df_5min['is_error'] > 0).astype(int)
    
    # Create proper target - predict delays 2 periods ahead WITHOUT using current delay
    # This removes data leakage completely
    df_5min['future_delay'] = df_5min['Delay_Detected'].shift(-1)
    df_5min['target'] = df_5min['future_delay'].fillna(0).astype(int)
    
    # Drop rows with NaN in lag features (first few rows)
    # df_5min = df_5min.dropna()
    
    print(f"After feature engineering: {df_5min.shape}")
    print(f"Target distribution: {df_5min['target'].value_counts().to_dict()}")
    print(f"Target balance: {df_5min['target'].mean()*100:.1f}% positive class")
    print(f"NaN values remaining: {df_5min.isnull().sum().sum()}")
    
    print("Preprocessing complete.")
    df_train = df_5min.iloc[:-10]
    df_test = df_5min.tail(10)
    print("train data ",df_train.tail(5))
    print("test_data",df_test.head())

    df_train.to_csv("./data/preprocessed_train.csv", index=False)
    df_test.to_csv("./data/preprocessed_test.csv",index=False)
    return df_train