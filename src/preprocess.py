# src/preprocessing.py - MODIFIED FOR TWO DATAFRAMES
import pandas as pd
import numpy as np
from src import config

def preprocessing_pipeline(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    Modified preprocessing pipeline for two dataframes
    df1: Training data (31 lakh records, ends 2025-05-22)
    df2: Validation data (2 lakh records, starts 2025-06-18)
    """
    print("Starting preprocessing for two dataframes...")
    
    # Process df1 (training data)
    print("\n=== Processing Training Data (df1) ===")
    df1_processed = process_single_dataframe(df1, "df1")
    
    # Process df2 (validation data)  
    print("\n=== Processing Validation Data (df2) ===")
    df2_processed = process_single_dataframe(df2, "df2")
    
    # Save processed data
    df1_processed.to_csv("./data/preprocessed_train.csv", index=False)
    df2_processed.to_csv("./data/preprocessed_test.csv", index=False)
    
    print(f"\n=== Summary ===")
    print(f"Training data: {df1_processed.shape}")
    print(f"Validation data: {df2_processed.shape}")
    print(f"Training target balance: {df1_processed['target'].mean()*100:.1f}% positive class")
    print(f"Validation target balance: {df2_processed['target'].mean()*100:.1f}% positive class")
    
    return df1_processed, df2_processed

def process_single_dataframe(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Process a single dataframe with the same logic as before"""
    
    print(f"Processing {df_name}...")
    
    # Create delay detection based on response time threshold
    df['Delay_Detected'] = (df['response_time'] > 2000).astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Original {df_name} shape: {df.shape}")
    print(f"Delay instances in {df_name}: {df['Delay_Detected'].sum()}/{len(df)} ({df['Delay_Detected'].mean()*100:.1f}%)")
    
    # Resample to fixed intervals with proper aggregation
    df_resampled = (
        df.set_index('timestamp')
        .resample(config.RESAMPLE_FREQ)
        .agg({
            'Delay_Detected': 'max',  # Any delay in window = 1
            'response_time': 'mean',
            'CPU': 'mean',
            'RAM': 'mean', 
            'is_error': 'max'
        })
    )
    
    print(f"After resampling {df_name}: {df_resampled.shape}")
    
    # Handle NaN values
    df_resampled['RAM'] = df_resampled['RAM'].ffill().fillna(df_resampled['RAM'].median())
    df_resampled["CPU"] = df_resampled['CPU'].fillna(0)
    df_resampled["response_time"] = df_resampled['response_time'].fillna(0)
    df_resampled['is_error'] = df_resampled['is_error'].fillna(0)
    df_resampled['Delay_Detected'] = df_resampled['Delay_Detected'].fillna(0)
    
    # Create time-based features
    df_resampled['hour'] = df_resampled.index.hour
    df_resampled['day_of_week'] = df_resampled.index.dayofweek
    df_resampled['is_weekend'] = df_resampled['day_of_week'].isin([5, 6]).astype(int)
    df_resampled["is_peak_hour"] = df_resampled["hour"].apply(lambda h: 1 if 9 <= h < 18 else 0)
    
    # Create lag features for system health
    for lag in [1, 2, 3]:
        df_resampled[f'cpu_lag_{lag}'] = df_resampled['CPU'].shift(lag)
        df_resampled[f'ram_lag_{lag}'] = df_resampled['RAM'].shift(lag)
        df_resampled[f'response_lag_{lag}'] = df_resampled['response_time'].shift(lag)
    
    # Rolling statistics for system health
    df_resampled['cpu_rolling_mean'] = df_resampled['CPU'].rolling(window=6, min_periods=1).mean()
    df_resampled['cpu_rolling_std'] = df_resampled['CPU'].rolling(window=6, min_periods=1).std()
    df_resampled['ram_rolling_mean'] = df_resampled['RAM'].rolling(window=6, min_periods=1).mean()
    df_resampled['response_rolling_mean'] = df_resampled['response_time'].rolling(window=6, min_periods=1).mean()
    
    # System health indicators
    df_resampled['high_cpu'] = (df_resampled['CPU'] > df_resampled['CPU'].quantile(0.8)).astype(int)
    df_resampled['high_ram'] = (df_resampled['RAM'] > df_resampled['RAM'].quantile(0.8)).astype(int)
    df_resampled['high_error_period'] = (df_resampled['is_error'] > 0).astype(int)
    
    # Create target - predict delays 2 periods ahead
    df_resampled['future_delay'] = df_resampled['Delay_Detected'].shift(-2)
    df_resampled['target'] = df_resampled['future_delay'].fillna(0).astype(int)
    
    print(f"After feature engineering {df_name}: {df_resampled.shape}")
    print(f"Target distribution in {df_name}: {df_resampled['target'].value_counts().to_dict()}")
    print(f"NaN values remaining in {df_name}: {df_resampled.isnull().sum().sum()}")
    
    return df_resampled

# Keep backward compatibility - if only one dataframe is passed
def preprocessing_pipeline_single(df: pd.DataFrame) -> pd.DataFrame:
    """Original function for single dataframe (backward compatibility)"""
    print("Using single dataframe preprocessing...")
    df_processed = process_single_dataframe(df, "single_df")
    
    df_train = df_processed
    df_test = df_processed.tail(5)
    
    df_train.to_csv("./data/preprocessed_train.csv", index=False)
    df_test.to_csv("./data/preprocessed_test.csv", index=False)
    
    return df_train