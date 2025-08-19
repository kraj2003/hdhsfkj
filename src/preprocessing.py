# src/preprocessing.py
import pandas as pd
from src import config

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    df['Delay_Detected'] = (df['time_taken'] > 2000).astype(int)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['datetime'].dt.date)
    df['minute'] = df['datetime'].dt.minute
    df['second'] = df['datetime'].dt.second
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    # Resample to fixed intervals
    df_5min = (
        df.set_index('datetime')
        .resample(config.RESAMPLE_FREQ)
        .agg({
            'Delay_Detected': 'sum',
            'time_taken': 'mean',
            'CPU': 'mean',
            'RAM': 'mean',
            'sc_status': 'mean',
            'is_error': 'sum'
        })
    )

    # Lags
    for lag in config.LAGS:
        df_5min[f'delay_lag_{lag}'] = df_5min['Delay_Detected'].shift(lag)

    # Rolling features
    df_5min['delay_rolling_avg_1h'] = df_5min['Delay_Detected'].rolling(config.ROLLING_AVG_WINDOW).mean()
    df_5min['delay_rolling_std_3h'] = df_5min['Delay_Detected'].rolling(config.ROLLING_STD_WINDOW).std()

    # Time features
    df_5min['minute'] = df_5min.index.minute
    df_5min['hour'] = df_5min.index.hour
    df_5min['day_of_week'] = df_5min.index.dayofweek
    df_5min['is_weekend'] = df_5min['day_of_week'].isin([5, 6]).astype(int)

    # High error indicator
    df_5min['high_error_period'] = (df_5min['is_error'] > df_5min['is_error'].quantile(0.9)).astype(int)

    print("Preprocessing complete.")
    return df_5min
