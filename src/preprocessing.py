# src/preprocessing.py
import pandas as pd
from src import config

def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    df['Delay_Detected'] = (df['response_time'] > 2000).astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['timestamp'].dt.date)
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    print("preprocessing",df[['CPU','RAM']])

    # Resample to fixed intervals
    df_5min = (
        df.set_index('timestamp')
        .resample(config.RESAMPLE_FREQ)
        .agg({
            'Delay_Detected': 'sum',
            'response_time': 'mean',
            'CPU': 'mean',
            'RAM': 'mean',
            'status_code': 'mean',
            'is_error': 'sum'
        })
    )
    print(df_5min.head())
    print("shape",df_5min.shape)
    print(df_5min['CPU'].isna().sum())
    
    df_5min['RAM'] = df_5min['RAM'].fillna(method='ffill')
    df_5min["CPU"]=df_5min['CPU'].fillna(0)
    df_5min["time_taken"]=df_5min['response_time'].fillna(0)

    print(df_5min['CPU'].isna().sum())

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
