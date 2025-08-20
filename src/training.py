import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv('MergedDF_192.168.11.193.csv')
df = pd.DataFrame(data)


def preprocessing_pipeline(df):
    print('Starting preprocessing...')
    df['Delay_Detected'] = (df['time_taken']>2000).astype(int)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    df['minute'] = df['datetime'].dt.minute      # Extract minute (0-59)
    df['second'] = df['datetime'].dt.second      # Extract second (0-59)
    df['day'] = df['datetime'].dt.day            # Extract day (1-31)
    df['month'] = df['datetime'].dt.month        # Extract month (1-12)
    df['year'] = df['datetime'].dt.year          # Extract year (e.g., 2023)
    print('Resampling...')
    # Resample to 5-minute intervals
    df_5min = (
        df.set_index('datetime')
        .resample('5T')
        .agg({
            'Delay_Detected': 'sum',  # Count delays per window
            'time_taken': 'mean',     # Avg response time
            'CPU': 'mean',            # Avg CPU usage
            'RAM': 'mean',            # Avg RAM usage
            'sc_status': 'mean',      # Avg HTTP status
            'is_error': 'sum'         # Total errors
        })
    )

    # for lag in [6, 12, 18, 24]:  # 30 mins, 1 hour, 1.5 hours, 2 hours
    #     df_5min[f'delay_lag_{lag}'] = df_5min['Delay_Detected'].shift(lag)
    print('Creating lags...')
    for lag in [2, 4, 6, 12]:  # 10 min, 20 min, 30 min, 1 hour
        df_5min[f'delay_lag_{lag}'] = df_5min['Delay_Detected'].shift(lag)


    df_5min['delay_rolling_avg_1h'] = df_5min['Delay_Detected'].rolling(12).mean()  # 1-hour avg
    df_5min['delay_rolling_std_3h'] = df_5min['Delay_Detected'].rolling(36).std()   # 3-hour volatility

    df_5min['minute'] = df_5min.index.minute
    df_5min['hour'] = df_5min.index.hour
    df_5min['day_of_week'] = df_5min.index.dayofweek  # 0=Monday
    df_5min['is_weekend'] = df_5min['day_of_week'].isin([5, 6]).astype(int)

    df_5min['high_error_period'] = (df_5min['is_error'] > df_5min['is_error'].quantile(0.9)).astype(int)
    print('Preprocessing over and out...')
    return df_5min


#
def lstm(df_5min):
    # ----------------------
    # 1. Feature Selection
    # ----------------------
    features = ['Delay_Detected', 'CPU', 'RAM']
    target = 'Delay_Detected'

    # Define binary target: will there be a delay in next 5 minutes?
    # df_5min['target'] = df_5min['Delay_Detected'].shift(-1).fillna(0).astype(int)
    # df_5min['target'] = (df_5min['Delay_Detected'].shift(-6).fillna(0) > 0).astype(int)
    df_5min['target'] = (df_5min['Delay_Detected'].shift(-2).fillna(0) > 0).astype(int)


    print('Creating raw faetures...')
    X_raw = df_5min[features].values
    y_raw = df_5min['target'].values

    # ----------------------
    # 2. Create Sequences
    # ----------------------
    window_size = 24  # Last 1 hour of data (12 x 5 min)
    X_seq, y_seq = [], []

    for i in range(len(X_raw) - window_size):
        seq_x = X_raw[i:i + window_size]
        seq_y = y_raw[i + window_size]
        X_seq.append(seq_x)
        y_seq.append(seq_y)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # ----------------------
    # 3. Scale Input Features
    # ----------------------
    X_seq = np.nan_to_num(X_seq)

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
    
    print('Train test split...')
    # ----------------------
    # 4. Train-Test Split
    # ----------------------
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_seq, test_size=0.2, shuffle=False)
    
    print('model creation...')
    # ----------------------
    # 5. LSTM Classification Model
    # ----------------------
    model = Sequential([
        LSTM(64, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
        Dense(1, activation='sigmoid')  # Binary output
    ])
    
    print('Model compilation...')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ----------------------
    # 6. Train the Model
    # ----------------------
    print('Model fitting...')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    print('Model Evaluation...')
    # ----------------------
    # 7. Evaluation
    # ----------------------
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # ----------------------
    # 8. Prediction for Latest Window
    # ----------------------
    latest_seq = X_raw[-window_size:]
    latest_scaled = x_scaler.transform(latest_seq).reshape(1, window_size, len(features))

    future_prob = model.predict(latest_scaled)[0][0]
    future_class = int(future_prob > 0.5)

    # print(f"\nPredicted delay in next 30 minutes? {'Yes' if future_class == 1 else 'No'} (Probability: {future_prob:.2f})")
    print(f"\nPredicted delay in next 10 minutes? {'Yes' if future_class == 1 else 'No'} (Probability: {future_prob:.2f})")

    model.save("my_lstm_model.h5")  # or .keras for the new format
    print('Model Saved...')

processed_df = preprocessing_pipeline(df)
lstm(processed_df)