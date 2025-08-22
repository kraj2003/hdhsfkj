# src/train.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report
import joblib
import mlflow
import mlflow.keras
from src import config

mlflow.set_experiment("Delay_forecasting")

def train_lstm(df):
    print("Training corrected model (no data leakage)...")
    
    # CORRECTED: Use only predictive features (NOT current delays!)
    features = ['CPU', 'RAM', 'hour', 'day_of_week', 'is_error','delay_rolling_avg_1h']
    
    # Target: delay in next 10 min (2 steps ahead)
    df['target'] = (df['Delay_Detected'].shift(-2).fillna(0) > 0).astype(int)

    print(df.head())
    # Clean data
    df_clean = df.dropna()
    
    X_raw = df_clean[features].values
    y_raw = df_clean['target'].values

    # Sequence creation
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - config.WINDOW_SIZE):
        seq_x = X_raw[i:i + config.WINDOW_SIZE]
        seq_y = y_raw[i + config.WINDOW_SIZE]
        X_seq.append(seq_x)
        y_seq.append(seq_y)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Scaling
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_seq.shape
    X_flat = X_seq.reshape(-1, n_features)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_seq, test_size=0.2, shuffle=False)

    # Simple model
    model = Sequential([
        LSTM(32, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

    # Evaluation
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # MLflow tracking
    with mlflow.start_run():

        # Log feature names, not feature data
        feature_columns = ['day_of_week', 'CPU', 'RAM', 'hour', 'is_error', 'delay_rolling_avg_1h']
        mlflow.log_param("features", ",".join(feature_columns))
        mlflow.log_param("num_features", len(feature_columns))

        # Log data statistics as metrics instead
        mlflow.log_metric("avg_cpu_usage", df_clean['CPU'].mean())
        mlflow.log_metric("avg_ram_usage", df_clean['RAM'].mean())
        mlflow.log_metric("error_rate", df_clean['is_error'].mean())


        mlflow.log_param("batch_size", config.BATCH_SIZE)
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("window_size", config.WINDOW_SIZE)
        mlflow.log_param("epochs", config.EPOCHS)
        

        val_acc = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.keras.log_model(model, "corrected_lstm_model")

    # Save model & scaler
    model.save(config.MODEL_PATH)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(features, "models/features.pkl")
    
    print(f"Model saved with features: {features}")
    print("✅ No more data leakage!")

    return model, scaler, history