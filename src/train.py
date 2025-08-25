# src/train_enhanced.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from src.mlflow_manager import MLflowManager
from src.data_manager import DataManager
from src.preprocessing import preprocessing_pipeline

def train_lstm():
    """Enhanced training with proper MLflow logging"""
    
    # Configuration
    config = {
        'window_size': 24,
        'epochs': 50,  # Increased for better training
        'batch_size': 32,
        'features': ['Delay_Detected', 'CPU', 'RAM', 'time_taken'],
        'dropout_rate': 0.2,
        'lstm_units': 64
    }
    
    # Load and preprocess data
    data_manager = DataManager()
    df = data_manager.load_data()
    processed_df = preprocessing_pipeline(df)
    
    # Create target variable
    processed_df['target'] = (processed_df['Delay_Detected'].shift(-2).fillna(0) > 0).astype(int)
    
    # Feature engineering
    X_raw = processed_df[config['features']].values
    y_raw = processed_df['target'].values
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_raw, y_raw, config['window_size'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scale_sequences(X_seq, scaler)
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_seq, test_size=0.2, shuffle=False
    )
    
    # Build model
    model = build_lstm_model(X_scaled.shape, config)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1
    )
    
    # Evaluate model
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    
    # Log to MLflow
    mlflow_manager = MLflowManager()
    run_id = mlflow_manager.log_training_run(model, scaler, history, config, metrics)
    
    # Save model locally
    model.save("models/lstm_model_latest.h5")
    joblib.dump(scaler, "models/scaler_latest.pkl")
    
    print(f"\nModel logged to MLflow with run_id: {run_id}")
    return model, scaler, metrics

def create_sequences(X_raw, y_raw, window_size):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    
    # Handle NaN values
    X_raw = np.nan_to_num(X_raw)
    y_raw = np.nan_to_num(y_raw)
    
    for i in range(len(X_raw) - window_size):
        seq_x = X_raw[i:i + window_size]  # Get window_size time steps
        seq_y = y_raw[i + window_size]    # Get target at next time step
        X_seq.append(seq_x)
        y_seq.append(seq_y)
    
    return np.array(X_seq), np.array(y_seq)

def scale_sequences(X_seq, scaler):
    """Scale sequence data properly for LSTM"""
    # X_seq shape: (samples, timesteps, features)
    n_samples, n_timesteps, n_features = X_seq.shape
    
    # Reshape to 2D for scaling: (samples * timesteps, features)
    X_flat = X_seq.reshape(-1, n_features)
    
    # Fit scaler and transform
    X_scaled_flat = scaler.fit_transform(X_flat)
    
    # Reshape back to 3D: (samples, timesteps, features)
    X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
    
    return X_scaled

def build_lstm_model(input_shape, config):
    """Build LSTM model with dropout"""
    model = Sequential([
        LSTM(config['lstm_units'], input_shape=(input_shape[1], input_shape[2])),
        Dropout(config['dropout_rate']),
        Dense(32, activation='relu'),
        Dropout(config['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model