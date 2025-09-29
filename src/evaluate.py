# src/evaluate.py - FIXED VERSION
import numpy as np
import json
from src import config

def predict_future(model, scaler, df, features, window_size):
    """Fixed prediction function with proper feature handling"""
    
    # Load the actual features used during training
    try:
        with open("./models/feature_names.json", "r") as f:
            trained_features = json.load(f)
        print(f"Using trained features: {trained_features}")
        features = trained_features
    except FileNotFoundError:
        print(f"Warning: Using provided features: {features}")
    
    # Verify features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in data: {missing_features}")
        print(f"Available columns: {list(df.columns)}")
        # Use only available features
        features = [f for f in features if f in df.columns]
        print(f"Using available features: {features}")
    
    if len(features) == 0:
        raise ValueError("No valid features available for prediction!")
    
    # Get latest window of data (ensure enough data points)
    if len(df) < window_size:
        print(f"Warning: Not enough data points ({len(df)} < {window_size})")
        # Pad with the last available row
        if len(df) > 0:
            last_row = df[features].iloc[-1:].values
            padding_needed = window_size - len(df)
            padding = np.tile(last_row, (padding_needed, 1))
            latest_seq = np.vstack([padding, df[features].values])
        else:
            # No data available - use zeros
            latest_seq = np.zeros((window_size, len(features)))
    else:
        latest_seq = df[features].tail(window_size).values

    
    # Handle NaN values
    latest_seq = np.nan_to_num(latest_seq, nan=0.0)
    
    print(f"Input sequence shape: {latest_seq.shape}")
    print(f"Sample values: {latest_seq[-1]}")  # Show last row
    
    # Scale the data
    try:
        # Reshape for scaling, then back to sequence format
        n_timesteps, n_features = latest_seq.shape
        latest_scaled = scaler.transform(latest_seq.reshape(-1, n_features))
        latest_scaled = latest_scaled.reshape(1, n_timesteps, n_features)
    except Exception as e:
        print(f"Scaling error: {e}")
        print(f"Scaler expects {scaler.n_features_in_} features, got {latest_seq.shape[1]}")
        raise
    
    # Make prediction
    try:
        future_prob = model.predict(latest_scaled, verbose=0)[0][0]
        probs=future_prob.flatten()
        
        with open("./models/best_threshold.json", "r") as f:
            best_thresh = json.load(f)["threshold"]

        future_class = (probs >0.3).astype(int).flatten()
        print(future_class)
        # future_class = int(future_prob > 0.3)
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Model input shape expected: {model.input_shape}")
        print(f"Actual input shape: {latest_scaled.shape}")
        raise
    
    # Display result
    result_text = "Yes" if future_class else "No"
    confidence = "High" if abs(future_prob - 0.7) > 0.3 else "Medium" if abs(future_prob - 0.5) > 0.1 else "Low"
    
    # This measures how far the probability is from the “uncertain zone”.
    print(f"\nPredicted delay in next 10 minutes? {result_text} (Prob: {future_prob:.3f}, Confidence: {confidence})")
    
    return future_class, future_prob