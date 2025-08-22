# src/evaluate.py
import numpy as np

def predict_future(model, scaler, df, features, window_size):
    """Simple prediction function - no data leakage"""
    
    # Get latest window of data
    latest_seq = df[features].tail(window_size).values
    latest_seq = np.nan_to_num(latest_seq, nan=0.0)  # Handle NaN
    
    # Scale and reshape for LSTM
    latest_scaled = scaler.transform(latest_seq.reshape(-1, len(features))).reshape(1, window_size, len(features))

    # Predict
    future_prob = model.predict(latest_scaled, verbose=0)[0][0]
    future_class = int(future_prob > 0.5)

    print(f"Predicted delay in next 10 minutes? {'Yes' if future_class else 'No'} (Prob: {future_prob:.3f})")
    
    return future_class, future_prob