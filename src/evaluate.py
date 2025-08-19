# src/evaluate.py
import numpy as np

def predict_future(model, scaler, df, features, window_size):
    latest_seq = df[features].values[-window_size:]
    latest_scaled = scaler.transform(latest_seq).reshape(1, window_size, len(features))

    future_prob = model.predict(latest_scaled)[0][0]
    future_class = int(future_prob > 0.5)

    print(f"Predicted delay in next 10 minutes? {'Yes' if future_class else 'No'} (Prob: {future_prob:.2f})")
    return future_class, future_prob
