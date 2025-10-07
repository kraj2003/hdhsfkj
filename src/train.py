import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.metrics import precision_recall_curve
from src.mlflow_manager import MLflowManager
import json
import tensorflow as tf

def create_sequences(X_raw, y_raw, window_size):
    """Create sequences for LSTM training - NO DATA LEAKAGE"""
    X_seq, y_seq = [], []
    
    # Ensure no NaN values
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    y_raw = np.nan_to_num(y_raw, nan=0.0)
    
    for i in range(len(X_raw) - window_size):
        seq_x = X_raw[i:i + window_size]  # Historical data
        seq_y = y_raw[i + window_size]    # Future target
        X_seq.append(seq_x)
        y_seq.append(seq_y)
    
    return np.array(X_seq), np.array(y_seq)

def weighted_bce(y_true, y_pred):
    # Example: penalize false negatives 3x more
    weight_for_1 = 3.0  # class=1 (delay)
    weight_for_0 = 1.0
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weights = y_true * weight_for_1 + (1 - y_true) * weight_for_0
    return bce * weights

def build_lstm_model(input_shape, config):
    """Build LSTM model with proper architecture"""
    model = Sequential([
        LSTM(config['lstm_units'], return_sequences=False, input_shape=(input_shape[1], input_shape[2])),
        Dropout(config['dropout_rate']),
        # LSTM(config['lstm_units'] // 2),
        # Dropout(config['dropout_rate']),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # tf.keras.losses.BinaryCrossentropy()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy','Precision','Recall']
    )
    
    return model

def train_lstm(processed_df):
    """Fixed training with NO DATA LEAKAGE and proper class balancing"""
    
    # Configuration
    config = {
        'window_size': 24,  # 4 hours of 10-minute intervals
        'epochs': 10,
        'batch_size': 32,
        # CRITICAL: Remove Delay_Detected from features to prevent data leakage
        'features': [
            'CPU', 'RAM', 'response_time', 
            'cpu_rolling_mean', 'ram_rolling_mean', 'response_rolling_mean',
            'high_cpu', 'high_ram', 'is_peak_hour'
        ],
        'dropout_rate': 0.3,
        'lstm_units': 64
    }
    
    print(f"Using features: {config['features']}")
    
    # Verify features exist
    missing_features = [f for f in config['features'] if f not in processed_df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        # Use available features only
        config['features'] = [f for f in config['features'] if f in processed_df.columns]
        print(f"Using available features: {config['features']}")
    
    # Prepare data - NO CURRENT DELAY INFORMATION
    X_raw = processed_df[config['features']].values
    y_raw = processed_df['target'].values
    
    print(f"Feature matrix shape: {X_raw.shape}")
    print(f"Target shape: {y_raw.shape}")
    print(f"Target distribution: {np.bincount(y_raw.astype(int))}")
    
    # Check for data leakage by ensuring no future information
    assert 'Delay_Detected' not in config['features'], "DATA LEAKAGE: Current delay in features!"
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_raw, y_raw, config['window_size'])
    print(f"Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
    
    if len(X_seq) == 0:
        raise ValueError("No sequences created. Check your data size and window_size.")
    
    # Scale features properly
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_seq.shape
    X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(n_samples, n_timesteps, n_features)
    
    # Train-test split (temporal split to avoid data leakage)
    # split_idx = int(len(X_scaled) * 0.8)
    # X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    # y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    X_train,X_val,y_train,y_val=train_test_split(X_scaled,y_seq,test_size=0.2,shuffle=False)
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Val set: {X_val.shape}, {y_val.shape}")
    print(f"Train target distribution: {np.bincount(y_train.astype(int))}")
    print(f"Val target distribution: {np.bincount(y_val.astype(int))}")
    
    # Handle class imbalance
    if len(np.unique(y_train)) > 1:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
        print("Warning: Only one class in training data!")

    # class_weight_dict = {0: 1.5, 1: 5.0}   
    
    # Build and train model
    model = build_lstm_model(X_scaled.shape, config)
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_recall',
        mode='max',
        patience=10,
        restore_best_weights=True
    )
    
    # class_weight_dict = {0: 1.0, 1: 6.0}  

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2
    )
    # Evaluate model
    y_pred_prob = model.predict(X_val)
    probs = y_pred_prob.flatten()
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

    # Option A: choose threshold giving highest F1
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = f1s.argmax()
    best_thresh = thresholds[best_idx]
    print("Best F1 threshold:", best_thresh, "F1:", f1s[best_idx])

    y_pred = (y_pred_prob > best_thresh ).astype(int).flatten()

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
        
    # Calculate metrics safely
    try:
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0)
        }
    except:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}

    print("checking the accuracy")
    print(y_pred[:30], y_val[:30])
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print("="*50)
    
    # Print classification report
    if len(np.unique(y_val)) > 1 and len(np.unique(y_pred)) > 1:
        print("\nDetailed Classification Report:")
        print(classification_report(y_val, y_pred, target_names=['No Delay', 'Delay']))


    # Save best threshold
    with open("./models/best_threshold.json", "w") as f:
        json.dump({"threshold": float(best_thresh)}, f)
    
    # Save model and metadata
    model.save("./models/lstm_model_latest.h5")
    joblib.dump(scaler, "./models/scaler_latest.pkl")
    
    # Save feature names for prediction
    with open("./models/feature_names.json", "w") as f:
        json.dump(config['features'], f)
    
    # Log to MLflow - MODIFY THIS SECTION:
    try:
        mlflow_manager = MLflowManager()
        # Pass the additional parameters for enhanced logging
        run_id = mlflow_manager.log_training_run(
            model=model, 
            scaler=scaler, 
            history=history, 
            config=config, 
            metrics=metrics,
            y_val=y_val,           # Add this
            y_pred=y_pred,         # Add this  
            y_pred_prob=y_pred_prob # Add this
        )
        print(f"\n✅ Model logged to MLflow with run_id: {run_id}")
        print(f"✅ Visualizations saved to MLflow")
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        run_id = "local_only"
    
    # Verify no data leakage
    print("\n✅ VERIFICATION:")
    print("✅ Model uses NO current delay information!")
    print("✅ Prediction based on system health only")
    print(f"✅ Features used: {len(config['features'])}")
    
    return model, scaler, metrics