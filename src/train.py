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

def train_lstm(df):
    features = ['Delay_Detected', 'CPU', 'RAM']
    target = 'Delay_Detected'

    # Target: delay in next 10 min (2 steps ahead)
    df['target'] = (df['Delay_Detected'].shift(-2).fillna(0) > 0).astype(int)

    X_raw = df[features].values
    y_raw = df['target'].values

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
    X_scaled = scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_seq, test_size=0.2, shuffle=False)

    # Model
    model = Sequential([
        LSTM(64, input_shape=(X_scaled.shape[1], X_scaled.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

    # Evaluation
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    with mlflow.start_run():
        mlflow.log_param("window_size", config.WINDOW_SIZE)
        mlflow.log_param("epochs", config.EPOCHS)
        mlflow.log_param("batch_size", config.BATCH_SIZE)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

        val_acc = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_acc)

        # Save model to MLflow
        mlflow.keras.log_model(model, "lstm_model")

    # Save model & scaler
    model.save(config.MODEL_PATH)
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Model saved to {config.MODEL_PATH}")

    return model, scaler, history
