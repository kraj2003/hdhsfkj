from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # validates input/output data automatically
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import logging
import uvicorn
from contextlib import asynccontextmanager
from src.data_manager import DataManager
from src.preprocessing import preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
scaler = None
data_manager = None

# Lifespan handler (runs at startup & shutdown)

# Loads the saved LSTM model from an H5 file , scaler . initalize data manager  , logs sucess and failure
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, data_manager
    try:
        model = load_model("./models/lstm_model_latest.h5")
        scaler = joblib.load("./models/scaler_latest.pkl")
        data_manager = DataManager()
        logger.info("✅ Model and scaler loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

    yield   # app runs while inside this block

    # Optional cleanup (shutdown)
    logger.info("🛑 API shutting down...")

app = FastAPI(title="Delay Prediction API", version="1.0.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    hours_back: int = 2

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    timestamp: str
    confidence: str

@app.get("/")
def home():
    return {"message": "Delay Forecasting API", "status": "running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = data_manager.load_data()
        processed_df = preprocessing_pipeline(df)

        features = ['CPU', 'RAM', 'response_time', 'is_error',
                    'cpu_lag_1', 'ram_lag_1', 'response_lag_1',
                    'cpu_rolling_mean', 'ram_rolling_mean', 'response_rolling_mean',
                    'high_cpu', 'high_ram', 'is_peak_hour', 'hour']
        latest_data = processed_df[features].tail(24).values

        # Checks if we have enough data points for the model
        if len(latest_data) < 24:
            # Calculates how many additional data points are needed
            padding_needed = 24 - len(latest_data)
            if len(latest_data) > 0:
                last_row = latest_data[-1]
                padding = np.tile(last_row, (padding_needed, 1))
                latest_data = np.vstack([padding, latest_data])
            else:
                latest_data = np.zeros((24, len(features)))

        latest_scaled = scaler.transform(latest_data.reshape(-1, len(features)))
        latest_scaled = latest_scaled.reshape(1, 24, len(features))

        probability = float(model.predict(latest_scaled)[0][0])
        prediction = int(probability > 0.5)

        confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.1 else "Low"

        # ✅ Monitoring metrics
        monitoring_metrics = monitor_predictions(
            pred_probs=[probability], 
            input_features=processed_df[["CPU", "RAM"]].tail(100)  # last 100 points for drift detection
        )
        logger.info(f"Monitoring: {monitoring_metrics}")

        logger.info(f"Prediction made: {prediction} (prob: {probability:.3f})")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.now().isoformat(),
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def monitor_predictions(pred_probs, input_features):
    metrics = {}
    # Std dev of prediction probabilities
    metrics['pred_prob_std'] = float(np.std(pred_probs))
    
    # Std dev of input features (CPU, RAM, etc.)
    metrics['cpu_std'] = float(np.std(input_features['CPU']))
    metrics['ram_std'] = float(np.std(input_features['RAM']))
    
    # Rolling std (e.g., last 50 predictions)
    if len(pred_probs) >= 50:
        metrics['rolling_pred_std'] = float(np.std(pred_probs[-50:]))
    
    return metrics

def add_health_endpoints(app: FastAPI):
    """Add comprehensive health check endpoints"""
    
    @app.get("/health")
    async def health_check():
        """Basic health check"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
