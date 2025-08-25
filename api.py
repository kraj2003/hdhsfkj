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
        model = load_model("models/lstm_model_latest.h5")
        scaler = joblib.load("models/scaler_latest.pkl")
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

@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = data_manager.load_data()
        processed_df = preprocessing_pipeline(df)

        features = ['Delay_Detected', 'CPU', 'RAM', 'time_taken']
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
