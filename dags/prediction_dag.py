from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# Add your project root to Python path
sys.path.append('/opt/airflow')  # This will be your project root in Airflow

# Simple DAG configuration
default_args = {
    'owner': 'prediction-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email': ['khuhsirajpurohit.com'],
    'email_on_failure': False,
    'email_on_retry': False,
}

def make_5min_prediction():
    """Make a prediction every 5 minutes using your existing code structure"""
    print("🔮 Making 5-minute delay prediction...")
    
    try:
        # Import your modules (same as in main.py)
        from src.data_manager import DataManager
        from src.preprocessing import preprocessing_pipeline
        from src.evaluate import predict_future
        from src import config
        from tensorflow.keras.models import load_model
        import joblib
        import json
        
        print("✅ Successfully imported all modules")
        
        # Check if model files exist (your actual file locations)
        model_path = "./models/lstm_model_latest.h5"
        scaler_path = "./models/scaler_latest.pkl"
        features_path = "./models/feature_names.json"
        
        
        # Alternative paths if above don't exist
        if not os.path.exists(model_path):
            model_path = "./models/lstm_model_latest.h5"  # Your original model name
            
        if not os.path.exists(scaler_path):
            scaler_path = "./models/scaler.pkl"  # Alternative scaler name
        
        missing_files = []
        for path, name in [(model_path, 'Model'), (scaler_path, 'Scaler'), (features_path, 'Features')]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return "missing_model_files"
        
        print(f"✅ Found all required files")
        
        # Load your data (exactly like main.py)
        print("📊 Loading data...")
        data_manager = DataManager()
        df = data_manager.load_data()
        print(f"✅ Loaded {len(df)} records")
        
        # Preprocess data (exactly like main.py)
        print("🔄 Preprocessing data...")
        processed_df = preprocessing_pipeline(df)
        print(f"✅ Processed to {len(processed_df)} samples")
        
        # Load model and scaler
        print("🤖 Loading model and scaler...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load features
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        print(f"✅ Model loaded with {len(features)} features")
        
        # Filter last 6 hours (like main.py)
        if 'timestamp' in processed_df.columns:
            last_6_hours_df = processed_df[
                processed_df['timestamp'] >= (processed_df['timestamp'].max() - pd.Timedelta(hours=2))
            ]
        else:
            # If timestamp is index
            last_6_hours_df = processed_df.tail(24)  # Approximate 6 hours of 5min intervals
        
        print(f"📈 Using {len(last_6_hours_df)} recent records")
        
        # Make prediction (exactly like main.py)
        print("🎯 Making prediction...")
        prediction_class, prediction_prob = predict_future(
            model, scaler, last_6_hours_df, features, config.WINDOW_SIZE
        )
        
        # Determine confidence
        confidence = (
            "High" if abs(prediction_prob - 0.5) > 0.3 
            else "Medium" if abs(prediction_prob - 0.5) > 0.1 
            else "Low"
        )
        
        # predicted_time = last_timestamp_in_data + timedelta(minutes=10)
        # Log results
        print(f"📊 Prediction Results:")
        print(f"   - Delay Expected: {'Yes' if prediction_class else 'No'}")
        print(f"   - Probability: {prediction_prob:.3f}")
        print(f"   - Confidence: {confidence}")
        print(f"   - Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save results (like main.py)
        results_df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'prediction': [prediction_class],
            'probability': [prediction_prob],
            'confidence': [confidence]
        })
        data_manager.save_results(results_df)
        
        # Alert if delay predicted
        if prediction_class == 1:
            print(f"🚨 DELAY ALERT: {prediction_prob:.3f} probability")
        else:
            print(f"✅ No delay expected: {prediction_prob:.3f} probability")
        
        return f"prediction_success_{prediction_class}_{prediction_prob:.3f}"
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        # Print more details for debugging
        import traceback
        traceback.print_exc()
        raise

def quick_health_check():
    """Quick check before making prediction"""
    print("🔍 Quick health check...")
    
    try:
        # Check if data file exists
        data_file = "./data/raw_data.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file missing: {data_file}")
            return "no_data_file"
        
        # Quick data load test
        df = pd.read_csv(data_file)
        if len(df) < 100:
            print(f"❌ Insufficient data: {len(df)} records")
            return "insufficient_data"
        
        print(f"✅ Data file OK: {len(df)} records")
        
        # Check model files
        model_files = [
            "./models/lstm_model_latest.h5",  # Alternative
            "./models/scaler_latest.pkl",
            "./feature_names.json"
        ]
        
        found_model = False
        for model_file in model_files[:2]:  # Check model files
            if os.path.exists(model_file):
                found_model = True
                print(f"✅ Found model: {model_file}")
                break
        
        if not found_model:
            print("❌ No model file found")
            return "no_model"
        
        print("✅ Health check passed")
        return "health_ok"
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return "health_check_failed"

# Create the DAG
dag = DAG(
    'delay_prediction_every_5min',
    default_args=default_args,
    description='Make delay prediction every 5 minutes',
    # schedule_interval='*/5 * * * *',  # Every 5 minutes
    schedule_interval='0 * * * *',  # Every hour at minute 0
    catchup=False,
    tags=['prediction', '5min', 'real-time'],
    max_active_runs=1,
)

# Define tasks
health_check_task = PythonOperator(
    task_id='quick_health_check',
    python_callable=quick_health_check,
    dag=dag,
)

prediction_task = PythonOperator(
    task_id='make_5min_prediction',
    python_callable=make_5min_prediction,
    dag=dag,
)

# Set task dependencies
health_check_task >> prediction_task