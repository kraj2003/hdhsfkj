from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append('/opt/airflow')

from src.data_manager import DataManager
from src.preprocessing import preprocessing_pipeline
from src.evaluate import predict_future

# Simple DAG configuration
default_args = {
    'owner': 'ops-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def make_prediction():
    """Simple prediction function"""
    print("🔮 Making delay prediction...")
    
    try:
        # Check if model exists
        if not os.path.exists('models/lstm_model_latest.h5'):
            print("❌ No trained model found")
            return "no_model"
        
        # Load recent data
        data_manager = DataManager()
        df = data_manager.load_data()
        
        # Get last few hours of data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_cutoff = df['timestamp'].max() - timedelta(hours=3)
        recent_df = df[df['timestamp'] >= recent_cutoff].copy()
        
        if len(recent_df) < 10:
            print("❌ Not enough recent data")
            return "insufficient_data"
        
        # Simple preprocessing
        processed_df = preprocessing_pipeline(recent_df)
        
        # Make prediction
        prediction, probability = predict_future(df=processed_df)
        
        # Save result
        result = pd.DataFrame({
            'timestamp': [datetime.now()],
            'prediction': [prediction],
            'probability': [probability]
        })
        data_manager.save_results(result)
        
        # Simple alert
        if prediction == 1:
            print(f"🚨 DELAY PREDICTED - Probability: {probability:.3f}")
        else:
            print(f"✅ No delay expected - Probability: {probability:.3f}")
        
        return f"prediction_{prediction}_prob_{probability:.3f}"
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return "prediction_failed"

def check_system():
    """Simple system check"""
    print("🔍 Quick system check...")
    
    # Check disk space
    import shutil
    disk_usage = shutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < 1:  # Less than 1GB free
        print(f"⚠️ Low disk space: {free_gb:.1f}GB")
    else:
        print(f"✅ Disk space OK: {free_gb:.1f}GB")
    
    # Check if API is running (simple)
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print(f"⚠️ API status: {response.status_code}")
    except:
        print("⚠️ API not reachable")
    
    return "system_checked"

# Create simple DAG
dag = DAG(
    'simple_hourly_predictions',
    default_args=default_args,
    description='Simple hourly predictions',
    schedule_interval='0 * * * *',  # Every hour
    catchup=False,
    tags=['predictions', 'simple']
)

# Simple tasks
system_check_task = PythonOperator(
    task_id='check_system',
    python_callable=check_system,
    dag=dag,
)

prediction_task = PythonOperator(
    task_id='make_prediction',
    python_callable=make_prediction,
    dag=dag,
)

# Simple dependency
system_check_task >> prediction_task