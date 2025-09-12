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
from src.train import train_lstm

# Simple DAG configuration
default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

def load_and_train():
    """Simple training function"""
    print("🚀 Starting model training...")
    
    # Load data
    data_manager = DataManager()
    df = data_manager.load_data()
    print(f"Loaded {len(df)} records")
    
    # Check if we have enough data
    if len(df) < 1000:
        raise ValueError("Not enough data for training")
    
    # Preprocess
    processed_df = preprocessing_pipeline(df)
    print(f"Processed to {len(processed_df)} samples")
    
    # Train model
    model, scaler, metrics = train_lstm(processed_df)
    
    # Simple validation
    if metrics['f1_score'] < 0.1:
        print("⚠️ Warning: Low model performance")
    
    print(f"✅ Training complete - F1: {metrics['f1_score']:.3f}")
    return "training_successful"

def backup_old_model():
    """Simple model backup"""
    print("💾 Backing up old model...")
    
    import shutil
    timestamp = datetime.now().strftime('%Y%m%d')
    
    # Backup if exists
    if os.path.exists('models/lstm_model_latest.h5'):
        shutil.copy('models/lstm_model_latest.h5', f'models/backup_model_{timestamp}.h5')
        print("✅ Model backed up")
    else:
        print("No existing model to backup")

# Create simple DAG
dag = DAG(
    'simple_monthly_training',
    default_args=default_args,
    description='Simple monthly model training',
    schedule_interval='0 2 1 * *',  # 2 AM on 1st of month
    catchup=False,
    tags=['training', 'simple']
)

# Simple tasks
backup_task = PythonOperator(
    task_id='backup_model',
    python_callable=backup_old_model,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=load_and_train,
    dag=dag,
)

# Simple dependency
backup_task >> train_task