from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append('/opt/airflow')

from src.data_manager import DataManager

# Simple DAG configuration
default_args = {
    'owner': 'ops-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

def check_data_health():
    """Simple data health check"""
    print("📊 Checking data health...")
    
    try:
        data_manager = DataManager()
        df = data_manager.load_data()
        
        if df.empty:
            print("❌ No data found")
            return "no_data"
        
        # Basic checks
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_time = df['timestamp'].max()
        hours_old = (datetime.now() - latest_time).total_seconds() / 3600
        
        if hours_old > 6:
            print(f"⚠️ Data is {hours_old:.1f} hours old")
            status = "stale_data"
        else:
            print(f"✅ Data is fresh ({hours_old:.1f} hours old)")
            status = "data_ok"
        
        print(f"Total records: {len(df)}")
        return status
        
    except Exception as e:
        print(f"❌ Data check failed: {e}")
        return "data_error"

def check_model_health():
    """Simple model health check"""
    print("🤖 Checking model health...")
    
    # Check if model files exist
    model_files = [
        'models/lstm_model_latest.h5',
        'models/scaler_latest.pkl',
        'feature_names.json'
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return "missing_files"
    
    # Check model age
    model_path = 'models/lstm_model_latest.h5'
    model_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).days
    
    if model_age > 35:
        print(f"⚠️ Model is {model_age} days old - consider retraining")
        status = "old_model"
    else:
        print(f"✅ Model is {model_age} days old")
        status = "model_ok"
    
    return status

def check_predictions():
    """Simple prediction check"""
    print("📈 Checking recent predictions...")
    
    try:
        # Check if predictions file exists
        if not os.path.exists('predictions.csv'):
            print("❌ No predictions file found")
            return "no_predictions"
        
        df = pd.read_csv('predictions.csv')
        
        if df.empty:
            print("❌ No predictions found")
            return "empty_predictions"
        
        # Check recent predictions
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent_predictions = df[df['timestamp'] >= datetime.now() - timedelta(days=1)]
            print(f"Predictions in last 24h: {len(recent_predictions)}")
            
            if len(recent_predictions) < 10:
                print("⚠️ Very few recent predictions")
                return "few_predictions"
        
        print("✅ Predictions look normal")
        return "predictions_ok"
        
    except Exception as e:
        print(f"❌ Prediction check failed: {e}")
        return "prediction_error"

def generate_simple_report():
    """Generate a simple daily report"""
    print("📋 Generating daily report...")
    
    # Get results from previous tasks
    # In a real implementation, you'd use xcom_pull here
    # For simplicity, we'll just create a basic report
    
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'status': 'Generated',
        'timestamp': datetime.now().isoformat()
    }
    
    # Save simple report
    import json
    os.makedirs('reports', exist_ok=True)
    report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report saved: {report_file}")
    return "report_generated"

# Create simple DAG
dag = DAG(
    'simple_daily_monitoring',
    default_args=default_args,
    description='Simple daily system monitoring',
    schedule_interval='0 8 * * *',  # 8 AM daily
    catchup=False,
    tags=['monitoring', 'simple']
)

# Simple tasks
data_check_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data_health,
    dag=dag,
)

model_check_task = PythonOperator(
    task_id='check_model',
    python_callable=check_model_health,
    dag=dag,
)

prediction_check_task = PythonOperator(
    task_id='check_predictions',
    python_callable=check_predictions,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_simple_report,
    dag=dag,
)

# Simple dependencies - run checks in parallel, then report
[data_check_task, model_check_task, prediction_check_task] >> report_task