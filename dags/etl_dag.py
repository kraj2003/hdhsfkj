from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
from src.data_manager import DataManager

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),  # Better than hardcoded date
    'email_on_failure': False,  # Set to False for testing
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

def extract_from_endpoints(**context):
    """Extract data from endpoints or use CSV as fallback"""
    endpoints = [
        # Add your actual endpoints here when ready
        # 'http://your-app.com/api/metrics',
        # 'http://your-server.com/health',
    ]

    all_data = []

    # If no endpoints, simulate data or use existing CSV
    if not endpoints:
        print("No endpoints configured, creating sample data...")
        # Create sample data that matches your CSV structure
        sample_data = {
            'datetime': datetime.now().isoformat(),
            'CPU': 45.5,
            'time_taken': 1200,
            'RAM': 67.8,
            'sc_status': 200,
            'is_error': 0
        }
        all_data.append(sample_data)
        return all_data

    # Process actual endpoints
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            data = response.json()  # Fixed: was missing ()
            
            records = {
                'datetime': datetime.now().isoformat(),
                'CPU': data.get('cpu_usage', 0),
                'time_taken': data.get('response_time', 0),
                'RAM': data.get('memory_usage', 0),
                'sc_status': data.get('status_code', 200),
                'is_error': 1 if data.get('status_code', 200) >= 400 else 0
            }
            all_data.append(records)
        except Exception as e:
            print(f"Failed to extract from {endpoint}: {e}")
            continue

    return all_data

def transform_data(**context):
    """Transform extracted data"""
    raw_data = context['task_instance'].xcom_pull(task_ids='extract_data')
    
    if not raw_data:
        raise ValueError("No data received from extraction")
    
    df = pd.DataFrame(raw_data)
    
    # Data validation
    assert not df.empty, "Transformed data is empty"
    assert 'datetime' in df.columns, "Missing datetime column"
    assert 'time_taken' in df.columns, "Missing time_taken column"
    
    print(f"Transformed {len(df)} records")
    return df.to_json(orient='records')

def load_to_storage(**context):
    """Load data to CSV or SQL based on your config"""
    transformed_data = context['task_instance'].xcom_pull(task_ids='transform_data')
    df = pd.read_json(transformed_data, orient='records')
    
    # Use your existing DataManager
    data_manager = DataManager()
    
    # Check if using SQL or CSV based on your .env
    use_sql = os.getenv('USE_SQL', 'false').lower() == 'true'
    
    if use_sql:
        # Use proper pandas to_sql method instead of deprecated execute
        df.to_sql('system_metrics', data_manager.db.engine, if_exists='append', index=False)
    else:
        # Append to existing CSV file
        csv_path = './data/raw_data.csv'
        if os.path.exists(csv_path):
            existing_data = pd.read_csv(csv_path)
            combined_data = pd.concat([existing_data, df], ignore_index=True)
        else:
            combined_data = df
        combined_data.to_csv(csv_path, index=False)
    
    print(f"Loaded {len(df)} records")

# Create DAG
dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='ETL pipeline for system metrics',
    schedule_interval='*/10 * * * *',  # Every 10 minutes
    catchup=False,  # Change to False for testing
    max_active_runs=1,
    tags=['etl', 'data-pipeline']
)

# Define tasks with correct parameter names
extract_task = PythonOperator(
    task_id='extract_data',  # Fixed: was task_1
    python_callable=extract_from_endpoints,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',  # Fixed: was task_1
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(  # Fixed: was missing
    task_id='load_data',
    python_callable=load_to_storage,
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> load_task