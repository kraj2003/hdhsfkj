from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
from src.data_manager import DataManager

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
}
def extract_from_endpoints(**context):
    endpoints=[
        # all endpoints comes here 
    ]

    all_data=[]

    for endpoint in endpoints:
        try:
            response=requests.get(endpoint,timeout=30)
            response.raise_for_status() # It will check the HTTP status code inside the response.
            data=response.json
            records={
                
                'datetime':datetime.now().isoformat(),
                'CPU':data.get('cpu_usage',0),
                'time_taken':data.get('response_time',0),
                'RAM':data.get('memory_usage',0),
                'sc_status':data.get('status_code',200),
                'is_error': 1 if data.get('status_code', 200) >=400 else 0
            }
            all_data.append(records)
        except Exception as e:
            print(f"Failed to extract from {endpoint}: {e}")

    return all_data

def transform_data(**context):
    """Transform extracted data"""
    raw_data = context['task_instance'].xcom_pull(task_ids='extract_data')
    
    if not raw_data:
        raise ValueError("No data received from extraction")
    
    # Create DataFrame matching your existing structure
    df = pd.DataFrame(raw_data)
    
    # Data validation
    assert not df.empty, "Transformed data is empty"
    assert 'datetime' in df.columns, "Missing datetime column"
    assert 'time_taken' in df.columns, "Missing time_taken column"
    
    return df.to_json(orient='records')


# loading the transformed data into sql 
def load_to_storage(**context):
    """Load data to CSV or SQL based on your config"""
    transformed_data = context['task_instance'].xcom_pull(task_ids='transform_data')
    df = pd.read_json(transformed_data, orient='records')
    
    # Use your existing DataManager
    data_manager = DataManager()
    
    # Check if using SQL or CSV based on your .env
    use_sql = os.getenv('USE_SQL', 'false').lower() == 'true'
    
    if use_sql:
        # Save to SQL using your existing database setup
        data_manager.db.engine.execute(
            f"INSERT INTO system_metrics (datetime, time_taken, CPU, RAM, sc_status, is_error) VALUES {','.join([str(tuple(row)) for row in df.values])}"
        )
    else:
        # Append to existing CSV file
        existing_data = pd.read_csv('./data/raw_data.csv') if os.path.exists('./data/raw_data.csv') else pd.DataFrame()
        combined_data = pd.concat([existing_data, df], ignore_index=True)
        combined_data.to_csv('./data/raw_data.csv', index=False)
    
    print(f"Loaded {len(df)} records")


dag1=DAG(
    'data_pipeline',
    default_args=default_args,
    description='ETL pipeline for system metrics',
    schedule_interval='*/10 * * * *',  # Every 10 minutes
    catchup=False,
    max_active_runs=1,
)

extract_task=PythonOperator(
    task_1='extract_data',
    python_callable=extract_from_endpoints,
    dag=dag1
)

transform_task=PythonOperator(
    task_1='transform_data',
    python_callable=transform_data,
    dag=dag1
)

# Dependencies
extract_task >> transform_task >> load_to_storage
