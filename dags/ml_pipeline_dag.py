from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
import sys
sys.path.append('/app')  # Add your project root to path

from src.data_manager import DataManager
from src.preprocessing import preprocessing_pipeline
from tensorflow.kers.model import load_model
import joblib
from src.train import train_lstm
from src.evaluate import predict_future
from src.mlflow_manager import MLflowManager

default_args = {
    'owner': 'ml-team',
    'depends_on_past': True, # a task will only run if the previous run of that same task succeeded.
    'start_date': datetime(2024, 1, 1), # The first date Airflow will schedule the DAG from.
    'email_on_failure': True, # Airflow will send an email when a task fails (requires email config in Airflow).
    'retries': 1, #  If a task fails, Airflow will retry it 1 extra time before marking it as failed. 
    'retry_delay': timedelta(minutes=10),# How long Airflow waits before retrying.
}

def check_data_availability(**context):
    """Check if we have sufficient data for training"""
    data_manager = DataManager()
    df = data_manager.load_data()
    
    min_records = 500  # Adjust based on your needs
    if len(df) < min_records:
        raise ValueError(f"Not enough data for training: {len(df)} records")
    
    print(f"✅ Data check passed: {len(df)} records available")
    return len(df)

def run_preprocessing(**context):
    """Run your preprocessing pipeline"""
    data_manager = DataManager()
    df = data_manager.load_data()
    
    # Use your existing preprocessing function
    processed_df = preprocessing_pipeline(df)
    
    print(f"✅ Preprocessing complete: {len(processed_df)} processed records")
    return processed_df.to_json(orient='records')

def train_model(**context):
    """Train LSTM model using your existing training function"""
    # Get preprocessed data
    processed_data = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    processed_df = pd.read_json(processed_data, orient='records', convert_dates=['datetime'] if 'datetime' in processed_data else [])
    
    # Train using your existing function
    model, scaler, metrics = train_lstm(processed_df)
    
    # Validate model performance
    min_accuracy = 0.60
    if metrics['accuracy'] < min_accuracy:
        print(f"⚠️ Model accuracy {metrics['accuracy']:.3f} is below threshold {min_accuracy}")
        # You can decide to fail the task or just warn
    
    print(f"✅ Model trained successfully - Accuracy: {metrics['accuracy']:.3f}")
    return {
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'model_saved': True
    }

def generate_prediction(**context):
    data_manager=DataManager()

    df=data_manager.load_data()

    preprocessed_df=preprocessing_pipeline(df)

    try:
        model=load_model("models/lstm_latest.h5")
        scaler=joblib.load("models/scaler_latest.pkl")
        features=['Delay_Detected','CPU','RAM','time_taken']

        prediction_class,prediction_prob=predict_future(model,scaler,preprocessed_df,features, window_size=24)

        # Save predictions using existing data manager
        import pandas as pd
        predictions_df = pd.DataFrame({
            'prediction': [prediction_class],
            'probability': [prediction_prob],
            'timestamp': [datetime.now()]
        })
        
        data_manager.save_results(predictions_df)
        
        print(f"✅ Prediction completed: {prediction_class} (prob: {prediction_prob:.3f})")
        return f"Prediction: {prediction_class}, Probability: {prediction_prob:.3f}"
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise 

dag2 = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML training and prediction pipeline',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    catchup=False,
    max_active_runs=1,
)

# Tasks
data_check_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data_availability,
    dag=dag2,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=run_preprocessing,
    dag=dag2,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag2,
)

predict_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_prediction,
    dag=dag2,
)

# set the ML pipeline dependencies
data_check_task<<preprocess_task<<train_task<<predict_task