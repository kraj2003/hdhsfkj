from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# Add your project root to Python path
sys.path.append('/opt/airflow')

# Simple DAG configuration
default_args = {
    'owner': 'ml-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
    'email_on_failure': False,
    'email_on_retry': False,
}

def backup_current_model():
    """Backup current model before training new one"""
    print("💾 Backing up current model...")
    
    try:
        import shutil
        
        # Your model file paths
        current_models = [
            "./models/lstm_model_latest.h5",
            # "./models/my_lstm_model.h5",
            "./models/scaler_latest.pkl",
            "./models/feature_names.json"
        ]
        
        # Create backup directory
        backup_dir = "./models/backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_count = 0
        
        for model_file in current_models:
            if os.path.exists(model_file):
                filename = os.path.basename(model_file)
                name, ext = os.path.splitext(filename)
                backup_name = f"{name}_backup_{timestamp}{ext}"
                backup_path = os.path.join(backup_dir, backup_name)
                
                shutil.copy2(model_file, backup_path)
                backup_count += 1
                print(f"✅ Backed up: {filename} -> {backup_name}")
        
        if backup_count == 0:
            print("ℹ️ No existing models to backup")
        else:
            print(f"✅ Backup complete: {backup_count} files backed up")
        
        return f"backup_success_{backup_count}"
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return "backup_failed"

def train_new_model():
    """Train new model using your exact main.py logic"""
    print("🚀 Training new model...")
    
    try:
        # Import your modules (exactly like main.py)
        from src.data_manager import DataManager
        from src.preprocessing import preprocessing_pipeline
        from src.train import train_lstm
        
        print("✅ Successfully imported training modules")
        
        # Load data (exactly like main.py)
        print("📊 Loading training data...")
        data_manager = DataManager()
        df = data_manager.load_data()
        print(f"✅ Loaded {len(df)} records for training")
        
        # Check if we have enough data
        if len(df) < 1000:
            print(f"❌ Insufficient data for training: {len(df)} records")
            raise ValueError(f"Need at least 1000 records, got {len(df)}")
        
        # Preprocess data (exactly like main.py)
        print("🔄 Preprocessing training data...")
        processed_df = preprocessing_pipeline(df)
        print(f"✅ Processed to {len(processed_df)} training samples")
        
        # Check processed data
        if len(processed_df) < 500:
            print(f"❌ Insufficient processed data: {len(processed_df)} samples")
            raise ValueError(f"Need at least 500 processed samples, got {len(processed_df)}")
        
        # Train model (exactly like main.py)
        print("🤖 Training LSTM model...")
        model, scaler, metrics = train_lstm(processed_df)
        
        # Validate training results
        print(f"📊 Training Results:")
        print(f"   - Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"   - Precision: {metrics.get('precision', 0):.3f}")
        print(f"   - Recall: {metrics.get('recall', 0):.3f}")
        print(f"   - F1 Score: {metrics.get('f1_score', 0):.3f}")
        
        # Check if model performance is acceptable
        min_f1_score = 0.1
        if metrics.get('f1_score', 0) < min_f1_score:
            print(f"⚠️ Warning: Low model performance (F1: {metrics.get('f1_score', 0):.3f})")
        else:
            print(f"✅ Model performance acceptable")
        
        # Verify model files were created
        expected_files = [
            "./models/lstm_model_latest.h5",
            "./models/scaler_latest.pkl", 
            "./models/feature_names.json"
        ]
        
        created_files = []
        for file_path in expected_files:
            if os.path.exists(file_path):
                created_files.append(file_path)
                print(f"✅ Created: {file_path}")
            else:
                print(f"❌ Missing: {file_path}")
        
        if len(created_files) < 3:
            print("❌ Some model files were not created properly")
            return "training_incomplete"
        
        print("✅ Training completed successfully!")
        return f"training_success_f1_{metrics.get('f1_score', 0):.3f}"
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        # Print detailed error for debugging
        import traceback
        traceback.print_exc()
        raise

def validate_new_model():
    """Test the newly trained model"""
    print("🔍 Validating newly trained model...")
    
    try:
        # Import required modules
        from src.data_manager import DataManager
        from src.preprocessing import preprocessing_pipeline
        from src.evaluate import predict_future
        from src import config
        from tensorflow.keras.models import load_model
        import joblib
        import json
        
        # Try loading the new model
        print("🤖 Loading newly trained model...")
        model = load_model("./models/lstm_model_latest.h5")
        scaler = joblib.load("./models/scaler_latest.pkl")
        
        with open("./models/feature_names.json", 'r') as f:
            features = json.load(f)
        
        print(f"✅ New model loaded successfully ({len(features)} features)")
        
        # Test prediction with recent data
        print("🧪 Testing prediction capability...")
        data_manager = DataManager()
        df = data_manager.load_data()
        
        # Get small sample for testing
        processed_df = preprocessing_pipeline(df)
        test_df = processed_df.tail(100)  # Last 100 records for test
        
        # Make test prediction
        prediction_class, prediction_prob = predict_future(
            model, scaler, test_df, features, config.WINDOW_SIZE
        )
        
        print(f"📊 Test Prediction Results:")
        print(f"   - Prediction: {'Delay' if prediction_class else 'No Delay'}")
        print(f"   - Probability: {prediction_prob:.3f}")
        
        # Basic validation checks
        if prediction_prob < 0 or prediction_prob > 1:
            print(f"❌ Invalid probability range: {prediction_prob}")
            return "invalid_prediction_range"
        
        if prediction_class not in [0, 1]:
            print(f"❌ Invalid prediction class: {prediction_class}")
            return "invalid_prediction_class"
        
        print("✅ Model validation passed!")
        return f"validation_success_{prediction_prob:.3f}"
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return "validation_failed"

def check_training_prerequisites():
    """Check if we have everything needed for training"""
    print("📋 Checking training prerequisites...")
    
    try:
        # Check data file
        data_file = "./data/raw_data.csv"
        if not os.path.exists(data_file):
            print(f"❌ Training data missing: {data_file}")
            return "no_training_data"
        
        # Check data size
        df = pd.read_csv(data_file)
        print(f"📊 Training data: {len(df)} records")
        
        if len(df) < 1000:
            print(f"❌ Insufficient data for training: {len(df)} < 1000")
            return "insufficient_data"
        
        # Check required columns
        required_cols = ['timestamp', 'response_time', 'CPU', 'RAM', 'is_error']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return "missing_columns"
        
        # Check disk space (basic check)
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        print(f"💽 Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 1:
            print("❌ Insufficient disk space for training")
            return "insufficient_disk_space"
        
        print("✅ All prerequisites met for training")
        return "prerequisites_ok"
        
    except Exception as e:
        print(f"❌ Prerequisites check failed: {e}")
        return "prerequisites_check_failed"

# Create the DAG
dag = DAG(
    'model_training_monthly',
    default_args=default_args,
    description='Train new model every month',
    schedule_interval='0 2 1 * *',  # 2 AM on 1st day of every month
    catchup=False,
    tags=['training', 'monthly', 'model'],
    max_active_runs=1,
)

# Define tasks
prerequisites_task = PythonOperator(
    task_id='check_training_prerequisites',
    python_callable=check_training_prerequisites,
    dag=dag,
)

backup_task = PythonOperator(
    task_id='backup_current_model',
    python_callable=backup_current_model,
    dag=dag,
)

training_task = PythonOperator(
    task_id='train_new_model',
    python_callable=train_new_model,
    dag=dag,
)

validation_task = PythonOperator(
    task_id='validate_new_model',
    python_callable=validate_new_model,
    dag=dag,
)

# Set task dependencies
prerequisites_task >> backup_task >> training_task >> validation_task