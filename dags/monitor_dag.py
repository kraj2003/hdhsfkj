from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
import json

# Add your project root to Python path
sys.path.append('/opt/airflow')

# Simple DAG configuration
default_args = {
    'owner': 'monitoring-team',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'email_on_failure': False,
    'email_on_retry': False,
}

def monitor_data_health():
    """Monitor your data file health using your DataManager"""
    print("📊 Monitoring data health...")
    
    try:
        # Use your exact data loading logic
        from src.data_manager import DataManager
        
        # Load data like you do in main.py
        data_manager = DataManager()
        df = data_manager.load_data()
        
        if df.empty:
            print("❌ No data found")
            return {"status": "no_data", "records": 0}
        
        # Convert timestamp like in your preprocessing
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Check data freshness
        latest_time = df['timestamp'].max()
        oldest_time = df['timestamp'].min()
        hours_old = (datetime.now() - latest_time).total_seconds() / 3600
        data_span_days = (latest_time - oldest_time).days
        
        print(f"📈 Data Health Check:")
        print(f"   - Total records: {len(df)}")
        print(f"   - Data span: {data_span_days} days")
        print(f"   - Latest data: {hours_old:.1f} hours old")
        print(f"   - Date range: {oldest_time.strftime('%Y-%m-%d')} to {latest_time.strftime('%Y-%m-%d')}")
        
        # Check your required columns (from your preprocessing)
        required_cols = ['timestamp', 'response_time', 'CPU', 'RAM', 'is_error']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        # Data quality checks
        quality_issues = []
        
        if missing_cols:
            quality_issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values in critical columns
        for col in ['CPU', 'RAM', 'response_time']:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > len(df) * 0.1:  # More than 10% missing
                    quality_issues.append(f"Too many missing values in {col}: {missing_count}")
        
        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Duplicate timestamps: {duplicates}")
        
        # Check data freshness for predictions
        if hours_old > 6:
            quality_issues.append(f"Data is stale: {hours_old:.1f} hours old")
        
        # Determine status
        if hours_old > 24:
            status = "critical_stale_data"
        elif quality_issues:
            status = "data_quality_issues"
        elif hours_old > 2:
            status = "data_warning"
        else:
            status = "data_healthy"
        
        result = {
            "status": status,
            "records": len(df),
            "hours_old": hours_old,
            "data_span_days": data_span_days,
            "quality_issues": quality_issues
        }
        
        if quality_issues:
            print(f"⚠️ Quality issues: {quality_issues}")
        else:
            print("✅ Data quality good")
        
        return result
        
    except Exception as e:
        print(f"❌ Data health monitoring failed: {e}")
        return {"status": "monitoring_failed", "error": str(e)}

def monitor_model_health():
    """Monitor your model files and performance"""
    print("🤖 Monitoring model health...")
    
    try:
        # Check your model files (exactly your file paths)
        model_files = {
            'main_model': "./models/lstm_model_latest.h5",
            'scaler': "./models/scaler_latest.pkl",
            'features': "./models/feature_names.json"
        }
        
        file_status = {}
        missing_files = []
        model_found = False
        
        # Check each file
        for name, path in model_files.items():
            if os.path.exists(path):
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
                
                file_status[name] = {
                    "exists": True,
                    "size_mb": round(file_size_mb, 2),
                    "age_days": file_age_days
                }
                
                if 'model' in name:
                    model_found = True
                    
                print(f"✅ {name}: {file_size_mb:.1f}MB, {file_age_days} days old")
            else:
                missing_files.append(name)
                file_status[name] = {"exists": False}
                print(f"❌ Missing: {name} ({path})")
        
        # Test model loading (basic test)
        model_loadable = False
        if model_found and not missing_files:
            try:
                from tensorflow.keras.models import load_model
                import joblib
                
                # Try loading your model
                if os.path.exists("./models/lstm_model_latest.h5"):
                    model = load_model("./models/lstm_model_latest.h5")
                else:
                    model = load_model("./models/my_lstm_model.h5")
                
                scaler = joblib.load("./models/scaler_latest.pkl")
                
                with open("./models/feature_names.json", 'r') as f:
                    features = json.load(f)
                
                model_loadable = True
                print(f"✅ Model loads successfully ({len(features)} features)")
                
            except Exception as e:
                print(f"❌ Model loading test failed: {e}")
        
        # Determine model status
        if not model_found:
            model_status = "no_model"
        elif missing_files:
            model_status = "incomplete_model_files"
        elif not model_loadable:
            model_status = "model_corrupted"
        else:
            # Check model age
            model_age = min([f.get('age_days', 999) for f in file_status.values() if f.get('exists')])
            if model_age > 45:
                model_status = "model_very_old"
            elif model_age > 30:
                model_status = "model_old"
            else:
                model_status = "model_healthy"
        
        print(f"🤖 Model Status: {model_status}")
        
        return {
            "status": model_status,
            "files": file_status,
            "missing_files": missing_files,
            "model_loadable": model_loadable,
            "model_found": model_found
        }
        
    except Exception as e:
        print(f"❌ Model health monitoring failed: {e}")
        return {"status": "monitoring_failed", "error": str(e)}

def monitor_prediction_activity():
    """Monitor your prediction activity and results"""
    print("📈 Monitoring prediction activity...")
    
    try:
        # Check your predictions.csv file
        predictions_file = "./predictions.csv"
        
        if not os.path.exists(predictions_file):
            print("ℹ️ No predictions file found")
            return {"status": "no_predictions_file", "recent_count": 0}
        
        # Load predictions
        df = pd.read_csv(predictions_file)
        
        if df.empty:
            print("ℹ️ Predictions file is empty")
            return {"status": "no_predictions", "recent_count": 0}
        
        print(f"📊 Total predictions recorded: {len(df)}")
        
        # Analyze recent predictions (last 24 hours)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_predictions = df[df['timestamp'] >= recent_cutoff]
            
            # Analyze last hour for 5-minute predictions
            last_hour_cutoff = datetime.now() - timedelta(hours=1)
            last_hour_predictions = df[df['timestamp'] >= last_hour_cutoff]
            
            print(f"📊 Predictions in last 24h: {len(recent_predictions)}")
            print(f"📊 Predictions in last hour: {len(last_hour_predictions)}")
            
            # Expected predictions for 5-minute schedule
            expected_hourly = 12  # 60 minutes / 5 minutes
            expected_daily = 24 * expected_hourly  # 288 predictions per day
            
            if len(recent_predictions) > 0:
                # Analysis
                delay_rate = recent_predictions['prediction'].mean() * 100
                
                prediction_stats = {
                    "recent_count": len(recent_predictions),
                    "hourly_count": len(last_hour_predictions),
                    "delay_prediction_rate": delay_rate,
                    "expected_daily": expected_daily,
                    "actual_daily": len(recent_predictions)
                }
                
                if 'probability' in recent_predictions.columns:
                    avg_prob = recent_predictions['probability'].mean()
                    high_conf_count = (recent_predictions['probability'] > 0.7).sum()
                    low_conf_count = (recent_predictions['probability'] < 0.3).sum()
                    
                    prediction_stats.update({
                        "avg_probability": avg_prob,
                        "high_confidence_predictions": high_conf_count,
                        "low_confidence_predictions": low_conf_count
                    })
                    
                    print(f"📊 Average prediction probability: {avg_prob:.3f}")
                    print(f"📊 High confidence predictions (>0.7): {high_conf_count}")
                
                print(f"📊 Delay prediction rate: {delay_rate:.1f}%")
                
                # Determine activity status
                if len(last_hour_predictions) < expected_hourly * 0.5:  # Less than 50% expected
                    activity_status = "low_activity"
                    print("⚠️ Low prediction activity in last hour")
                elif len(recent_predictions) < expected_daily * 0.7:  # Less than 70% expected daily
                    activity_status = "reduced_activity"
                    print("⚠️ Reduced prediction activity")
                else:
                    activity_status = "normal_activity"
                    print("✅ Normal prediction activity")
                
                return {
                    "status": activity_status,
                    **prediction_stats
                }
                
            else:
                print("⚠️ No recent predictions found")
                return {"status": "no_recent_predictions", "recent_count": 0}
        
        else:
            print("❌ Invalid predictions file format (no timestamp)")
            return {"status": "invalid_predictions_format", "total_count": len(df)}
        
    except Exception as e:
        print(f"❌ Prediction activity monitoring failed: {e}")
        return {"status": "monitoring_failed", "error": str(e)}

def monitor_system_resources():
    """Monitor system resources (disk space, memory, etc.)"""
    print("💻 Monitoring system resources...")
    
    try:
        import shutil
        
        # Check disk space in project directory
        disk_usage = shutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        total_gb = disk_usage.total / (1024**3)
        used_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
        
        print(f"💽 Disk Usage:")
        print(f"   - Used: {used_percent:.1f}% ({total_gb - free_gb:.1f}GB / {total_gb:.1f}GB)")
        print(f"   - Free: {free_gb:.1f}GB")
        
        # Check important directories
        important_dirs = {
            'data': './data',
            'models': './models',
            'logs': './logs',
            'reports': './reports'
        }
        
        dir_status = {}
        
        for name, path in important_dirs.items():
            if os.path.exists(path):
                try:
                    files_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                    dir_size_mb = sum(os.path.getsize(os.path.join(path, f)) 
                                    for f in os.listdir(path) 
                                    if os.path.isfile(os.path.join(path, f))) / (1024**2)
                    
                    dir_status[name] = {
                        "exists": True,
                        "files": files_count,
                        "size_mb": round(dir_size_mb, 2)
                    }
                    print(f"📁 {name}: {files_count} files, {dir_size_mb:.1f}MB")
                except:
                    dir_status[name] = {"exists": True, "error": "access_denied"}
            else:
                dir_status[name] = {"exists": False}
                print(f"❌ Missing directory: {path}")
        
        # Determine system status
        if free_gb < 0.5:
            system_status = "critical_low_disk"
        elif free_gb < 2:
            system_status = "low_disk_space"
        elif used_percent > 90:
            system_status = "disk_nearly_full"
        else:
            system_status = "system_healthy"
        
        return {
            "status": system_status,
            "free_space_gb": free_gb,
            "used_percent": used_percent,
            "directories": dir_status
        }
        
    except Exception as e:
        print(f"❌ System resource monitoring failed: {e}")
        return {"status": "monitoring_failed", "error": str(e)}

def generate_monitoring_report(**context):
    """Generate comprehensive monitoring report"""
    print("📋 Generating monitoring report...")
    
    try:
        # Get results from previous tasks using XCom
        data_health = context['task_instance'].xcom_pull(task_ids='monitor_data_health')
        model_health = context['task_instance'].xcom_pull(task_ids='monitor_model_health')
        prediction_activity = context['task_instance'].xcom_pull(task_ids='monitor_prediction_activity')
        system_resources = context['task_instance'].xcom_pull(task_ids='monitor_system_resources')
        
        # Create comprehensive report
        report = {
            "monitoring_date": datetime.now().strftime('%Y-%m-%d'),
            "monitoring_timestamp": datetime.now().isoformat(),
            "data_health": data_health or {"status": "not_monitored"},
            "model_health": model_health or {"status": "not_monitored"},
            "prediction_activity": prediction_activity or {"status": "not_monitored"},
            "system_resources": system_resources or {"status": "not_monitored"}
        }
        
        # Determine overall system health
        all_statuses = []
        
        for component, result in [
            ("Data", data_health),
            ("Model", model_health), 
            ("Predictions", prediction_activity),
            ("System", system_resources)
        ]:
            if result:
                status = result.get('status', 'unknown')
                all_statuses.append(status)
                print(f"📊 {component} Status: {status}")
        
        # Overall health assessment
        critical_statuses = ['critical_stale_data', 'no_model', 'critical_low_disk', 'monitoring_failed']
        warning_statuses = ['data_quality_issues', 'model_old', 'low_activity', 'low_disk_space']
        
        if any(status in critical_statuses for status in all_statuses):
            overall_status = "CRITICAL"
            alert_level = "🚨"
        elif any(status in warning_statuses for status in all_statuses):
            overall_status = "WARNING"
            alert_level = "⚠️"
        else:
            overall_status = "HEALTHY"
            alert_level = "✅"
        
        report["overall_status"] = overall_status
        
        print(f"\n{alert_level} SYSTEM HEALTH REPORT {alert_level}")
        print(f"Overall Status: {overall_status}")
        print(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Summary by component
        if data_health:
            print(f"📊 Data: {data_health.get('records', 'N/A')} records, {data_health.get('hours_old', 'N/A'):.1f}h old")
        
        if model_health:
            print(f"🤖 Model: {'Available' if model_health.get('model_found') else 'Missing'}")
        
        if prediction_activity:
            recent = prediction_activity.get('recent_count', 0)
            print(f"📈 Predictions: {recent} in last 24h")
        
        if system_resources:
            free_space = system_resources.get('free_space_gb', 0)
            print(f"💽 Disk: {free_space:.1f}GB free")
        
        # Save report
        reports_dir = "./reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        report_file = f"{reports_dir}/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Monitoring report saved: {report_file}")
        
        # Alert messages
        if overall_status == "CRITICAL":
            print("🚨 CRITICAL ISSUES DETECTED - IMMEDIATE ACTION REQUIRED!")
        elif overall_status == "WARNING":
            print("⚠️ WARNING - SYSTEM ISSUES DETECTED")
        else:
            print("✅ All systems operating normally")
        
        return report
        
    except Exception as e:
        print(f"❌ Monitoring report generation failed: {e}")
        return {"status": "report_failed", "error": str(e)}

# Create the DAG
dag = DAG(
    'system_monitoring_hourly',
    default_args=default_args,
    description='Comprehensive system monitoring every hour',
    schedule_interval='0 * * * *',  # Every hour at minute 0
    catchup=False,
    tags=['monitoring', 'hourly', 'health'],
    max_active_runs=1,
)

# Define tasks
data_health_task = PythonOperator(
    task_id='monitor_data_health',
    python_callable=monitor_data_health,
    dag=dag,
)

model_health_task = PythonOperator(
    task_id='monitor_model_health',
    python_callable=monitor_model_health,
    dag=dag,
)

prediction_activity_task = PythonOperator(
    task_id='monitor_prediction_activity',
    python_callable=monitor_prediction_activity,
    dag=dag,
)

system_resources_task = PythonOperator(
    task_id='monitor_system_resources',
    python_callable=monitor_system_resources,
    dag=dag,
)

monitoring_report_task = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report,
    provide_context=True,
    dag=dag,
)

# Set task dependencies - run all monitoring in parallel, then generate report
[data_health_task, model_health_task, prediction_activity_task, system_resources_task] >> monitoring_report_task