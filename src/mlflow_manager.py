# src/mlflow_manager.py
import mlflow
import mlflow.keras
import pandas as pd
from datetime import datetime
import joblib
import json

class MLflowManager:
    def __init__(self, experiment_name="delay_prediction"):
        mlflow.set_experiment(experiment_name)
        
    def log_training_run(self, model, scaler, history, config, metrics):
        """Log complete training run"""
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("window_size", config['window_size'])
            mlflow.log_param("epochs", config['epochs'])
            mlflow.log_param("batch_size", config['batch_size'])
            mlflow.log_param("features", ",".join(config['features']))
            mlflow.log_param("model_type", "LSTM")
            
            # Log metrics
            mlflow.log_metric("final_accuracy", metrics['accuracy'])
            mlflow.log_metric("final_precision", metrics['precision'])
            mlflow.log_metric("final_recall", metrics['recall'])
            mlflow.log_metric("final_f1", metrics['f1_score'])
            
            # Log training history
            for epoch, (loss, val_loss, acc, val_acc) in enumerate(zip(
                history.history['loss'],
                history.history['val_loss'],
                history.history['accuracy'],
                history.history['val_accuracy']
            )):
                mlflow.log_metric("train_loss", loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("train_accuracy", acc, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            # Log model artifacts
            mlflow.keras.log_model(model, "lstm_model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Log feature names
            with open("feature_names.json", "w") as f:
                json.dump(config['features'], f)
            mlflow.log_artifact("feature_names.json")
            
            return mlflow.active_run().info.run_id
    
    def get_best_model(self, metric="final_f1"):
        """Get best performing model"""
        experiment = mlflow.get_experiment_by_name("delay_prediction")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if len(runs) > 0:
            best_run_id = runs.iloc[0].run_id
            return mlflow.keras.load_model(f"runs:/{best_run_id}/lstm_model")
        return None