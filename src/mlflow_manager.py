# src/mlflow_manager.py - Enhanced Version with Visualizations
import mlflow
import mlflow.keras
import pandas as pd
from datetime import datetime
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import io
import base64

class MLflowManager:
    def __init__(self, experiment_name="delay_prediction"):
        mlflow.set_experiment(experiment_name)
        
    def log_training_run(self, model, scaler, history, config, metrics, y_val, y_pred, y_pred_prob):
        """Enhanced log training run with visualizations"""
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param("window_size", config['window_size'])
            mlflow.log_param("epochs", config['epochs'])
            mlflow.log_param("batch_size", config['batch_size'])
            mlflow.log_param("features", ",".join(config['features']))
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("dropout_rate", config['dropout_rate'])
            mlflow.log_param("lstm_units", config['lstm_units'])
            
            # Log metrics
            mlflow.log_metric("final_accuracy", metrics['accuracy'])
            mlflow.log_metric("final_precision", metrics['precision'])
            mlflow.log_metric("final_recall", metrics['recall'])
            mlflow.log_metric("final_f1", metrics['f1_score'])
            
            # Log additional metrics
            mlflow.log_metric("val_samples", len(y_val))
            mlflow.log_metric("positive_class_ratio", np.mean(y_val))
            mlflow.log_metric("prediction_positive_ratio", np.mean(y_pred))
            mlflow.log_metric("mean_prediction_probability", np.mean(y_pred_prob))
            mlflow.log_metric("std_prediction_probability", np.std(y_pred_prob))
            
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
            
            # Create and log visualizations
            self._log_confusion_matrix(y_val, y_pred)
            self._log_prediction_distribution(y_pred_prob, y_val)
            self._log_precision_recall_curve(y_val, y_pred_prob)
            self._log_training_curves(history)
            self._log_probability_analysis(y_pred_prob, y_val, y_pred)
            
            # Log model artifacts
            mlflow.keras.log_model(model, "lstm_model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Log feature names
            with open("feature_names.json", "w") as f:
                json.dump(config['features'], f)
            mlflow.log_artifact("feature_names.json")
            
            # Log detailed predictions for analysis
            self._log_prediction_details(y_val, y_pred, y_pred_prob)
            
            return mlflow.active_run().info.run_id
    
    def _log_confusion_matrix(self, y_true, y_pred):
        """Create and log confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Delay', 'Delay'],
                   yticklabels=['No Delay', 'Delay'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add percentage annotations
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total:.1%})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()
        
        # Log confusion matrix values as metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
        mlflow.log_metric("true_negatives", tn)
        mlflow.log_metric("false_positives", fp)
        mlflow.log_metric("false_negatives", fn) 
        mlflow.log_metric("true_positives", tp)
    
    def _log_prediction_distribution(self, y_pred_prob, y_true):
        """Create and log prediction probability distribution"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall probability distribution
        axes[0,0].hist(y_pred_prob.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[0,0].set_title('Prediction Probability Distribution')
        axes[0,0].set_xlabel('Predicted Probability')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # 2. Distribution by true class
        prob_no_delay = y_pred_prob[y_true == 0].flatten()
        prob_delay = y_pred_prob[y_true == 1].flatten()
        
        axes[0,1].hist(prob_no_delay, bins=30, alpha=0.7, label='True: No Delay', color='green')
        axes[0,1].hist(prob_delay, bins=30, alpha=0.7, label='True: Delay', color='red')
        axes[0,1].axvline(0.5, color='black', linestyle='--', label='Threshold')
        axes[0,1].set_title('Probability Distribution by True Class')
        axes[0,1].set_xlabel('Predicted Probability')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # 3. Probability vs Time (sample)
        sample_size = min(500, len(y_pred_prob))
        sample_indices = np.random.choice(len(y_pred_prob), sample_size, replace=False)
        axes[1,0].plot(y_pred_prob.flatten()[sample_indices], 'o', alpha=0.6, markersize=3)
        axes[1,0].axhline(0.5, color='red', linestyle='--', label='Threshold')
        axes[1,0].set_title(f'Prediction Probabilities (Sample of {sample_size})')
        axes[1,0].set_xlabel('Sample Index')
        axes[1,0].set_ylabel('Predicted Probability')
        axes[1,0].legend()
        
        # 4. Probability ranges
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        range_counts = [np.sum((y_pred_prob >= r[0]) & (y_pred_prob < r[1])) for r in ranges]
        range_labels = [f'{r[0]:.1f}-{r[1]:.1f}' for r in ranges]
        
        axes[1,1].bar(range_labels, range_counts, color='lightcoral', alpha=0.7)
        axes[1,1].set_title('Predictions by Probability Range')
        axes[1,1].set_xlabel('Probability Range')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "prediction_distributions.png")
        plt.close()
    
    def _log_precision_recall_curve(self, y_true, y_pred_prob):
        """Create and log Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob.flatten())
        
        plt.figure(figsize=(10, 6))
        
        # Subplot 1: PR Curve
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.fill_between(recall, precision, alpha=0.2, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: F1 Score vs Threshold
        plt.subplot(1, 2, 2)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        plt.plot(thresholds, f1_scores, color='green', lw=2)
        plt.axvline(best_threshold, color='red', linestyle='--', 
                   label=f'Best F1 Threshold: {best_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "precision_recall_analysis.png")
        plt.close()
        
        # Log best threshold and corresponding metrics
        mlflow.log_metric("best_f1_threshold", best_threshold)
        mlflow.log_metric("best_f1_score", f1_scores[best_threshold_idx])
    
    def _log_training_curves(self, history):
        """Create and log training curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Training Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Additional metrics (if available)
        plt.subplot(1, 3, 3)
        if 'Recall' in history.history:
            plt.plot(history.history['Recall'], label='Training Recall', color='blue')
            plt.plot(history.history['val_Recall'], label='Validation Recall', color='red')
            plt.title('Model Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "training_curves.png")
        plt.close()
    
    def _log_probability_analysis(self, y_pred_prob, y_true, y_pred):
        """Detailed probability analysis"""
        prob_flat = y_pred_prob.flatten()
        
        # Calculate confidence metrics
        high_confidence_threshold = 0.8
        low_confidence_threshold = 0.2
        
        high_conf_correct = np.sum((prob_flat > high_confidence_threshold) & (y_pred == y_true))
        high_conf_total = np.sum(prob_flat > high_confidence_threshold)
        
        low_conf_correct = np.sum((prob_flat < low_confidence_threshold) & (y_pred == y_true))  
        low_conf_total = np.sum(prob_flat < low_confidence_threshold)
        
        # Log confidence metrics
        mlflow.log_metric("high_confidence_accuracy", high_conf_correct / (high_conf_total + 1e-12))
        mlflow.log_metric("low_confidence_accuracy", low_conf_correct / (low_conf_total + 1e-12))
        mlflow.log_metric("high_confidence_predictions", high_conf_total)
        mlflow.log_metric("low_confidence_predictions", low_conf_total)
        
        # Calibration analysis
        plt.figure(figsize=(10, 6))
        
        # Reliability diagram
        plt.subplot(1, 2, 1)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(bins) - 1):
            mask = (prob_flat >= bins[i]) & (prob_flat < bins[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_accuracies.append(bin_acc)
            else:
                bin_accuracies.append(0)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_centers, bin_accuracies, 'ro-', label='Model Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence histogram
        plt.subplot(1, 2, 2)
        confidence_scores = np.maximum(prob_flat, 1 - prob_flat)
        plt.hist(confidence_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Model Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "probability_analysis.png")
        plt.close()
    
    def _log_prediction_details(self, y_true, y_pred, y_pred_prob):
        """Log detailed prediction results"""
        # Create prediction summary DataFrame
        results_df = pd.DataFrame({
            'true_label': y_true.flatten(),
            'predicted_label': y_pred.flatten(),
            'predicted_probability': y_pred_prob.flatten(),
            'correct': (y_true.flatten() == y_pred.flatten()).astype(int),
            'confidence': np.maximum(y_pred_prob.flatten(), 1 - y_pred_prob.flatten())
        })
        
        # Save detailed results
        results_df.to_csv("prediction_details.csv", index=False)
        mlflow.log_artifact("prediction_details.csv")
        
        # Log summary statistics
        mlflow.log_metric("mean_confidence", results_df['confidence'].mean())
        mlflow.log_metric("min_confidence", results_df['confidence'].min())
        mlflow.log_metric("max_confidence", results_df['confidence'].max())
        
        # Class-wise statistics
        for class_val in [0, 1]:
            class_data = results_df[results_df['true_label'] == class_val]
            mlflow.log_metric(f"class_{class_val}_mean_prob", class_data['predicted_probability'].mean())
            mlflow.log_metric(f"class_{class_val}_std_prob", class_data['predicted_probability'].std())
    
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