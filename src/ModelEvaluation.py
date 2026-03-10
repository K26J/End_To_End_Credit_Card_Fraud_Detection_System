import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import mlflow
import yaml
import json
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix

# 1. Load the active parameters
with open("params.yaml", "r") as f:
    all_params = yaml.safe_load(f)

EXPERIMENT_NAME = all_params["experiment_name"]

def evaluate_model():
    print(f"Starting Evaluation for: {EXPERIMENT_NAME}")
    
    # ---------------------------------------------------------
    # SAFETY NET: Create placeholders for all 6 models!
    # ---------------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    models_list = [
        "baseline_model", 
        "generalized_base_model", 
        "class_weights_model",
        "custom_class_weights_model",
        "controlled_class_weights_model",
        "final_model"
    ]
    
    for m in models_list:
        path = os.path.join("reports", f"{m}_metrics.json")
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({"status": "not evaluated yet"}, f) 
    # ---------------------------------------------------------

    # 2. Load the completely unseen Test Data
    test_data_path = os.path.join("data", "test_data_scaled.csv")
    df_test = pd.read_csv(test_data_path)
    
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]
    
    # 3. Load the specifically trained model
    model_path = os.path.join("models", f"{EXPERIMENT_NAME}.h5")
    model = load_model(model_path)
    
    # 4. Connect to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{EXPERIMENT_NAME}_test_evaluation"):
        
        print("Predicting on unseen test data...")
        y_pred_prob = model.predict(X_test)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)
        
        # METRIC CALCULATIONS
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall_vals, precision_vals)
        
        test_precision = precision_score(y_test, y_pred_binary)
        test_recall = recall_score(y_test, y_pred_binary)
        test_accuracy = accuracy_score(y_test, y_pred_binary)
        
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        
        # VISUAL ARTIFACTS
        cm = confusion_matrix(y_test, y_pred_binary)
        fig_cm = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {EXPERIMENT_NAME}')
        plt.ylabel('Actual Fraud Status')
        plt.xlabel('Predicted Fraud Status')
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)
        
        fig_pr = plt.figure(figsize=(8,6))
        plt.plot(recall_vals, precision_vals, color='darkorange', linewidth=2, label=f"Test PR-AUC= {pr_auc:.4f}")
        plt.title('Precision-Recall Curve on Unseen Test Data', fontsize=14, fontweight='bold')
        plt.xlabel('Recall (Percentage of Fraud Caught)')
        plt.ylabel('Precision (Accuracy of Alarms)')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.7)
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")
        plt.close(fig_pr)
        
        # DVC REPORTING
        report_path = os.path.join("reports", f"{EXPERIMENT_NAME}_metrics.json")        

        report_dict = {
            "test_pr_auc": float(pr_auc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall)
        }
        
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
            
        print(f"Evaluation complete. Report saved to {report_path} and uploaded to MLflow.")

if __name__ == "__main__":
    evaluate_model()