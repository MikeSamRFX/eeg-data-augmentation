# This file trains the classifier only on the base data, testing hyperparameters and returning the best hyperparameter combinations.
# The objective function tests EEGNet and ASDClassifier using Optuna.

import os
import sys
# Import torch first to prevent CUDA library collision and Floating Point Exceptions with TensorFlow
try:
    import torch
except ImportError:
    pass
import csv
import json
from datetime import datetime
import numpy as np
import optuna
import tensorflow as tf

# Set up environment and paths
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import config
from folding import get_subject_folds, load_split_data
from classifier import (
    train_eegnet, evaluate_eegnet,
    train_pytorch_classifier, evaluate_pytorch_classifier,
    standardize_features
)

# Setup GPU growth for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Ensure results CSV path
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results_optuna.csv")

def initialize_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Fold',
                'Model',
                'Batch_size',
                'Learning_rate',
                'Filters_conv1',
                'Filters_conv2',
                'Filters_conv3',
                'Dropout',
                'Kernel_size',
                'Accuracy',
                'Precision',
                'Recall',
                'F1'
            ])

def log_to_csv(fold, model_name, params, metrics):
    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fold,
            model_name,
            params.get("batch_size"),
            params.get("learning_rate"),
            params.get("filters_conv1"),
            params.get("filters_conv2"),
            params.get("filters_conv3", "None"),
            params.get("dropout"),
            str(params.get("kernel_size", "None")),
            round(metrics['accuracy'], 4),
            round(metrics['precision'], 4),
            round(metrics['recall'], 4),
            round(metrics['f1_score'], 4)
        ])

def objective(trial):
    # Sample common parameters
    classifier_name = trial.suggest_categorical("classifier_name", ["EEGNet", "ASDClassifier"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.5])
    
    # Sample model-specific parameters
    if classifier_name == "EEGNet":
        filters_conv1 = trial.suggest_categorical("filters_conv1", [8, 16, 32])
        filters_conv2 = trial.suggest_categorical("filters_conv2", [16, 32, 64])
        filters_conv3 = None
        kernel_size = None
        params = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "filters_conv1": filters_conv1,
            "filters_conv2": filters_conv2
        }
    else:
        filters_conv1 = trial.suggest_categorical("filters_conv1", [8, 16, 32])
        filters_conv2 = trial.suggest_categorical("filters_conv2", [16, 32, 64])
        filters_conv3 = trial.suggest_categorical("filters_conv3", [32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [(3, 3), (3, 5), (5, 5)])
        params = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "filters_conv1": filters_conv1,
            "filters_conv2": filters_conv2,
            "filters_conv3": filters_conv3,
            "kernel_size": kernel_size
        }

    # Load/prepare stratified splits
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    else:
        splits = get_subject_folds(config.PROCESSED_DIR)

    fold_accuracies = []
    
    print(f"\n--- Starting Trial {trial.number} | Model: {classifier_name} ---")
    print(f"Params: {params}")

    for split in splits:
        fold_idx = split["fold_idx"]
        print(f"  Fold {fold_idx}/{len(splits)}...")
        
        # Load fold data (raw base data only)
        X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
        
        # Standardize features channel-wise
        X_train_norm, X_test_norm = standardize_features(X_train, X_test)
        
        if classifier_name == "EEGNet":
            # Clear keras session to avoid memory growth/leakage
            tf.keras.backend.clear_session()
            
            # Train and evaluate EEGNet
            model = train_eegnet(
                X_train_norm, y_train,
                chans=config.CHANNELS,
                filters_conv1=filters_conv1,
                filters_conv2=filters_conv2,
                dropout=dropout,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=config.CNN_EPOCHS
            )
            metrics, _ = evaluate_eegnet(model, X_test_norm, y_test)
            
        else:
            # Train and evaluate ASDClassifier (PyTorch)
            model = train_pytorch_classifier(
                X_train_norm, y_train,
                chans=config.CHANNELS,
                filters_conv1=filters_conv1,
                filters_conv2=filters_conv2,
                filters_conv3=filters_conv3,
                dropout=dropout,
                kernel_size=kernel_size,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=config.CNN_EPOCHS
            )
            metrics, _ = evaluate_pytorch_classifier(model, X_test_norm, y_test)

        # Log results for this fold
        log_to_csv(fold_idx, classifier_name, params, metrics)
        fold_accuracies.append(metrics["accuracy"])
        print(f"    Fold {fold_idx} Accuracy: {metrics['accuracy']:.4f} | F1-score: {metrics['f1_score']:.4f}")

    avg_accuracy = np.mean(fold_accuracies)
    print(f"Trial {trial.number} Finished | Avg Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy

def main(n_trials=20):
    initialize_csv()
    
    print("Initializing Optuna study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n=== OPTIMIZATION COMPLETED ===")
    print("Best Trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    # Allow passing number of trials from command line argument
    trials = 100
    if len(sys.argv) > 1:
        try:
            trials = int(sys.argv[1])
        except ValueError:
            pass
    main(n_trials=trials)
