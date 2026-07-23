# Optuna script to optimize Transformer models using 1 fold and evaluating classifier quality on real test data.

import os
import sys
# Import torch first to prevent CUDA library collision and Floating Point Exceptions with TensorFlow
try:
    import torch
except ImportError:
    pass
import csv
import json
import tempfile
import subprocess
from datetime import datetime
import numpy as np
import optuna
import tensorflow as tf

# Setup paths and environment
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import config
from folding import get_subject_folds, load_split_data
from classifier import (
    train_eegnet, evaluate_eegnet,
    train_pytorch_classifier, evaluate_pytorch_classifier,
    standardize_features
)
from augmentations import fit_and_refine_transformer
from models import build_standard_transformer, build_slowdown_transformer

# Setup GPU growth for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python scratch/transformer_optimizer.py <Standard_Transformer|Slowdown_Transformer> [n_trials]")
    sys.exit(1)

transformer_type = sys.argv[1]
if transformer_type not in ["Standard_Transformer", "Slowdown_Transformer"]:
    print(f"Error: Invalid transformer type '{transformer_type}'. Choose 'Standard_Transformer' or 'Slowdown_Transformer'.")
    sys.exit(1)

n_trials = 20
if len(sys.argv) > 2:
    try:
        n_trials = int(sys.argv[2])
    except ValueError:
        pass

RESULTS_CSV = os.path.join(os.path.dirname(__file__), f"results_transformer_optuna_{transformer_type}.csv")

def initialize_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Trial',
                'Transformer_Type',
                'D_Model',
                'Num_Heads',
                'FF_Dim',
                'Learning_Rate',
                'Epochs',
                'Batch_size',
                'Mask_ratio',
                'Slowdown_noise',
                'Slowdown_dropout',
                'Classifier_Type',
                'Accuracy',
                'Precision',
                'Recall',
                'F1',
                'Status'
            ])

def log_to_csv(trial_num, params, metrics, status="Success"):
    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            trial_num,
            transformer_type,
            params.get("d_model"),
            params.get("num_heads"),
            params.get("ff_dim"),
            params.get("transformer_lr"),
            params.get("transformer_epochs"),
            params.get("transformer_batch"),
            params.get("transformer_mask_ratio"),
            params.get("slowdown_noise", 0.0),
            params.get("slowdown_dropout", 0.0),
            config.CLASSIFIER_TYPE,
            round(metrics.get('accuracy', 0.0), 4),
            round(metrics.get('precision', 0.0), 4),
            round(metrics.get('recall', 0.0), 4),
            round(metrics.get('f1_score', 0.0), 4),
            status
        ])

def objective(trial):
    # 1. Suggest parameters for the transformer
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
    transformer_lr = trial.suggest_categorical("transformer_lr", [1e-4, 5e-4, 1e-3, 5e-3])
    transformer_epochs = trial.suggest_categorical("transformer_epochs", [10])
    transformer_batch = trial.suggest_categorical("transformer_batch", [16, 32, 64])
    transformer_mask_ratio = trial.suggest_categorical("transformer_mask_ratio", [0.10, 0.15, 0.20, 0.30])
    
    slowdown_noise = 0.0
    slowdown_dropout = 0.0
    
    if transformer_type == "Slowdown_Transformer":
        slowdown_noise = trial.suggest_categorical("slowdown_noise", [0.0, 0.01, 0.05, 0.10])
        slowdown_dropout = trial.suggest_categorical("slowdown_dropout", [0.0, 0.1, 0.2, 0.3])

    params = {
        "d_model": d_model,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "transformer_lr": transformer_lr,
        "transformer_epochs": transformer_epochs,
        "transformer_batch": transformer_batch,
        "transformer_mask_ratio": transformer_mask_ratio,
        "slowdown_noise": slowdown_noise,
        "slowdown_dropout": slowdown_dropout
    }

    print(f"\n--- Starting Trial {trial.number} | Transformer: {transformer_type} ---")
    print(f"Params: {params}")

    try:
        # Load splits and use only fold 1 (index 0)
        splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
        if os.path.exists(splits_path):
            with open(splits_path, 'r') as f:
                splits = json.load(f)
        else:
            splits = get_subject_folds(config.PROCESSED_DIR)
            
        split = splits[0]
        
        # Load and standardize fold data
        X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
        X_train_std, X_test_std = standardize_features(X_train, X_test)
        
        # Train transformer and generate refined features
        build_fn = build_slowdown_transformer if transformer_type == "Slowdown_Transformer" else build_standard_transformer
        
        X_tr_ref, X_te_ref = fit_and_refine_transformer(
            X_train_std, y_train, X_test_std, build_fn, transformer_type,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            transformer_epochs=transformer_epochs,
            transformer_batch=transformer_batch,
            transformer_lr=transformer_lr,
            transformer_mask_ratio=transformer_mask_ratio,
            slowdown_noise=slowdown_noise,
            slowdown_dropout=slowdown_dropout
        )
        
        # Clear keras session to reclaim GPU resources
        tf.keras.backend.clear_session()
        
        # Train and evaluate selected classifier on the generated training features
        classifier_name = getattr(config, "CLASSIFIER_TYPE", "ASDClassifier")
        
        if classifier_name == "EEGNet":
            print(f"Training EEGNet classifier on generated data...")
            tf.keras.backend.clear_session()
            model = train_eegnet(
                X_tr_ref, y_train,
                chans=config.CHANNELS,
                filters_conv1=getattr(config, "FILTERS_CONV1", 32),
                filters_conv2=getattr(config, "FILTERS_CONV2", 64),
                dropout=getattr(config, "DROPOUT", 0.2),
                learning_rate=getattr(config, "LEARNING_RATE", 0.001),
                batch_size=getattr(config, "CNN_BATCH", 32),
                epochs=getattr(config, "CNN_EPOCHS", 30)
            )
            metrics, _ = evaluate_eegnet(model, X_test_std, y_test)
        elif classifier_name == "ASDClassifier":
            print(f"Training ASDClassifier classifier on generated data (Subprocess)...")
            with tempfile.TemporaryDirectory() as tmpdir:
                X_train_path = os.path.join(tmpdir, "X_train.npy")
                y_train_path = os.path.join(tmpdir, "y_train.npy")
                X_test_path = os.path.join(tmpdir, "X_test.npy")
                y_test_path = os.path.join(tmpdir, "y_test.npy")
                out_results_json_path = os.path.join(tmpdir, "results.json")
                
                np.save(X_train_path, X_tr_ref)
                np.save(y_train_path, y_train)
                np.save(X_test_path, X_test_std)
                np.save(y_test_path, y_test)
                
                cmd = [
                    sys.executable,
                    os.path.join(config.BASE_DIR, "train_classifier.py"),
                    X_train_path,
                    y_train_path,
                    X_test_path,
                    y_test_path,
                    str(config.CHANNELS),
                    out_results_json_path
                ]
                
                subprocess.run(cmd, check=True)
                
                with open(out_results_json_path, 'r') as f:
                    metrics = json.load(f)
        else:
            raise ValueError(f"Unknown classifier name: {classifier_name}")
            
        print(f"Trial {trial.number} Finished | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
        log_to_csv(trial.number, params, metrics, "Success")
        return metrics["accuracy"]

    except Exception as e:
        print(f"Error in Trial {trial.number}: {e}")
        # Log failure in CSV and return 0.0 accuracy to avoid breaking study
        dummy_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        log_to_csv(trial.number, params, dummy_metrics, f"Failed: {str(e)}")
        return 0.0

def main():
    initialize_csv()
    print(f"Starting Optuna study for {transformer_type}...")
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
    main()
