import os
import sys
import json
import numpy as np
import tempfile
import subprocess
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
import config
from folding import load_split_data
from classifier import standardize_features
from augmentations import fit_and_refine_transformer
from models import build_standard_transformer

def run_4_folds_transformer():
    print(" Starting 4-Fold Cross Validation with 20 Epochs...")
    
    # 1. Load splits configuration
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    if not os.path.exists(splits_path):
        print("Generating folds splits...")
        from folding import get_subject_folds
        splits = get_subject_folds(config.PROCESSED_DIR, n_folds=4)
    else:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
    # Save original configurations to restore later
    old_tf_epochs = config.TRANSFORMER_EPOCHS
    old_cnn_epochs = config.CNN_EPOCHS
    
    # Set requested number of epochs (20 epochs for both Autoencoder and Classifier)
    config.TRANSFORMER_EPOCHS = 20
    config.CNN_EPOCHS = 20
    
    all_metrics = []
    
    try:
        for idx, split in enumerate(splits):
            fold_idx = split["fold_idx"]
            print(f"\n==========================================")
            print(f"       RUNNING FOLD {fold_idx}/4")
            print(f"==========================================")
            
            # Load and standardize training/test slices
            X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
            X_train_std, X_test_std = standardize_features(X_train, X_test)
            
            # Train Autoencoder and reconstruct signals
            print(f"\n--- Fold {fold_idx} Step 1: Training Autoencoder ---")
            X_tr_aug, X_te = fit_and_refine_transformer(
                X_train_std, y_train, X_test_std, build_standard_transformer, "Standard_Transformer"
            )
            
            # Determine channels dynamically
            chans = X_tr_aug.shape[1]
            
            # Train classifier on reconstructed signals
            print(f"\n--- Fold {fold_idx} Step 2: Training PyTorch Classifier ---")
            with tempfile.TemporaryDirectory() as tmpdir:
                X_train_path = os.path.join(tmpdir, "X_train.npy")
                y_train_path = os.path.join(tmpdir, "y_train.npy")
                X_test_path = os.path.join(tmpdir, "X_test.npy")
                y_test_path = os.path.join(tmpdir, "y_test.npy")
                out_results_json_path = os.path.join(tmpdir, "results.json")
                
                np.save(X_train_path, X_tr_aug)
                np.save(y_train_path, y_train)
                np.save(X_test_path, X_test_std) # Evaluate on real test signals
                np.save(y_test_path, y_test)
                
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "..", "code", "train_classifier.py"),
                    X_train_path,
                    y_train_path,
                    X_test_path,
                    y_test_path,
                    str(chans),
                    out_results_json_path
                ]
                
                subprocess.run(cmd, check=True)
                
                with open(out_results_json_path, 'r') as f:
                    metrics = json.load(f)
                    
            print(f"Fold {fold_idx} metrics: {metrics}")
            all_metrics.append(metrics)
            
        # 4-Fold Summary Report
        print("\n" + "="*50)
        print("         4-FOLD EVALUATION SUMMARY")
        print("="*50)
        print(f"{'Fold':<8} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-"*55)
        
        accs, precs, recs, f1s = [], [], [], []
        for i, m in enumerate(all_metrics):
            print(f"Fold {i+1:<4} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
            accs.append(m['accuracy'])
            precs.append(m['precision'])
            recs.append(m['recall'])
            f1s.append(m['f1_score'])
            
        print("-"*55)
        print(f"AVERAGE  | {np.mean(accs):<10.4f} | {np.mean(precs):<10.4f} | {np.mean(recs):<10.4f} | {np.mean(f1s):<10.4f}")
        print("="*50)
        
    finally:
        # Restore original epochs config
        config.TRANSFORMER_EPOCHS = old_tf_epochs
        config.CNN_EPOCHS = old_cnn_epochs

if __name__ == "__main__":
    run_4_folds_transformer()
