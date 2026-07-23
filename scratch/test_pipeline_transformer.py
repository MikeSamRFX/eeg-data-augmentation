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

def run_one_fold_transformer():
    print(" Running a single fold test using the new Generative Standard Transformer...")
    
    # 1. Load splits configuration
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    if not os.path.exists(splits_path):
        print(" splits.json not found in processed directory. Generating folds first...")
        from folding import get_subject_folds
        splits = get_subject_folds(config.PROCESSED_DIR, n_folds=4)
    else:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
    # 2. Select Fold 1
    split = splits[0]
    fold_idx = split["fold_idx"]
    print(f"Loaded Split Fold: {fold_idx}")
    
    # 3. Load and standardize training/test slices
    X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
    X_train_std, X_test_std = standardize_features(X_train, X_test)
    print(f"X_train_std shape: {X_train_std.shape}")
    print(f"X_test_std shape: {X_test_std.shape}")
    
    # Temporarily set epochs small for quick verification if needed, or run full
    # Let's run with 2 epochs for refiner and 2 epochs for classifier to make it ultra-fast
    old_tf_epochs = config.TRANSFORMER_EPOCHS
    old_cnn_epochs = config.CNN_EPOCHS
    config.TRANSFORMER_EPOCHS = 2
    config.CNN_EPOCHS = 2
    
    try:
        # 4. Train Autoencoder and reconstruct signals
        print("\n--- Step 1: Training Transformer Autoencoder (2 epochs) ---")
        X_tr_aug, X_te = fit_and_refine_transformer(
            X_train_std, y_train, X_test_std, build_standard_transformer, "Standard_Transformer"
        )
        print(f"Successfully generated reconstructed training data shape: {X_tr_aug.shape}")
        print(f"Successfully generated reconstructed test data shape: {X_te.shape}")
        
        # 5. Determine channels dynamically
        chans = X_tr_aug.shape[1]
        print(f"Dynamic channels determined: {chans}")
        
        # 6. Train classifer on reconstructed signals
        print("\n--- Step 2: Training Classifier on reconstructed signals (2 epochs) ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            X_train_path = os.path.join(tmpdir, "X_train.npy")
            y_train_path = os.path.join(tmpdir, "y_train.npy")
            X_test_path = os.path.join(tmpdir, "X_test.npy")
            y_test_path = os.path.join(tmpdir, "y_test.npy")
            out_results_json_path = os.path.join(tmpdir, "results.json")
            
            np.save(X_train_path, X_tr_aug)
            np.save(y_train_path, y_train)
            np.save(X_test_path, X_test_std) # Test on original standardized signals as per updated pipeline
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
                
        print("\n Fold test completed successfully!")
        print("Metrics on original test signals:")
        print(json.dumps(metrics, indent=4))
        
    finally:
        # Restore epochs config
        config.TRANSFORMER_EPOCHS = old_tf_epochs
        config.CNN_EPOCHS = old_cnn_epochs

if __name__ == "__main__":
    run_one_fold_transformer()
