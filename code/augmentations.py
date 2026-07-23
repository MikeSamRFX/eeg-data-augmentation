import os
import subprocess
import tempfile
import sys
import numpy as np
import tensorflow as tf
import config
from models import build_standard_transformer, build_slowdown_transformer

# 1. Jittering
def apply_jitter(X, jitter_level=None):
    """
    Adds random Gaussian noise to the signal slices.
    X shape: (Batch, Channels, Samples)
    """
    if jitter_level is None:
        jitter_level = config.JITTER_LEVEL
    noise = np.random.normal(0, jitter_level, X.shape)
    return X + noise

# 2. Scaling
def apply_scaling(X, sigma=None):
    """
    Randomly scales the amplitude of each signal slice.
    X shape: (Batch, Channels, Samples)
    """
    if sigma is None:
        sigma = config.SCALING_SIGMA
    # Generate a random scaling factor for each slice in the batch
    factors = np.random.normal(1.0, sigma, (X.shape[0], 1, 1))
    return X * factors

# 3. Sliding Window
def apply_sliding_window(continuous_data, window_size=1024, overlap=None):
    """
    Extracts overlapping slices from continuous ROI matrices to augment dataset.
    continuous_data: list of tuples (roi_matrix, label, subject_id)
    """
    if overlap is None:
        overlap = config.SLIDING_WINDOW_OVERLAP
        
    stride = int(window_size * (1.0 - overlap))
    X_aug, y_aug = [], []
    
    for roi_matrix, label, _ in continuous_data:
        # roi_matrix shape: (Channels, TimePoints)
        num_timepoints = roi_matrix.shape[1]
        start = 0
        while start + window_size <= num_timepoints:
            slice_data = roi_matrix[:, start:start+window_size]
            X_aug.append(slice_data)
            y_aug.append(label)
            start += stride
            
    if not X_aug:
        raise ValueError("No slices extracted during sliding window. Check signal lengths.")
        
    return np.array(X_aug), np.array(y_aug)

# 4. Classic Mixup
def apply_mixup(X, y, alpha=0.2):
    """
    Blends pairs of training slices and their labels.
    Returns concatenated (original + mixed) training set.
    """
    batch_size = X.shape[0]
    indices = np.random.permutation(batch_size)
    
    # Sample mixing weight from Beta distribution
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1, 1))
    lam_y = lam.squeeze(-1).squeeze(-1) # Shape (Batch,)
    
    X_mixed = lam * X + (1.0 - lam) * X[indices]
    y_mixed = lam_y * y + (1.0 - lam_y) * y[indices]
    
    # Combine original data and augmented data
    X_combined = np.concatenate([X, X_mixed], axis=0)
    y_combined = np.concatenate([y, y_mixed], axis=0)
    return X_combined, y_combined

# 5. Fit Transformer and Refine
def fit_and_refine_transformer(X_train, y_train, X_test, build_fn, name="Transformer", **kwargs):
    """
    Helper function to build and train a Transformer model to learn
    representations, and use its encoder to produce 64-channel refined data.
    Runs the TensorFlow training in a separate subprocess to completely isolate
    the TensorFlow GPU context and release all of its VRAM before PyTorch is invoked.
    """
    build_fn_name = build_fn.__name__
    
    with tempfile.TemporaryDirectory() as tmpdir:
        X_train_path = os.path.join(tmpdir, "X_train.npy")
        y_train_path = os.path.join(tmpdir, "y_train.npy")
        X_test_path = os.path.join(tmpdir, "X_test.npy")
        out_train_path = os.path.join(tmpdir, "X_train_ref.npy")
        out_test_path = os.path.join(tmpdir, "X_test_ref.npy")
        
        np.save(X_train_path, X_train)
        np.save(y_train_path, y_train)
        np.save(X_test_path, X_test)
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "train_refiner.py"),
            X_train_path,
            y_train_path,
            X_test_path,
            out_train_path,
            out_test_path,
            build_fn_name,
            name
        ]
        
        env = os.environ.copy()
        for k, v in kwargs.items():
            if v is not None:
                env[k.upper()] = str(v)
        
        print(f"Running refiner training subprocess for {name}...")
        subprocess.run(cmd, env=env, check=True)
        
        X_train_ref = np.load(out_train_path)
        X_test_ref = np.load(out_test_path)
        
    return X_train_ref, X_test_ref
