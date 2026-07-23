import os
import subprocess
import tempfile
import sys
import json
import numpy as np
import tensorflow as tf
# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass
import config
from preprocess import run_preprocessing
from folding import get_subject_folds, load_split_data, get_continuous_train_data
from augmentations import (
    apply_jitter, apply_scaling, apply_sliding_window, apply_mixup,
    fit_and_refine_transformer
)
from classifier import (
    train_eegnet, evaluate_eegnet, log_results_to_csv,
    train_pytorch_classifier, evaluate_pytorch_classifier,
    standardize_features
)
from models import build_standard_transformer, build_slowdown_transformer
from digest_results import digest_results

STAGES = [
    "preprocess",
    "folding",
    "baseline_raw",
    "jittering",
    "jittering_combined",
    "scaling",
    "scaling_combined",
    "slide_window",
    "slide_window_combined",
    "transformer",
    "transformer_combined",
    "slowdown",
    "slowdown_combined",
    "mixup",
    "mixup_transformer",
    "mixup_slowdown"
]

def save_checkpoint(stage, results):
    """
    Saves the pipeline state to checkpoint.json.
    """
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.json")
    state = {
        "current_stage": stage,
        "results": results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=4)
    print(f" Checkpoint saved: stage '{stage}' is completed.")

def load_checkpoint():
    """
    Loads pipeline state from checkpoint.json if it exists.
    """
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            return state["current_stage"], state["results"]
        except Exception as e:
            print(f"Warning: Failed to load checkpoint.json ({e}). Starting fresh.")
    return None, []

def clear_checkpoint():
    """
    Clears the checkpoint file.
    """
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

def run_pipeline(force_restart=False, resume=True, n_folds=4):
    """
    Orchestrates the sequential training pipelines and manages checkpoints.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass
            
    # Dict caches to share augmented data and refined features across stages to avoid recomputation
    cached_jitter = {}
    cached_scale = {}
    cached_sw_x = {}
    cached_sw_y = {}
    cached_ref_x = {}
    cached_ref_test = {}
    
    # 2. Check checkpoint status
    current_stage = None
    results = []
    
    if force_restart:
        print(" Force restart requested. Clearing old checkpoints...")
        clear_checkpoint()
        if os.path.exists(config.RESULTS_FILE):
            os.remove(config.RESULTS_FILE)
    elif resume:
        checkpoint_stage, checkpoint_results = load_checkpoint()
        if checkpoint_stage:
            current_stage = checkpoint_stage
            results = checkpoint_results
            print(f" Resuming pipeline from stage: '{current_stage}'...")
            
    # Find starting index in STAGES
    start_idx = 0
    if current_stage in STAGES:
        start_idx = STAGES.index(current_stage) + 1 # Next stage to run
        
    # Load folds/splits if already created
    splits = None
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
    # 3. Pipeline loop
    for idx in range(start_idx, len(STAGES)):
        stage = STAGES[idx]
        print(f"\n === RUNNING STAGE: {stage.upper()} ===")
        
        if stage == "preprocess":
            run_preprocessing()
            save_checkpoint(stage, results)
            
        elif stage == "folding":
            splits = get_subject_folds(config.PROCESSED_DIR, n_folds=n_folds)
            save_checkpoint(stage, results)
            
        else:
            # Classification and Augmentation stages
            if splits is None:
                raise ValueError("Splits must be initialized before classification stages.")
                
            stage_metrics_eegnet = []
            stage_metrics_asd = []
            
            for split in splits:
                fold_idx = split["fold_idx"]
                print(f"\n--- Fold {fold_idx}/{len(splits)} ---")
                
                # Clear session to release GPU memory OOM accumulation
                tf.keras.backend.clear_session()
                
                # Load train/test data
                X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
                
                # Standardize raw signals fold-level channel-wise before augmentation
                X_train_std, X_test_std = standardize_features(X_train, X_test)
                
                # Apply Augmentation corresponding to stage
                if stage == "baseline_raw":
                    # Raw baseline (standardized) - No augmentation
                    X_tr_aug, y_tr_aug = X_train_std, y_train
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "jittering":
                    X_tr_aug = apply_jitter(X_train_std)
                    cached_jitter[fold_idx] = X_tr_aug
                    y_tr_aug = y_train
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "jittering_combined":
                    if fold_idx in cached_jitter:
                        X_jitter = cached_jitter[fold_idx]
                    else:
                        X_jitter = apply_jitter(X_train_std)
                    X_tr_aug = np.concatenate([X_train_std, X_jitter], axis=0)
                    y_tr_aug = np.concatenate([y_train, y_train], axis=0)
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "scaling":
                    X_tr_aug = apply_scaling(X_train_std)
                    cached_scale[fold_idx] = X_tr_aug
                    y_tr_aug = y_train
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "scaling_combined":
                    if fold_idx in cached_scale:
                        X_scale = cached_scale[fold_idx]
                    else:
                        X_scale = apply_scaling(X_train_std)
                    X_tr_aug = np.concatenate([X_train_std, X_scale], axis=0)
                    y_tr_aug = np.concatenate([y_train, y_train], axis=0)
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "slide_window":
                    continuous_data = get_continuous_train_data(config.PROCESSED_DIR, split)
                    X_tr_aug_raw, y_tr_aug = apply_sliding_window(continuous_data)
                    X_tr_aug, X_te = standardize_features(X_tr_aug_raw, X_test)
                    cached_sw_x[fold_idx] = X_tr_aug
                    cached_sw_y[fold_idx] = y_tr_aug
                    y_te = y_test
                    chans = config.CHANNELS
                    
                elif stage == "slide_window_combined":
                    if fold_idx in cached_sw_x:
                        X_sw = cached_sw_x[fold_idx]
                        y_sw = cached_sw_y[fold_idx]
                    else:
                        continuous_data = get_continuous_train_data(config.PROCESSED_DIR, split)
                        X_tr_aug_raw, y_sw = apply_sliding_window(continuous_data)
                        X_sw, X_te = standardize_features(X_tr_aug_raw, X_test)
                    X_tr_aug = np.concatenate([X_train_std, X_sw], axis=0)
                    y_tr_aug = np.concatenate([y_train, y_sw], axis=0)
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "transformer":
                    X_tr_aug, _ = fit_and_refine_transformer(
                        X_train_std, y_train, X_test_std, build_standard_transformer, "Standard_Transformer",
                        d_model=config.STANDARD_D_MODEL,
                        num_heads=config.STANDARD_NUM_HEADS,
                        ff_dim=config.STANDARD_FF_DIM,
                        transformer_lr=config.STANDARD_LR,
                        transformer_epochs=config.STANDARD_EPOCHS,
                        transformer_batch=config.STANDARD_BATCH,
                        transformer_mask_ratio=config.STANDARD_MASK_RATIO
                    )
                    cached_ref_x[fold_idx] = X_tr_aug
                    y_tr_aug, y_te = y_train, y_test
                    X_te = X_test_std
                    tf.keras.backend.clear_session()
                    
                elif stage == "transformer_combined":
                    if fold_idx in cached_ref_x:
                        X_ref = cached_ref_x[fold_idx]
                    else:
                        X_ref, _ = fit_and_refine_transformer(
                            X_train_std, y_train, X_test_std, build_standard_transformer, "Standard_Transformer",
                            d_model=config.STANDARD_D_MODEL,
                            num_heads=config.STANDARD_NUM_HEADS,
                            ff_dim=config.STANDARD_FF_DIM,
                            transformer_lr=config.STANDARD_LR,
                            transformer_epochs=config.STANDARD_EPOCHS,
                            transformer_batch=config.STANDARD_BATCH,
                            transformer_mask_ratio=config.STANDARD_MASK_RATIO
                        )
                    # Stack generated and real signals along the batch/sample dimension (axis 0)
                    X_tr_aug = np.concatenate([X_train_std, X_ref], axis=0)
                    y_tr_aug = np.concatenate([y_train, y_train], axis=0)
                    # Test classifier strictly on real signals
                    X_te = X_test_std
                    y_te = y_test
                    tf.keras.backend.clear_session()
                    
                elif stage == "slowdown":
                    X_tr_aug, _ = fit_and_refine_transformer(
                        X_train_std, y_train, X_test_std, build_slowdown_transformer, "Slowdown_Transformer",
                        d_model=config.SLOWDOWN_D_MODEL,
                        num_heads=config.SLOWDOWN_NUM_HEADS,
                        ff_dim=config.SLOWDOWN_FF_DIM,
                        transformer_lr=config.SLOWDOWN_LR,
                        transformer_epochs=config.SLOWDOWN_EPOCHS,
                        transformer_batch=config.SLOWDOWN_BATCH,
                        transformer_mask_ratio=config.SLOWDOWN_MASK_RATIO,
                        slowdown_noise=config.SLOWDOWN_NOISE,
                        slowdown_dropout=config.SLOWDOWN_DROPOUT,
                        slowdown_factor=config.SLOWDOWN_FACTOR
                    )
                    y_tr_aug, y_te = y_train, y_test
                    X_te = X_test_std
                    tf.keras.backend.clear_session()

                elif stage == "slowdown_combined":
                    X_ref_slow, _ = fit_and_refine_transformer(
                        X_train_std, y_train, X_test_std, build_slowdown_transformer, "Slowdown_Transformer",
                        d_model=config.SLOWDOWN_D_MODEL,
                        num_heads=config.SLOWDOWN_NUM_HEADS,
                        ff_dim=config.SLOWDOWN_FF_DIM,
                        transformer_lr=config.SLOWDOWN_LR,
                        transformer_epochs=config.SLOWDOWN_EPOCHS,
                        transformer_batch=config.SLOWDOWN_BATCH,
                        transformer_mask_ratio=config.SLOWDOWN_MASK_RATIO,
                        slowdown_noise=config.SLOWDOWN_NOISE,
                        slowdown_dropout=config.SLOWDOWN_DROPOUT,
                        slowdown_factor=config.SLOWDOWN_FACTOR
                    )
                    X_tr_aug = np.concatenate([X_train_std, X_ref_slow], axis=0)
                    y_tr_aug = np.concatenate([y_train, y_train], axis=0)
                    X_te = X_test_std
                    y_te = y_test
                    tf.keras.backend.clear_session()

                elif stage == "mixup":
                    X_tr_aug, y_tr_aug = apply_mixup(X_train_std, y_train)
                    X_te, y_te = X_test_std, y_test
                    chans = config.CHANNELS
                    
                elif stage == "mixup_transformer":
                    # First refine
                    X_tr_ref, _ = fit_and_refine_transformer(
                        X_train_std, y_train, X_test_std, build_standard_transformer, "Standard_Transformer",
                        d_model=config.STANDARD_D_MODEL,
                        num_heads=config.STANDARD_NUM_HEADS,
                        ff_dim=config.STANDARD_FF_DIM,
                        transformer_lr=config.STANDARD_LR,
                        transformer_epochs=config.STANDARD_EPOCHS,
                        transformer_batch=config.STANDARD_BATCH,
                        transformer_mask_ratio=config.STANDARD_MASK_RATIO
                    )
                    # Apply Mixup on refined features
                    X_tr_aug, y_tr_aug = apply_mixup(X_tr_ref, y_train)
                    X_te = X_test_std
                    y_te = y_test
                    tf.keras.backend.clear_session()
                    
                elif stage == "mixup_slowdown":
                    # First refine
                    X_tr_ref, _ = fit_and_refine_transformer(
                        X_train_std, y_train, X_test_std, build_slowdown_transformer, "Slowdown_Transformer",
                        d_model=config.SLOWDOWN_D_MODEL,
                        num_heads=config.SLOWDOWN_NUM_HEADS,
                        ff_dim=config.SLOWDOWN_FF_DIM,
                        transformer_lr=config.SLOWDOWN_LR,
                        transformer_epochs=config.SLOWDOWN_EPOCHS,
                        transformer_batch=config.SLOWDOWN_BATCH,
                        transformer_mask_ratio=config.SLOWDOWN_MASK_RATIO,
                        slowdown_noise=config.SLOWDOWN_NOISE,
                        slowdown_dropout=config.SLOWDOWN_DROPOUT,
                        slowdown_factor=config.SLOWDOWN_FACTOR
                    )
                    # Apply Mixup on refined features
                    X_tr_aug, y_tr_aug = apply_mixup(X_tr_ref, y_train)
                    X_te = X_test_std
                    y_te = y_test
                    tf.keras.backend.clear_session()
                    
                else:
                    raise ValueError(f"Unknown stage: {stage}")
                    
                # Features are already standardized channel-wise before / during augmentations
                X_tr_aug_norm, X_te_norm = X_tr_aug, X_te
                chans = X_tr_aug_norm.shape[1]
                
                classifier_name = getattr(config, "CLASSIFIER_TYPE", "Both")
                
                # Train and evaluate EEGNet
                if classifier_name in ["EEGNet", "Both"]:
                    print(f"Training EEGNet with {chans} channels on fold {fold_idx}...")
                    model_eegnet = train_eegnet(
                        X_tr_aug_norm, y_tr_aug,
                        chans=chans,
                        filters_conv1=getattr(config, "EEGNET_FILTERS_CONV1", 8),
                        filters_conv2=getattr(config, "EEGNET_FILTERS_CONV2", 64),
                        dropout=getattr(config, "EEGNET_DROPOUT", 0.1),
                        learning_rate=getattr(config, "EEGNET_LR", 1e-5),
                        batch_size=getattr(config, "EEGNET_BATCH", 16),
                        epochs=getattr(config, "EEGNET_EPOCHS", 20)
                    )
                    
                    print(f"Evaluating EEGNet on fold {fold_idx}...")
                    metrics_eegnet, _ = evaluate_eegnet(model_eegnet, X_te_norm, y_te)
                    log_results_to_csv("EEGNet", stage, fold_idx, metrics_eegnet)
                    stage_metrics_eegnet.append(metrics_eegnet)
                
                # Train and evaluate ASDClassifier (Subprocess)
                if classifier_name in ["ASDClassifier", "Both"]:
                    print(f"Training ASDClassifier with {chans} channels on fold {fold_idx} (Subprocess)...")
                    with tempfile.TemporaryDirectory() as tmpdir:
                        X_train_path = os.path.join(tmpdir, "X_train.npy")
                        y_train_path = os.path.join(tmpdir, "y_train.npy")
                        X_test_path = os.path.join(tmpdir, "X_test.npy")
                        y_test_path = os.path.join(tmpdir, "y_test.npy")
                        out_results_json_path = os.path.join(tmpdir, "results.json")
                        
                        np.save(X_train_path, X_tr_aug_norm)
                        np.save(y_train_path, y_tr_aug)
                        np.save(X_test_path, X_te_norm)
                        np.save(y_test_path, y_te)
                        
                        cmd = [
                            sys.executable,
                            os.path.join(os.path.dirname(__file__), "train_classifier.py"),
                            X_train_path,
                            y_train_path,
                            X_test_path,
                            y_test_path,
                            str(chans),
                            out_results_json_path
                        ]
                        
                        subprocess.run(cmd, check=True)
                        
                        with open(out_results_json_path, 'r') as f:
                            metrics_asd = json.load(f)
                    
                    log_results_to_csv("ASDClassifier", stage, fold_idx, metrics_asd)
                    stage_metrics_asd.append(metrics_asd)
                
            # Compute average metrics across folds for final checkpoint overview
            res_entry = {"stage": stage}
            
            if len(stage_metrics_eegnet) > 0:
                avg_acc_eeg = np.mean([m["accuracy"] for m in stage_metrics_eegnet])
                avg_f1_eeg = np.mean([m["f1_score"] for m in stage_metrics_eegnet])
                res_entry.update({
                    "eegnet_avg_accuracy": avg_acc_eeg,
                    "eegnet_avg_f1_score": avg_f1_eeg
                })
            else:
                res_entry.update({
                    "eegnet_avg_accuracy": 0.0,
                    "eegnet_avg_f1_score": 0.0
                })
                
            if len(stage_metrics_asd) > 0:
                avg_acc_asd = np.mean([m["accuracy"] for m in stage_metrics_asd])
                avg_f1_asd = np.mean([m["f1_score"] for m in stage_metrics_asd])
                res_entry.update({
                    "asd_avg_accuracy": avg_acc_asd,
                    "asd_avg_f1_score": avg_f1_asd,
                    # For legacy compatibility
                    "avg_accuracy": avg_acc_asd,
                    "avg_f1_score": avg_f1_asd
                })
            else:
                res_entry.update({
                    "asd_avg_accuracy": 0.0,
                    "asd_avg_f1_score": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_f1_score": 0.0
                })
                
            results.append(res_entry)
            print(f" Stage {stage} completed!")
            if len(stage_metrics_eegnet) > 0:
                print(f"   EEGNet        - Avg Accuracy: {res_entry['eegnet_avg_accuracy']:.4f}, Avg F1: {res_entry['eegnet_avg_f1_score']:.4f}")
            if len(stage_metrics_asd) > 0:
                print(f"   ASDClassifier - Avg Accuracy: {res_entry['asd_avg_accuracy']:.4f}, Avg F1: {res_entry['asd_avg_f1_score']:.4f}")
            save_checkpoint(stage, results)
            
    print("\n === COMPARATIVE PIPELINE COMPLETED SUCCESSFULLY! ===")
    print("Accumulated Stage Averages:")
    print(f"{'Stage':<20} | {'EEGNet Acc':<12} | {'EEGNet F1':<12} | {'ASD Acc':<12} | {'ASD F1':<12}")
    print("-" * 75)
    for res in results:
        print(f"{res['stage']:<20} | {res.get('eegnet_avg_accuracy', 0.0):<12.4f} | {res.get('eegnet_avg_f1_score', 0.0):<12.4f} | {res.get('asd_avg_accuracy', 0.0):<12.4f} | {res.get('asd_avg_f1_score', 0.0):<12.4f}")
        
    # Clean up checkpoint since run finished successfully
    clear_checkpoint()

    # Digest 4-fold cross validation results and update Digested_results.csv
    try:
        digest_results()
    except Exception as e:
        print(f"Warning: Failed to digest results: {e}")
