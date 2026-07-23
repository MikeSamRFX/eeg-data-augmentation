import os
import sys
import tempfile
import json
import subprocess
import csv
from datetime import datetime
import numpy as np
import optuna
import tensorflow as tf

# Setup paths and environment
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))

import config
from folding import get_subject_folds, load_split_data
from classifier import train_eegnet, evaluate_eegnet, standardize_features
from augmentations import fit_and_refine_transformer
from models import build_standard_transformer, build_slowdown_transformer

if len(sys.argv) < 2:
    print("Usage: python scratch/tune_transformers_4folds.py <Standard_Transformer|Slowdown_Transformer> [n_trials]")
    sys.exit(1)

transformer_type = sys.argv[1]
n_trials = 12
if len(sys.argv) > 2:
    try:
        n_trials = int(sys.argv[2])
    except ValueError:
        pass

RESULTS_CSV = os.path.join(os.path.dirname(__file__), f"results_4fold_optuna_{transformer_type}.csv")

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
                'EEGNet_4Fold_Acc',
                'ASD_4Fold_Acc',
                'Combined_Mean_Acc',
                'Status'
            ])

def log_to_csv(trial_num, params, eegnet_acc, asd_acc, comb_acc, status="Success"):
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
            round(eegnet_acc, 4),
            round(asd_acc, 4),
            round(comb_acc, 4),
            status
        ])

def objective(trial):
    # Suggest search space parameters
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
    transformer_lr = trial.suggest_categorical("transformer_lr", [1e-4, 5e-4, 1e-3])
    transformer_epochs = trial.suggest_categorical("transformer_epochs", [15, 25])
    transformer_batch = trial.suggest_categorical("transformer_batch", [16, 32])
    transformer_mask_ratio = trial.suggest_categorical("transformer_mask_ratio", [0.15, 0.25, 0.35])
    
    slowdown_noise = 0.0
    slowdown_dropout = 0.0
    if transformer_type == "Slowdown_Transformer":
        slowdown_noise = trial.suggest_categorical("slowdown_noise", [0.005, 0.01, 0.03])
        slowdown_dropout = trial.suggest_categorical("slowdown_dropout", [0.0, 0.1, 0.2])

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

    print(f"\n--- Starting Trial {trial.number} (4 Folds) | {transformer_type} ---")
    print(f"Params: {params}")

    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
    else:
        splits = get_subject_folds(config.PROCESSED_DIR)

    eegnet_accs = []
    asd_accs = []

    try:
        build_fn = build_slowdown_transformer if transformer_type == "Slowdown_Transformer" else build_standard_transformer

        for split in splits:
            fold_idx = split["fold_idx"]
            X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
            X_train_std, X_test_std = standardize_features(X_train, X_test)

            # Fit transformer and extract refined features
            X_tr_ref, _ = fit_and_refine_transformer(
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
            tf.keras.backend.clear_session()

            # We test classifier trained on concatenated real + transformer signals
            X_tr_aug = np.concatenate([X_train_std, X_tr_ref], axis=0)
            y_tr_aug = np.concatenate([y_train, y_train], axis=0)
            chans = X_tr_aug.shape[1]

            # 1. EEGNet evaluation
            model_eeg = train_eegnet(
                X_tr_aug, y_tr_aug,
                chans=chans,
                filters_conv1=config.EEGNET_FILTERS_CONV1,
                filters_conv2=config.EEGNET_FILTERS_CONV2,
                dropout=config.EEGNET_DROPOUT,
                learning_rate=config.EEGNET_LR,
                batch_size=config.EEGNET_BATCH,
                epochs=config.EEGNET_EPOCHS
            )
            metrics_eeg, _ = evaluate_eegnet(model_eeg, X_test_std, y_test)
            eegnet_accs.append(metrics_eeg["accuracy"])
            tf.keras.backend.clear_session()

            # 2. ASDClassifier evaluation (Subprocess)
            with tempfile.TemporaryDirectory() as tmpdir:
                X_tr_p = os.path.join(tmpdir, "X_tr.npy")
                y_tr_p = os.path.join(tmpdir, "y_tr.npy")
                X_te_p = os.path.join(tmpdir, "X_te.npy")
                y_te_p = os.path.join(tmpdir, "y_te.npy")
                out_p = os.path.join(tmpdir, "res.json")
                np.save(X_tr_p, X_tr_aug)
                np.save(y_tr_p, y_tr_aug)
                np.save(X_te_p, X_test_std)
                np.save(y_te_p, y_test)

                cmd = [
                    sys.executable,
                    os.path.join(config.BASE_DIR, "train_classifier.py"),
                    X_tr_p, y_tr_p, X_te_p, y_te_p, str(chans), out_p
                ]
                subprocess.run(cmd, check=True)
                with open(out_p, 'r') as f:
                    metrics_asd = json.load(f)
                asd_accs.append(metrics_asd["accuracy"])

        mean_eegnet = float(np.mean(eegnet_accs))
        mean_asd = float(np.mean(asd_accs))
        combined_score = (mean_eegnet + mean_asd) / 2.0

        print(f"Trial {trial.number} Finished | EEGNet 4-Fold Acc: {mean_eegnet:.4f} | ASD 4-Fold Acc: {mean_asd:.4f} | Combined: {combined_score:.4f}")
        log_to_csv(trial.number, params, mean_eegnet, mean_asd, combined_score, "Success")
        return combined_score

    except Exception as e:
        print(f"Trial {trial.number} Error: {e}")
        log_to_csv(trial.number, params, 0.0, 0.0, 0.0, f"Failed: {str(e)}")
        return 0.0

def main():
    initialize_csv()
    print(f" Starting 4-Fold Optuna tuning for {transformer_type} ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n================ OPTIMIZATION FINISHED ================")
    print("Best Trial:")
    t = study.best_trial
    print(f"  Combined Mean 4-Fold Accuracy: {t.value:.4f}")
    print("  Best Hyperparameters:")
    for k, v in t.params.items():
        print(f"    {k}: {v}")

    out_json = os.path.join(os.path.dirname(__file__), f"best_params_{transformer_type}.json")
    with open(out_json, "w") as f:
        json.dump(t.params, f, indent=4)
    print(f"Saved best params to {out_json}")

if __name__ == "__main__":
    main()
