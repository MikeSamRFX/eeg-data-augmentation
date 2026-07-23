import os
# Import torch first to prevent CUDA library collision and segmentation faults with TensorFlow
try:
    import torch
except ImportError:
    pass
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path
import config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential EEG ASD Classification Pipeline with Data Augmentations and Checkpoints"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to the directory containing raw .set / .fdt files."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to the output directory for processed files."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to the CSV file to write results."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the pipeline execution from the last saved checkpoint."
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Restart the pipeline from the beginning, deleting previous checkpoints and results."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of cross-validation folds (default: 4)."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force TensorFlow and PyTorch to execute on CPU."
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        choices=["EEGNet", "ASDClassifier"],
        help="Classifier model implementation to use (EEGNet or ASDClassifier)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Force CPU if requested
    if args.no_gpu:
        print(" Forcing CPU execution...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        
    # 2. Dynamically override config paths if provided
    if args.raw_dir:
        config.RAW_DIR = Path(args.raw_dir).resolve()
        print(f"Config Override: RAW_DIR = {config.RAW_DIR}")
    if args.processed_dir:
        config.PROCESSED_DIR = Path(args.processed_dir).resolve()
        # Recreate directory if override changed it
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        print(f"Config Override: PROCESSED_DIR = {config.PROCESSED_DIR}")
    if args.results_file:
        config.RESULTS_FILE = Path(args.results_file).resolve()
        print(f"Config Override: RESULTS_FILE = {config.RESULTS_FILE}")
    if args.classifier:
        config.CLASSIFIER_TYPE = args.classifier
        print(f"Config Override: CLASSIFIER_TYPE = {config.CLASSIFIER_TYPE}")
        
    # Import pipeline here to ensure any path overrides in config are reflected before import dependencies
    from pipeline import run_pipeline
    
    # 3. Run orchestrator
    resume_flag = args.resume
    if not args.resume and not args.force_restart:
        # Default behavior: if checkpoint exists, ask/default to resume, otherwise start fresh
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            print(" Found an existing checkpoint. Defaulting to --resume. Use --force-restart to start fresh.")
            resume_flag = True
            
    run_pipeline(
        force_restart=args.force_restart,
        resume=resume_flag,
        n_folds=args.folds
    )

if __name__ == "__main__":
    main()