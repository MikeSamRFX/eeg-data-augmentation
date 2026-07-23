import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Directories & Paths ---
RAW_DIR = BASE_DIR.parent / "data-set" / "raw"
PROCESSED_DIR = BASE_DIR.parent / "data-set" / "processed"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"
RESULTS_FILE = BASE_DIR.parent / "experiment_results_both.csv"

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- EEG Signal Properties ---
CHANNELS = 6            # Number of ROI groups / channels
SAMPLES = 1024          # Time points per slice (2 seconds at 512Hz)
SAMPLING_RATE = 512     # Sampling frequency in Hz
SLICE_DURATION = 2.0    # Duration of slices in seconds
SLICES_PER_FILE = 50    # Number of slices to extract per subject

# --- EEGNet Optimal Hyperparameters ---
EEGNET_EPOCHS = 20
EEGNET_BATCH = 16
EEGNET_LR = 1e-5
EEGNET_FILTERS_CONV1 = 8
EEGNET_FILTERS_CONV2 = 64
EEGNET_DROPOUT = 0.1

# --- ASDClassifier Optimal Hyperparameters ---
ASD_EPOCHS = 30
ASD_BATCH = 32
ASD_LR = 1e-5
ASD_FILTERS_CONV1 = 8
ASD_FILTERS_CONV2 = 64
ASD_FILTERS_CONV3 = 32
ASD_KERNEL_SIZE = (5, 5)
ASD_DROPOUT = 0.1

# --- Legacy Default Properties (Mapped to ASDClassifier parameters) ---
CNN_EPOCHS = ASD_EPOCHS
CNN_BATCH = ASD_BATCH
LEARNING_RATE = ASD_LR
FILTERS_CONV1 = ASD_FILTERS_CONV1
FILTERS_CONV2 = ASD_FILTERS_CONV2
FILTERS_CONV3 = ASD_FILTERS_CONV3
KERNEL_SIZE = ASD_KERNEL_SIZE
DROPOUT = ASD_DROPOUT

# --- Standard Transformer Optimal parameters ---
STANDARD_D_MODEL = 32
STANDARD_NUM_HEADS = 2
STANDARD_FF_DIM = 256
STANDARD_LR = 0.0001
STANDARD_EPOCHS = 15
STANDARD_BATCH = 8
STANDARD_MASK_RATIO = 0.25

# --- Slowdown Transformer Optimal parameters ---
SLOWDOWN_D_MODEL = 32
SLOWDOWN_NUM_HEADS = 2
SLOWDOWN_FF_DIM = 128
SLOWDOWN_LR = 0.0005
SLOWDOWN_EPOCHS = 25
SLOWDOWN_BATCH = 8
SLOWDOWN_MASK_RATIO = 0.35
SLOWDOWN_NOISE = 0.03
SLOWDOWN_DROPOUT = 0.2
SLOWDOWN_FACTOR = 2

# Legacy default properties (fallback values)
TRANSFORMER_EPOCHS = STANDARD_EPOCHS
TRANSFORMER_BATCH = STANDARD_BATCH
D_MODEL = STANDARD_D_MODEL
NUM_HEADS = STANDARD_NUM_HEADS
FF_DIM = STANDARD_FF_DIM
TRANSFORMER_LR = STANDARD_LR
TRANSFORMER_MASK_RATIO = STANDARD_MASK_RATIO
SLOWDOWN_FACTOR = 2
SLOWDOWN_NOISE = SLOWDOWN_NOISE
SLOWDOWN_DROPOUT = SLOWDOWN_DROPOUT

# --- Augmentation Parameters ---
JITTER_LEVEL = 0.02
SCALING_SIGMA = 0.1
TIME_SHIFT_LIMIT = 50   # Max samples for sliding window / shift logic if needed
SLIDING_WINDOW_OVERLAP = 0.5 # 50% overlap for sliding window augmentation

# --- Classifier Selection ---
CLASSIFIER_TYPE = "Both"  # Options: "EEGNet", "ASDClassifier", "Both"