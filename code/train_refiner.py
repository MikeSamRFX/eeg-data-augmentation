import os
import sys
import numpy as np
import tensorflow as tf

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from models import build_standard_transformer, build_slowdown_transformer

def apply_random_mask(X, mask_ratio=0.15):
    """
    Applies random masking to EEG signals.
    X shape: (Batch, Channels, Samples)
    Sets a random subset of samples to 0.0 along the time dimension.
    """
    X_masked = X.copy()
    num_samples = X.shape[2]
    num_mask = int(num_samples * mask_ratio)
    
    for i in range(X.shape[0]):
        # Randomly choose time indices to mask
        mask_indices = np.random.choice(num_samples, num_mask, replace=False)
        X_masked[i, :, mask_indices] = 0.0
        
    return X_masked

def main():
    if len(sys.argv) < 6:
        print("Usage: train_refiner.py <X_train_path> <y_train_path> <X_test_path> <out_train_path> <out_test_path> <build_fn_name> <name>")
        sys.exit(1)
        
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    X_test_path = sys.argv[3]
    out_train_path = sys.argv[4]
    out_test_path = sys.argv[5]
    build_fn_name = sys.argv[6]
    name = sys.argv[7]
    
    # Enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pass
            
    # Load inputs
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    
    # Read overrides or default to config
    d_model = int(os.environ.get("D_MODEL", config.D_MODEL))
    num_heads = int(os.environ.get("NUM_HEADS", config.NUM_HEADS))
    ff_dim = int(os.environ.get("FF_DIM", config.FF_DIM))
    epochs = int(os.environ.get("TRANSFORMER_EPOCHS", config.TRANSFORMER_EPOCHS))
    batch_size = int(os.environ.get("TRANSFORMER_BATCH", config.TRANSFORMER_BATCH))
    lr = float(os.environ.get("TRANSFORMER_LR", config.TRANSFORMER_LR))
    mask_ratio = float(os.environ.get("TRANSFORMER_MASK_RATIO", config.TRANSFORMER_MASK_RATIO))
    slowdown_factor = int(os.environ.get("SLOWDOWN_FACTOR", getattr(config, "SLOWDOWN_FACTOR", 2)))
    slowdown_noise = float(os.environ.get("SLOWDOWN_NOISE", getattr(config, "SLOWDOWN_NOISE", 0.01)))
    slowdown_dropout = float(os.environ.get("SLOWDOWN_DROPOUT", getattr(config, "SLOWDOWN_DROPOUT", 0.2)))

    # Apply random masking for Masked Autoencoder training
    X_train_masked = apply_random_mask(X_train, mask_ratio=mask_ratio)
    
    # Build models
    if build_fn_name == "build_slowdown_transformer":
        full_model, refiner_model = build_slowdown_transformer(
            input_shape=(config.CHANNELS, config.SAMPLES),
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            slowdown_factor=slowdown_factor,
            slowdown_noise=slowdown_noise,
            slowdown_dropout=slowdown_dropout
        )
    else:
        full_model, refiner_model = build_standard_transformer(
            input_shape=(config.CHANNELS, config.SAMPLES),
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim
        )
        
    print(f"Subprocess: Training {name} Autoencoder on GPU...")
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        jit_compile=False,
        run_eagerly=False
    )
    
    # Train model to reconstruct original unmasked signal from masked signal and class label
    full_model.fit(
        [X_train_masked, y_train], X_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print(f"Subprocess: Extracting {name}-Reconstructed features...")
    X_train_ref = []
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        X_train_ref.append(np.array(refiner_model([batch_x, batch_y], training=False)))
    X_train_ref = np.transpose(np.concatenate(X_train_ref, axis=0), (0, 2, 1))

    X_test_ref = []
    dummy_y_test = np.zeros((len(X_test), 1))
    for i in range(0, len(X_test), batch_size):
        batch_x = X_test[i:i+batch_size]
        batch_y = dummy_y_test[i:i+batch_size]
        X_test_ref.append(np.array(refiner_model([batch_x, batch_y], training=False)))
    X_test_ref = np.transpose(np.concatenate(X_test_ref, axis=0), (0, 2, 1))
    
    # Save outputs
    np.save(out_train_path, X_train_ref)
    np.save(out_test_path, X_test_ref)
    print("Subprocess completed successfully!")

if __name__ == "__main__":
    main()
