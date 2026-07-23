import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
import config
from models import build_standard_transformer, build_slowdown_transformer

def test_reconstruction_shapes():
    print("TensorFlow Version:", tf.__version__)
    
    # 1. Test Standard Transformer Autoencoder
    print("\n--- Testing Standard Transformer Autoencoder ---")
    full_model_std, ref_model_std = build_standard_transformer(
        input_shape=(config.CHANNELS, config.SAMPLES), 
        d_model=config.D_MODEL
    )
    
    # Compile with MSE loss
    full_model_std.compile(optimizer='adam', loss='mse')
    print("Standard Transformer compiled successfully!")
    
    # Mock data
    dummy_input = np.random.normal(size=(5, config.CHANNELS, config.SAMPLES)) # shape (5, 6, 1024)
    dummy_label = np.random.randint(0, 2, size=(5, 1)) # shape (5, 1)
    
    reconstructed = full_model_std.predict([dummy_input, dummy_label])
    print("Input shape:", dummy_input.shape)
    print("Reconstructed output shape:", reconstructed.shape)
    
    # Verify refiner output shape (used for feature/reconstructed data extraction in subprocess)
    refiner_out = ref_model_std.predict([dummy_input, dummy_label])
    print("Refiner model output shape (non-transposed):", refiner_out.shape)
    # Perform the transpose done in train_refiner.py
    extracted_features = np.transpose(refiner_out, (0, 2, 1))
    print("Extracted/Transposed signal shape:", extracted_features.shape)
    
    assert reconstructed.shape == dummy_input.shape, "Standard reconstruction shape mismatch!"
    assert extracted_features.shape == dummy_input.shape, "Standard extracted feature shape mismatch!"
    
    # 2. Test Slowdown Transformer Autoencoder
    print("\n--- Testing Slowdown Transformer Autoencoder ---")
    full_model_slow, ref_model_slow = build_slowdown_transformer(
        input_shape=(config.CHANNELS, config.SAMPLES), 
        d_model=config.D_MODEL
    )
    
    full_model_slow.compile(optimizer='adam', loss='mse')
    print("Slowdown Transformer compiled successfully!")
    
    reconstructed_slow = full_model_slow.predict([dummy_input, dummy_label])
    print("Input shape:", dummy_input.shape)
    print("Reconstructed output shape:", reconstructed_slow.shape)
    
    refiner_out_slow = ref_model_slow.predict([dummy_input, dummy_label])
    print("Refiner model output shape (non-transposed):", refiner_out_slow.shape)
    extracted_features_slow = np.transpose(refiner_out_slow, (0, 2, 1))
    print("Extracted/Transposed signal shape:", extracted_features_slow.shape)
    
    assert reconstructed_slow.shape == dummy_input.shape, "Slowdown reconstruction shape mismatch!"
    assert extracted_features_slow.shape == dummy_input.shape, "Slowdown extracted feature shape mismatch!"
    
    print("\n All tests passed! Reconstruction shapes and compilation are 100% correct.")

if __name__ == "__main__":
    test_reconstruction_shapes()
