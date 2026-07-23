import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))
import config
from models import build_standard_transformer

# Setup GPU growth for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Generate dummy data
X_dummy = np.random.normal(0, 1, (100, config.CHANNELS, config.SAMPLES))
y_dummy = np.random.randint(0, 2, (100,))

print("Building transformer model...")
full_model, refiner_model = build_standard_transformer(
    input_shape=(config.CHANNELS, config.SAMPLES),
    d_model=config.D_MODEL
)

full_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting fit on GPU...")
full_model.fit(X_dummy, y_dummy, epochs=2, batch_size=32)
print("TF fit on GPU succeeded!")
