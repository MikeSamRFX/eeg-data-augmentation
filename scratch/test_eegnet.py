import os
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Setup GPU growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Create a small dummy dataset
X = np.random.randn(64, 6, 1024, 1).astype(np.float32)
y = np.random.randint(0, 2, (64, 1)).astype(np.float32)

# Build EEGNet
model = models.Sequential()
model.add(layers.Input(shape=(6, 1024, 1)))
model.add(layers.Conv2D(8, (1, 64), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.DepthwiseConv2D((6, 1), use_bias=False, depth_multiplier=2))
model.add(layers.BatchNormalization())
model.add(layers.Activation('elu'))
model.add(layers.AveragePooling2D((1, 4)))
model.add(layers.Dropout(0.5))
model.add(layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('elu'))
model.add(layers.AveragePooling2D((1, 8)))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Starting model fit...")
model.fit(X, y, epochs=5, batch_size=32)
print("Model fit completed successfully!")
