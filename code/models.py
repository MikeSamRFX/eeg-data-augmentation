import tensorflow as tf
from tensorflow.keras import layers, models

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.sequence_length, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="pos_emb"
        )
        super().build(input_shape)
        
    def call(self, inputs):
        return inputs + self.pos_emb

# =====================================================================
# 1. TENSORFLOW / KERAS MODELS
# =====================================================================

def transformer_block(x, d_model, num_heads, ff_dim):
    """
    Single Transformer Encoder Block with multi-head self-attention and position-wise feedforward network.
    """
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.LayerNormalization()(x + attn)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(d_model)(ffn)
    return layers.LayerNormalization()(x + ffn)

def build_standard_transformer(input_shape=(6, 1024), d_model=None, num_heads=None, ff_dim=None):
    """
    Conditional Standard Transformer autoencoder that reconstructs EEG inputs.
    Input shape: (Channels, Samples) e.g., (6, 1024)
    Conditioned on binary label (1,).
    """
    import config
    if d_model is None:
        d_model = getattr(config, "D_MODEL", 64)
    if num_heads is None:
        num_heads = getattr(config, "NUM_HEADS", 4)
    if ff_dim is None:
        ff_dim = getattr(config, "FF_DIM", 128)

    inputs_eeg = layers.Input(shape=input_shape, name="input_eeg")
    inputs_label = layers.Input(shape=(1,), name="input_label")
    
    # --- ENCODER ---
    # Permute to (Samples, Channels) for the transformer block processing
    x = layers.Permute((2, 1))(inputs_eeg)
    x = layers.Dense(d_model)(x)  # Project channels to latent space representation (1024, 64)
    x = PositionalEmbedding(input_shape[1], d_model)(x)

    # Stack of two transformer blocks
    x = transformer_block(x, d_model, num_heads, ff_dim)
    enc_out = transformer_block(x, d_model, num_heads, ff_dim)

    # --- LABEL CONDITIONING ---
    # Embed binary label to d_model space: (Batch, 1, d_model)
    label_emb = layers.Embedding(input_dim=2, output_dim=d_model)(inputs_label)
    label_emb = layers.Reshape((d_model,))(label_emb) # (Batch, d_model)
    # Repeat label embedding across the time dimension: (Batch, 1024, d_model)
    label_emb = layers.RepeatVector(input_shape[1])(label_emb)
    
    # Add label conditioning to encoder representation
    z = layers.Add()([enc_out, label_emb])

    # Latent space perturbation for stochastic reconstruction/generation (active only in training)
    z = layers.GaussianNoise(0.05)(z)

    # --- DECODER ---
    # Transformer decoder block
    dec = transformer_block(z, d_model, num_heads, ff_dim)
    # Project back to original channels
    dec = layers.Dense(input_shape[0])(dec) # (1024, 6)
    
    # Permute back to (Channels, Samples) for the full model reconstruction target
    outputs = layers.Permute((2, 1))(dec) # (6, 1024)

    full_model = models.Model(inputs=[inputs_eeg, inputs_label], outputs=outputs, name="Conditional_Standard_Transformer")
    # feature_model returns dec (1024, 6) which is transposed to (6, 1024) in train_refiner.py
    feature_model = models.Model(inputs=[inputs_eeg, inputs_label], outputs=dec, name="Conditional_Standard_Generator")
    return full_model, feature_model

def build_slowdown_transformer(input_shape=(6, 1024), d_model=None, num_heads=None, ff_dim=None, slowdown_factor=None, slowdown_noise=None, slowdown_dropout=None):
    """
    Conditional Slowdown Transformer autoencoder containing an encoder, temporal bottleneck (Slowdown), and decoder.
    Reconstructs EEG inputs.
    Conditioned on binary label (1,).
    """
    import config
    if d_model is None:
        d_model = getattr(config, "D_MODEL", 64)
    if num_heads is None:
        num_heads = getattr(config, "NUM_HEADS", 4)
    if ff_dim is None:
        ff_dim = getattr(config, "FF_DIM", 128)
    if slowdown_factor is None:
        slowdown_factor = getattr(config, "SLOWDOWN_FACTOR", 2)
    if slowdown_noise is None:
        slowdown_noise = getattr(config, "SLOWDOWN_NOISE", 0.01)
    if slowdown_dropout is None:
        slowdown_dropout = getattr(config, "SLOWDOWN_DROPOUT", 0.2)

    inputs_eeg = layers.Input(shape=input_shape, name="input_eeg")
    inputs_label = layers.Input(shape=(1,), name="input_label")
    
    # --- ENCODER ---
    # Permute to (Samples, Channels)
    x = layers.Permute((2, 1))(inputs_eeg)
    x = layers.Dense(d_model)(x)
    x = PositionalEmbedding(input_shape[1], d_model)(x)

    # 1. Encoder Block
    enc_out = transformer_block(x, d_model, num_heads, ff_dim)

    # 2. Slowdown Bottleneck:
    # Upsample the sequence to compute details at a higher resolution (slowdown)
    slow = layers.UpSampling1D(size=slowdown_factor)(enc_out)
    slow = PositionalEmbedding(input_shape[1] * slowdown_factor, d_model)(slow)
    # Introduce noise and dropout to break deterministic smoothing and add stochastic diversity
    slow = layers.GaussianNoise(slowdown_noise)(slow)
    slow = layers.Dropout(slowdown_dropout)(slow)
    slow = transformer_block(slow, d_model, num_heads, ff_dim)
    # Downsample back to original sequence length via average pooling
    slow_out = layers.AveragePooling1D(pool_size=slowdown_factor)(slow)

    # 3. Decoder Block: Add the original encoder representation and the slowdown features
    dec_in = layers.Add()([enc_out, slow_out])
    
    # --- LABEL CONDITIONING ---
    # Embed binary label: (Batch, 1, d_model)
    label_emb = layers.Embedding(input_dim=2, output_dim=d_model)(inputs_label)
    label_emb = layers.Reshape((d_model,))(label_emb) # (Batch, d_model)
    label_emb = layers.RepeatVector(input_shape[1])(label_emb) # (Batch, 1024, d_model)
    
    dec_in_cond = layers.Add()([dec_in, label_emb])
    
    # Latent space perturbation
    dec_in_noise = layers.GaussianNoise(0.05)(dec_in_cond)
    dec_out = transformer_block(dec_in_noise, d_model, num_heads, ff_dim)

    # --- DECODER PROJECTION ---
    # Project back to original channels
    dec = layers.Dense(input_shape[0])(dec_out) # (1024, 6)
    outputs = layers.Permute((2, 1))(dec) # (6, 1024)

    full_model = models.Model(inputs=[inputs_eeg, inputs_label], outputs=outputs, name="Conditional_Slowdown_Transformer")
    feature_model = models.Model(inputs=[inputs_eeg, inputs_label], outputs=dec, name="Conditional_Slowdown_Generator")

    return full_model, feature_model

def build_eegnet(chans=6, samples=1024, filters_conv1=8, filters_conv2=16, dropout=0.5):
    """
    Standard EEGNet architecture tailored for EEG classification.
    Ref: Lawhern et al. (2018), 'EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces'
    """
    model = models.Sequential(name=f"EEGNet_{chans}ch")
    model.add(layers.Input(shape=(chans, samples, 1)))
    
    # 1. Temporal Filter: Conv2D across time points
    model.add(layers.Conv2D(filters_conv1, (1, 64), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    
    # 2. Spatial Filter: DepthwiseConv2D across channels (spatially couples signals)
    model.add(layers.DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D((1, 4)))
    model.add(layers.Dropout(dropout))
    
    # 3. Pointwise Filter: SeparableConv2D to combine temporal/spatial patterns
    model.add(layers.SeparableConv2D(filters_conv2, (1, 16), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D((1, 8)))
    
    # 4. Classification Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# =====================================================================
# 2. PYTORCH MODEL (CNN-LSTM)
# =====================================================================

try:
    import torch
    import torch.nn as nn

    
    class ASDClassifier(nn.Module):
        """
        PyTorch classifier combining 2D Convolutions and LSTMs to process 
        Temporal-Spatial Matrix Brain Functional Connectivity (TSM-BFC) inputs.
        """
        def __init__(self, num_features, time_steps, hidden_size=128, filters_conv1=32, filters_conv2=64, filters_conv3=128, dropout=0.5, kernel_size=(3, 3)):
            """
            Args:
                num_features: Number of selected connectivity features (e.g., channels).
                time_steps: Number of time frames in the TSM-BFC.
                hidden_size: Number of units in the LSTM hidden layer.
            """
            super(ASDClassifier, self).__init__()
            
            # Calculate padding to keep same height/width after Conv2d
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            
            # 1. 2D Convolutional Block: Extracts spatial-temporal patterns from the TSM-BFC grid
            self.cnn_block = nn.Sequential(
                # First Conv Layer: 1 input channel -> filters_conv1 feature maps
                nn.Conv2d(1, filters_conv1, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(filters_conv1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                
                # Second Conv Layer: filters_conv1 -> filters_conv2 feature maps
                nn.Conv2d(filters_conv1, filters_conv2, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(filters_conv2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                
                # Third Conv Layer: filters_conv2 -> filters_conv3 feature maps
                nn.Conv2d(filters_conv2, filters_conv3, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(filters_conv3),
                nn.ReLU(),
                # Collapses spatial feature dimension, leaving the reduced time steps dimension
                nn.AdaptiveAvgPool2d((1, None)) 
            )
            
            # Input size for the LSTM equals the final channels (filters_conv3) of the Conv block
            self.lstm_input_dim = filters_conv3 
            
            # 2. LSTM Block: Tracks temporal dependencies across the sequence of features
            self.lstm = nn.LSTM(
                input_size=self.lstm_input_dim, 
                hidden_size=hidden_size, 
                num_layers=1, 
                batch_first=True
            )
            
            # 3. Dense Classifier Block: Final binary prediction (ASD vs TD/Control)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout), # Regularization to prevent overfitting
                nn.Linear(64, 1),
                nn.Sigmoid()     # Sigmoid output for binary classification probability
            )

        def forward(self, x):
            # Input shape: (Batch, 1, num_features, time_steps)
            
            # Extract features through Conv block: result shape (Batch, filters_conv3, 1, time_steps_reduced)
            x = self.cnn_block(x) 
            
            # Reshape for LSTM: squeeze the spatial dimension (height=1 at index 2) and permute to (Batch, Sequence_length, Features)
            x = x.squeeze(2).permute(0, 2, 1) 
            
            # Pass sequence through LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Extract the hidden state from the final time step
            last_hidden = h_n[-1] 
            
            # Classify the sequence representation
            out = self.fc(last_hidden)
            return out
except ImportError:
    # Fallback if torch is not installed or when running import tests
    class ASDClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch 'torch' module is required to instantiate ASDClassifier.")
