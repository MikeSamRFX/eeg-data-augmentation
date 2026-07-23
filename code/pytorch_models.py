import torch
import torch.nn as nn
import config

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
