import os
import csv
from datetime import datetime
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import config
from models import build_eegnet

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from models import ASDClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def z_score_data(X):
    """
    Standardize the data along the channel and time dimensions for each individual slice.
    X: shape (Batch, Channels, Samples)
    """
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True)
    return (X - mean) / (std + 1e-8)

def standardize_features(X_train, X_test):
    """
    Standardize EEG signals channel-wise across the entire dataset.
    Calculates channel-wise mean and std from X_train, and applies them to both X_train and X_test.
    X_train: shape (N_train, Channels, Samples)
    X_test: shape (N_test, Channels, Samples)
    """
    # Mean and std computed channel-wise across all batch items and temporal samples
    mean = np.mean(X_train, axis=(0, 2), keepdims=True) # (1, Channels, 1)
    std = np.std(X_train, axis=(0, 2), keepdims=True)   # (1, Channels, 1)
    
    # Avoid division by zero
    std_adj = np.where(std == 0.0, 1.0, std)
    
    X_train_norm = (X_train - mean) / std_adj
    X_test_norm = (X_test - mean) / std_adj
    return X_train_norm, X_test_norm

def train_eegnet(X_train, y_train, chans=6, filters_conv1=8, filters_conv2=16, dropout=0.5, learning_rate=None, batch_size=None, epochs=None):
    """
    Builds, compiles, and trains an EEGNet baseline on the provided training set.
    """
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if batch_size is None:
        batch_size = config.CNN_BATCH
    if epochs is None:
        epochs = config.CNN_EPOCHS

    # 1. Expand dimensions for CNN channels (Batch, Chans, Samples, 1)
    X_train_norm = np.expand_dims(X_train, -1)
    
    # 3. Instantiate model
    model = build_eegnet(chans=chans, samples=config.SAMPLES, filters_conv1=filters_conv1, filters_conv2=filters_conv2, dropout=dropout)
    
    # 4. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
        jit_compile=False,
        run_eagerly=False
    )
    
    # 5. Fit model
    print("Starting model fit...")
    model.fit(
        X_train_norm, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    print("Model fit completed successfully!")
    
    return model

def evaluate_eegnet(model, X_test, y_test):
    """
    Evaluates the model on the test dataset and calculates metrics.
    Returns: acc, precision, recall, f1, predictions
    """
    # 1. Expand dimensions for CNN channels (Batch, Chans, Samples, 1)
    X_test_norm = np.expand_dims(X_test, -1)
    
    # 2. Predict probabilities and get binary classes
    probs = model.predict(X_test_norm, verbose=0)
    preds = (probs > 0.5).astype(int)
    
    # 3. Calculate metrics (average='binary' since label is ASD/Control)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='binary', zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }, preds

def log_results_to_csv(model_name, data_source, fold_idx, metrics, file_path=None):
    """
    Appends metrics to the central comparison results CSV file.
    """
    if file_path is None:
        file_path = config.RESULTS_FILE
        
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'Timestamp',
                'Model',
                'Data_Source',
                'Fold',
                'Accuracy',
                'Precision',
                'Recall_ASD',
                'F1_Score'
            ])
            
        # Write metric entry
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            data_source,
            fold_idx,
            round(metrics['accuracy'], 4),
            round(metrics['precision'], 4),
            round(metrics['recall'], 4),
            round(metrics['f1_score'], 4)
        ])
        
    print(f" FOLD {fold_idx} - Results for {model_name} ({data_source}) logged to {file_path}")

class PurePythonAdam:
    """
    A pure Python implementation of the Adam optimizer.
    Bypasses native PyTorch C++ optimizer calls to prevent library collision segmentation faults
    when both TensorFlow and PyTorch are imported in the same process.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.state = {}
        for p in self.params:
            self.state[p] = {
                'step': 0,
                'exp_avg': torch.zeros_like(p),
                'exp_avg_sq': torch.zeros_like(p)
            }
            
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
                
    def step(self):
        beta1, beta2 = self.betas
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state['step'] += 1
                
                # Update biased first moment estimate
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # Update biased second raw moment estimate
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                
                step_size = self.lr / bias_correction1
                denom = (state['exp_avg_sq'].sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
                
                p.addcdiv_(state['exp_avg'], denom, value=-step_size)

def train_pytorch_classifier(X_train, y_train, chans=6, filters_conv1=32, filters_conv2=64, filters_conv3=128, dropout=0.5, kernel_size=(3, 3), hidden_size=128, learning_rate=None, batch_size=None, epochs=None):
    """
    Builds, compiles, and trains an ASDClassifier (PyTorch) on the provided training set.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to run train_pytorch_classifier.")
        
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if batch_size is None:
        batch_size = config.CNN_BATCH
    if epochs is None:
        epochs = config.CNN_EPOCHS

    # 1. Expand dimensions for PyTorch CNN shape (Batch, 1, Channels, Samples)
    X_train_norm = np.expand_dims(X_train, 1)
    
    # 2. Build dataset and loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Instantiate model
    model = ASDClassifier(
        num_features=chans,
        time_steps=config.SAMPLES,
        hidden_size=hidden_size,
        filters_conv1=filters_conv1,
        filters_conv2=filters_conv2,
        filters_conv3=filters_conv3,
        dropout=dropout,
        kernel_size=kernel_size
    )
    model.to(device)
    
    # 4. Compile / Optimizer
    optimizer = PurePythonAdam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # 5. Fit model
    model.train()
    print("Starting PyTorch model fit...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
    print("PyTorch model fit completed successfully!")
    return model

def evaluate_pytorch_classifier(model, X_test, y_test):
    """
    Evaluates the PyTorch model on the test dataset and calculates metrics.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to run evaluate_pytorch_classifier.")
        
    # 1. Expand dimensions for PyTorch CNN shape (Batch, 1, Channels, Samples)
    X_test_norm = np.expand_dims(X_test, 1)
    
    # 2. Build dataset and loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.CNN_BATCH, shuffle=False)
    
    model.eval()
    probs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs.append(outputs.cpu().numpy())
            
    probs = np.concatenate(probs, axis=0)
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average='binary', zero_division=0
    )
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }, preds
