import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from pytorch_models import ASDClassifier

def main():
    if len(sys.argv) < 7:
        print("Usage: train_classifier.py <X_train_path> <y_train_path> <X_test_path> <y_test_path> <chans> <out_results_json_path>")
        sys.exit(1)
        
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    X_test_path = sys.argv[3]
    y_test_path = sys.argv[4]
    chans = int(sys.argv[5])
    out_results_json_path = sys.argv[6]
    
    # Load inputs
    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    # Run PyTorch classifier training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Subprocess: Training ASDClassifier on {device}...")
    
    # 1. Expand dimensions for PyTorch CNN shape (Batch, 1, Channels, Samples)
    X_train_norm = np.expand_dims(X_train, 1)
    X_test_norm = np.expand_dims(X_test, 1)
    
    # 2. Build datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.CNN_BATCH, shuffle=True)
    
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )
    test_loader = DataLoader(test_dataset, batch_size=config.CNN_BATCH, shuffle=False)
    
    # 3. Instantiate model
    model = ASDClassifier(
        num_features=chans,
        time_steps=config.SAMPLES,
        hidden_size=128,
        filters_conv1=config.FILTERS_CONV1,
        filters_conv2=config.FILTERS_CONV2,
        filters_conv3=config.FILTERS_CONV3,
        dropout=config.DROPOUT,
        kernel_size=config.KERNEL_SIZE
    )
    model.to(device)
    
    # 4. Optimizer: Native torch.optim.Adam since we run in a clean isolated subprocess
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # 5. Fit model
    model.train()
    for epoch in range(config.CNN_EPOCHS):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # 6. Evaluate model
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = (outputs >= 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1)
    all_targets = np.concatenate(all_targets, axis=0).squeeze(-1)
    
    # Calculate metrics manually to avoid dependencies inside subprocess
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    
    accuracy = (tp + tn) / len(all_targets) if len(all_targets) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    }
    
    # Save results
    with open(out_results_json_path, 'w') as f:
        json.dump(metrics, f)
        
    print("Subprocess: Classifier training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
