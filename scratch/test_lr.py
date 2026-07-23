import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))
import config
from folding import load_split_data
from classifier import standardize_features, PurePythonAdam
from models import ASDClassifier

def test_run(lr=5e-4, batch_size=32, epochs=30):
    splits_path = os.path.join(config.PROCESSED_DIR, "splits.json")
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    split = splits[0] # Fold 1
    
    X_train, y_train, X_test, y_test = load_split_data(config.PROCESSED_DIR, split)
    X_train, X_test = standardize_features(X_train, X_test)
    
    # Pre-processing shape
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = ASDClassifier(
        num_features=config.CHANNELS,
        time_steps=config.SAMPLES,
        hidden_size=128,
        filters_conv1=8,
        filters_conv2=64,
        filters_conv3=32,
        dropout=0.1,
        kernel_size=(5, 5)
    )
    model.to(device)
    
    optimizer = PurePythonAdam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    print(f"\n=== Testing LR={lr} | Batch Size={batch_size} ===")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        train_acc = correct / total
        train_loss = epoch_loss / total
        
        # Eval
        model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X_test, dtype=torch.float32).to(device)
            t_y = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            val_out = model(t_x)
            val_loss = criterion(val_out, t_y).item()
            val_preds = (val_out > 0.5).float()
            val_acc = (val_preds == t_y).sum().item() / t_y.size(0)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    test_run(lr=5e-4, batch_size=32)
    test_run(lr=1e-3, batch_size=32)
    test_run(lr=1e-5, batch_size=32)
