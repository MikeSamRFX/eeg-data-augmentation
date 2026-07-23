import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../code")))
from pytorch_models import ASDClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate dummy data matching mixup_transformer size (4200, 1, 64, 1024)
X_dummy = torch.randn(4200, 1, 64, 1024)
y_dummy = torch.randn(4200, 1).sigmoid()

dataset = TensorDataset(X_dummy, y_dummy)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ASDClassifier(
    num_features=64,
    time_steps=1024,
    hidden_size=128,
    filters_conv1=32,
    filters_conv2=64,
    filters_conv3=128,
    dropout=0.2
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("Starting training speed benchmark...")
start_time = time.time()

for epoch in range(2):
    epoch_start = time.time()
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds.")

print(f"Benchmark completed in {time.time() - start_time:.2f} seconds.")
