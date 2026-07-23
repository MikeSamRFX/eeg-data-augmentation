import numpy as np
import os
import json

processed_dir = "/home/maycon/tcc/data-set/processed"
slices_dir = os.path.join(processed_dir, "slices")
files = [f for f in os.listdir(slices_dir) if f.endswith('_slices.npy')]
print(f"Total subjects: {len(files)}")

first_file = os.path.join(slices_dir, files[0])
data = np.load(first_file)
print(f"Shape of one subject data: {data.shape}")

splits_path = os.path.join(processed_dir, "splits.json")
if os.path.exists(splits_path):
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    print(f"Splits load successful: {len(splits)} folds")
    fold = splits[0]
    print(f"Train files count in Fold 1: {len(fold['train_files'])}")
    print(f"Test files count in Fold 1: {len(fold['test_files'])}")
