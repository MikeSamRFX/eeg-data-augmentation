import os
import json
import numpy as np
import config

def get_subject_folds(processed_dir, n_folds=4, seed=42):
    """
    Groups subjects into stratified folds.
    Ensures that all slices from a single subject stay in the same fold (subject-wise).
    Saves the splits to splits.json for reproducibility.
    """
    slices_dir = os.path.join(processed_dir, "slices")
    if not os.path.exists(slices_dir):
        raise FileNotFoundError(f"Slices directory not found: {slices_dir}")
        
    all_files = [f for f in os.listdir(slices_dir) if f.endswith('_slices.npy')]
    
    asd_subjects = sorted([f for f in all_files if "ASD" in f])
    control_subjects = sorted([f for f in all_files if "Control" in f])
    
    # Shuffle with seed for reproducibility
    rng = np.random.default_rng(seed)
    rng.shuffle(asd_subjects)
    rng.shuffle(control_subjects)
    
    # Divide stratified
    asd_folds = np.array_split(asd_subjects, n_folds)
    control_folds = np.array_split(control_subjects, n_folds)
    
    splits = []
    for i in range(n_folds):
        test_files = list(asd_folds[i]) + list(control_folds[i])
        
        train_files = []
        for j in range(n_folds):
            if j != i:
                train_files.extend(list(asd_folds[j]) + list(control_folds[j]))
                
        splits.append({
            "fold_idx": i + 1,
            "train_files": sorted(train_files),
            "test_files": sorted(test_files)
        })
        
    # Save to file
    splits_path = os.path.join(processed_dir, "splits.json")
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Stratified subject-wise folds created and saved to {splits_path}")
    return splits

def load_split_data(processed_dir, split_info):
    """
    Given a split dictionary, loads the train and test slices and labels.
    X shape: (N_slices, Channels, Samples)
    y shape: (N_slices,)
    """
    slices_dir = os.path.join(processed_dir, "slices")
    
    def load_files(file_list):
        X, y = [], []
        for f_name in file_list:
            file_path = os.path.join(slices_dir, f_name)
            data = np.load(file_path) # Shape: (Slices, Channels, Samples)
            X.append(data)
            label = 1 if "ASD" in f_name else 0
            y.extend([label] * data.shape[0])
        return np.concatenate(X, axis=0), np.array(y)
        
    X_train, y_train = load_files(split_info["train_files"])
    X_test, y_test = load_files(split_info["test_files"])
    return X_train, y_train, X_test, y_test

def get_continuous_train_data(processed_dir, split_info):
    """
    Loads continuous (full length) ROI data for training subjects.
    Used for sliding window augmentation.
    Returns: list of tuples (roi_matrix, label, subject_name)
    """
    full_roi_dir = os.path.join(processed_dir, "full_roi")
    continuous_data = []
    
    for f_name in split_info["train_files"]:
        # Find continuous file matching the subject name
        subject_id = f_name.replace("_slices.npy", "")
        full_file_name = f"{subject_id}_full.npy"
        file_path = os.path.join(full_roi_dir, full_file_name)
        
        if os.path.exists(file_path):
            data = np.load(file_path) # Shape: (Channels, TimePoints)
            label = 1 if "ASD" in f_name else 0
            continuous_data.append((data, label, subject_id))
            
    return continuous_data
