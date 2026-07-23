import os
import numpy as np
import mne
from pymatreader import read_mat
import config

ROI_GROUPS = {
    'Frontal_Pole': ['Fp1', 'Fp2', 'Fpz', 'C16', 'C17', 'C32', 'AFz'],
    'Frontal': ['Fz', 'F3', 'F4', 'F7', 'F8', 'F1', 'F2', 'C1', 'C3', 'C4', 'D2', 'D4'],
    'Central': ['Cz', 'C3', 'C4', 'FCz', 'CPz', 'A1', 'A3', 'B1', 'B2', 'C19'],
    'Parietal': ['Pz', 'P3', 'P4', 'P1', 'P2', 'A19', 'A21', 'A23'],
    'Occipital': ['Oz', 'O1', 'O2', 'Iz', 'POz', 'A15', 'A17', 'A30'],
    'Temporal': ['T7', 'T8', 'TP7', 'TP8', 'D12', 'B12', 'D26', 'B26']
}

def preprocess_subject(file_path, label_name, output_dir):
    """
    Processes a single EEGLAB .set file.
    Averages electrodes within predefined ROIs, saves the full ROI-averaged
    matrix, and slices it into 2.0-second non-overlapping segments.
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    # Apply a bandpass filter of 1.0 - 40.0 Hz to remove DC offset/drifts and high frequency noise
    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    data = raw.get_data()
    ch_names = raw.ch_names
    
    # Initialize ROI matrix (Channels, TimePoints)
    roi_matrix = np.zeros((len(ROI_GROUPS), data.shape[1]))
    
    for i, (region, members) in enumerate(ROI_GROUPS.items()):
        indices = [ch_names.index(m) for m in members if m in ch_names]
        if indices:
            roi_matrix[i, :] = np.mean(data[indices, :], axis=0)
            
    # Save the continuous ROI averaged matrix
    base_name = os.path.basename(file_path).replace('.set', '')
    full_roi_dir = os.path.join(output_dir, "full_roi")
    os.makedirs(full_roi_dir, exist_ok=True)
    full_save_path = os.path.join(full_roi_dir, f"{label_name}_{base_name}_full.npy")
    np.save(full_save_path, roi_matrix)
    
    # Slicing the continuous signal into epochs of fixed length
    info = mne.create_info(ch_names=list(ROI_GROUPS.keys()), sfreq=raw.info['sfreq'], ch_types='eeg')
    roi_raw = mne.io.RawArray(roi_matrix, info, verbose=False)
    epochs = mne.make_fixed_length_epochs(roi_raw, duration=config.SLICE_DURATION, preload=True, verbose=False)
    
    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    
    if len(epochs) >= config.SLICES_PER_FILE:
        final_slices = epochs.get_data()[:config.SLICES_PER_FILE] # Shape: (Slices, Channels, Samples)
        slice_save_path = os.path.join(slices_dir, f"{label_name}_{base_name}_slices.npy")
        np.save(slice_save_path, final_slices)
        return True
    else:
        print(f"Skipping slices for {base_name}: not enough data ({len(epochs)} < {config.SLICES_PER_FILE} epochs).")
        return False

def run_preprocessing():
    """
    Main entry point for preprocessing raw datasets.
    """
    print(f"Starting preprocessing from {config.RAW_DIR}...")
    if not os.path.exists(config.RAW_DIR):
        raise FileNotFoundError(f"Raw data directory not found: {config.RAW_DIR}")
        
    all_files = [f for f in os.listdir(config.RAW_DIR) if f.endswith('.set')]
    if not all_files:
        print("No raw EEGLAB (.set) files found.")
        return
        
    processed_count = 0
    for idx, f_name in enumerate(all_files):
        file_path = os.path.join(config.RAW_DIR, f_name)
        data_dict = read_mat(file_path)
        
        # Determine classification label from metadata setname
        setname = str(data_dict['EEG']['setname'])
        label = "ASD" if "ASD" in setname else "Control"
        
        success = preprocess_subject(file_path, label, config.PROCESSED_DIR)
        if success:
            processed_count += 1
            
    print(f"Preprocessing completed! Successfully processed {processed_count}/{len(all_files)} files.")

if __name__ == '__main__':
    run_preprocessing()
