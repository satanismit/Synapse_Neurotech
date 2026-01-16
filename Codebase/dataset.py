import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, lfilter, iirnotch
from config import Config

class SynapseDataset(Dataset):
    def __init__(self, root_dir, subjects=None, mode='train'):
        """
        Args:
            root_dir (str): Path to the dataset root (containing SessionX folders).
            subjects (list): List of subject IDs to include.
            mode (str): 'train' or 'test' (affects transform behavior if any).
        """
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        self.mode = mode
        
        # Collect all files
        for session in Config.SESSIONS:
            session_path = os.path.join(root_dir, session)
            if not os.path.exists(session_path):
                continue
                
            subject_dirs = os.listdir(session_path)
            for subj_dir in subject_dirs:
                # Expected format: sessionX_subject_Y
                try:
                    parts = subj_dir.split('_')
                    subj_id = int(parts[-1])
                except:
                    continue
                
                if subjects is not None and subj_id not in subjects:
                    continue
                    
                subj_path = os.path.join(session_path, subj_dir)
                csv_files = glob.glob(os.path.join(subj_path, "*.csv"))
                
                for f in csv_files:
                    # Filename format: gestureXX_trialYY.csv
                    basename = os.path.basename(f)
                    try:
                        gesture_str = basename.split('_')[0] # gesture00
                        label = int(gesture_str.replace('gesture', ''))
                        if 0 <= label < Config.NUM_CLASSES:
                            self.files.append(f)
                            self.labels.append(label)
                    except:
                        continue

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]
        
        # Load Data
        try:
            df = pd.read_csv(filepath)
            # Ensure we have 8 columns
            if df.shape[1] != Config.NUM_CHANNELS:
                # Handle cases where there might be extra columns or headers issues
                # Assuming first 8 columns are the channels
                data = df.iloc[:, :Config.NUM_CHANNELS].values
            else:
                data = df.values
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return a dummy zero tensor in case of read error to avoid crashing
            return torch.zeros(Config.NUM_CHANNELS, Config.WINDOW_SIZE, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Truncate or Pad to fixed length
        if data.shape[0] > Config.WINDOW_SIZE:
            data = data[:Config.WINDOW_SIZE, :]
        elif data.shape[0] < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - data.shape[0]
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')

        # Transpose to (Channels, Time) -> (8, 2560)
        data = data.T 

        # Preprocessing
        data = self.apply_filters(data)
        data = self.normalize(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def apply_filters(self, data):
        # Data is (Channels, Time)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            channel_data = data[i, :]
            
            # Notch Filter (50Hz)
            b_notch, a_notch = iirnotch(Config.NOTCH_FREQ, 30.0, Config.SAMPLE_RATE)
            channel_data = lfilter(b_notch, a_notch, channel_data)
            
            # Bandpass Filter
            nyquist = 0.5 * Config.SAMPLE_RATE
            low = Config.LOWCUT / nyquist
            high = Config.HIGHCUT / nyquist
            b_band, a_band = butter(Config.FILTER_ORDER, [low, high], btype='band')
            channel_data = lfilter(b_band, a_band, channel_data)
            
            filtered_data[i, :] = channel_data
        return filtered_data

    def normalize(self, data):
        # Z-score normalization per channel
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)
