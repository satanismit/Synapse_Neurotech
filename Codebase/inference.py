import torch
import pandas as pd
import numpy as np
import os
import argparse
from scipy.signal import butter, lfilter, iirnotch
from config import Config
from model import EMGConv1D

class InferencePipeline:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EMGConv1D().to(self.device)
        
        if model_path is None:
            model_path = Config.MODEL_PATH
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}. Using random weights (for testing only).")
            
        self.model.eval()

    def preprocess(self, data):
        # Data shape: (Time, Channels) -> (2560, 8)
        
        # Truncate or Pad
        if data.shape[0] > Config.WINDOW_SIZE:
            data = data[:Config.WINDOW_SIZE, :]
        elif data.shape[0] < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - data.shape[0]
            data = np.pad(data, ((0, pad_len), (0, 0)), mode='constant')
            
        # Transpose to (Channels, Time) -> (8, 2560)
        data = data.T
        
        # Filters
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            channel_data = data[i, :]
            # Notch
            b_notch, a_notch = iirnotch(Config.NOTCH_FREQ, 30.0, Config.SAMPLE_RATE)
            channel_data = lfilter(b_notch, a_notch, channel_data)
            # Bandpass
            nyquist = 0.5 * Config.SAMPLE_RATE
            low = Config.LOWCUT / nyquist
            high = Config.HIGHCUT / nyquist
            b_band, a_band = butter(Config.FILTER_ORDER, [low, high], btype='band')
            channel_data = lfilter(b_band, a_band, channel_data)
            filtered_data[i, :] = channel_data
            
        # Normalize
        mean = np.mean(filtered_data, axis=1, keepdims=True)
        std = np.std(filtered_data, axis=1, keepdims=True)
        normalized_data = (filtered_data - mean) / (std + 1e-8)
        
        return torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0) # Add batch dim

    def predict(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            if df.shape[1] != Config.NUM_CHANNELS:
                data = df.iloc[:, :Config.NUM_CHANNELS].values
            else:
                data = df.values
                
            input_tensor = self.preprocess(data).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                
            return predicted.item()
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sEMG Gesture Inference")
    parser.add_argument("input", type=str, help="Path to input CSV file or directory")
    parser.add_argument("--model", type=str, default=Config.MODEL_PATH, help="Path to trained model")
    args = parser.parse_args()
    
    pipeline = InferencePipeline(args.model)
    
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.csv')]
        print(f"Found {len(files)} CSV files in directory.")
        for f in files:
            pred = pipeline.predict(f)
            print(f"{os.path.basename(f)} -> Gesture {pred}")
    else:
        pred = pipeline.predict(args.input)
        print(f"Prediction for {args.input}: Gesture {pred}")
