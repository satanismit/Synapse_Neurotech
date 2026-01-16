import torch
import torch.nn as nn
from config import Config

class EMGConv1D(nn.Module):
    def __init__(self):
        super(EMGConv1D, self).__init__()
        
        # Input: (Batch, 8, 2560)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=Config.NUM_CHANNELS, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            # Block 4
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, Config.NUM_CLASSES)
        )

    def forward(self, x):
        # x: (Batch, 8, 2560)
        x = self.features(x)
        # x: (Batch, 256, 1)
        x = x.view(x.size(0), -1)
        # x: (Batch, 256)
        x = self.classifier(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Sanity check
    model = EMGConv1D()
    print(f"Model Parameters: {model.get_num_params()}")
    dummy_input = torch.randn(2, Config.NUM_CHANNELS, Config.WINDOW_SIZE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
