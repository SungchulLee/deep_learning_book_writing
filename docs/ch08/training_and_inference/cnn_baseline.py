"""
CNN Baseline for Comparison
"""
import torch.nn as nn

class CNNBaseline(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)
