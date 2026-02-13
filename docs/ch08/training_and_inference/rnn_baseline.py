"""
RNN Baseline for Comparison
"""
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])
