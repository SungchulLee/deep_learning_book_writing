#!/usr/bin/env python3
'''
Seq2Seq - Sequence to Sequence Learning with Neural Networks
Paper: "Sequence to Sequence Learning with Neural Networks" (2014)
Key: Encoder-Decoder architecture for sequence transduction
'''
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers)
    
    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, _, _ = self.decoder(trg, hidden, cell)
        return outputs

if __name__ == "__main__":
    model = Seq2Seq(input_size=1000, output_size=1000)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
