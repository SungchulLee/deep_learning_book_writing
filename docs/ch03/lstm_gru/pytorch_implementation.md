# PyTorch LSTM and GRU Implementation Guide

## Quick Reference

```python
import torch
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, 
               batch_first=True, dropout=0.2, bidirectional=False)

# GRU  
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2,
             batch_first=True, dropout=0.2, bidirectional=False)

# Forward pass
x = torch.randn(32, 50, 10)  # (batch, seq_len, input_size)

# LSTM returns output and (hidden, cell)
output, (h_n, c_n) = lstm(x)

# GRU returns output and hidden only
output, h_n = gru(x)
```

## Common Patterns

### Text Classification

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))
```

### Time Series Forecasting

```python
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])  # Last timestep
```

## Essential Tips

1. **batch_first=True**: Input shape is (batch, seq, features)
2. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`
3. **Detach hidden states** between batches for TBPTT
4. **Use model.eval()** during inference to disable dropout

## Variable Length Sequences

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pack sequences (requires lengths in descending order or enforce_sorted=False)
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
output_packed, hidden = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)
```

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/nn.html#lstm
