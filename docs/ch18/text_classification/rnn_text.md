# RNN for Text Classification

## Overview

RNN/LSTM classifiers process text sequentially, using the final hidden state or attention-pooled states for classification.

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        fc_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_dim, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        output, (h_n, _) = self.lstm(embeds)
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        return self.fc(self.dropout(hidden))
```

*See accompanying code `imdb_classifier.py` for an IMDB sentiment classification example.*

## Pooling strategies: last hidden, max pool, mean pool, or self-attention pooling.

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
