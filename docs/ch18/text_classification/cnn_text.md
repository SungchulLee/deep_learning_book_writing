# CNN for Text Classification

## Overview

TextCNN (Kim, 2014) applies 1D convolutions over word embeddings to capture local n-gram patterns.

## Architecture

$$c_i = \text{ReLU}(\mathbf{w} \cdot \mathbf{x}_{i:i+h-1} + b)$$

Max-over-time pooling: $\hat{c} = \max_i c_i$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, filter_sizes=(2, 3, 4), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embeds = self.embedding(x).transpose(1, 2)
        conv_outs = [F.relu(conv(embeds)).max(dim=2)[0] for conv in self.convs]
        pooled = torch.cat(conv_outs, dim=1)
        return self.fc(self.dropout(pooled))
```

Multiple filter sizes (2, 3, 4) capture bigram, trigram, and 4-gram patterns simultaneously.

## References

1. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *EMNLP*.
