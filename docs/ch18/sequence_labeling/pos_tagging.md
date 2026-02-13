# POS Tagging

## Overview

Part-of-Speech (POS) tagging assigns grammatical categories (noun, verb, adjective, etc.) to each token. POS tags provide syntactic features useful for NER, parsing, and information extraction.

## Tag Sets

| Tag Set | Tags | Example |
|---------|------|---------|
| Universal Dependencies | 17 | NOUN, VERB, ADJ, ADV, ... |
| Penn Treebank | 45 | NN, VBZ, JJ, RB, DT, ... |

## Classical Approaches

### Hidden Markov Model (HMM)

$$P(\mathbf{y} | \mathbf{x}) \propto \prod_i P(x_i | y_i) \cdot P(y_i | y_{i-1})$$

### Maximum Entropy Markov Model (MEMM)

$$P(y_i | y_{i-1}, \mathbf{x}) = \frac{\exp(\mathbf{w} \cdot \mathbf{f}(y_i, y_{i-1}, \mathbf{x}))}{Z(y_{i-1}, \mathbf{x})}$$

## Neural POS Tagging

LSTM-based taggers achieve >97% accuracy on Penn Treebank:

```python
import torch
import torch.nn as nn

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        return self.classifier(lstm_out)
```

## Benchmark Results

| Model | PTB Accuracy |
|-------|-------------|
| HMM | 96.0% |
| CRF | 97.0% |
| BiLSTM | 97.3% |
| BERT-base | 97.8% |

## Summary

1. POS tagging is a foundational sequence labeling task with high baseline accuracy
2. Neural models achieve >97% accuracy, approaching human agreement (~97%)
3. POS tags remain useful as features for downstream NLP tasks
4. Universal Dependencies provides a cross-lingual POS tag standard
