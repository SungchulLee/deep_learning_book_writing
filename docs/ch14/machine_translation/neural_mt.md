# Neural Machine Translation

## Learning Objectives

- Understand the encoder-decoder architecture for MT
- Implement a basic seq2seq translation model
- Identify the information bottleneck problem

## Encoder-Decoder Architecture

Neural MT (Sutskever et al., 2014) uses a single neural network to directly model $P(\mathbf{y} \mid \mathbf{x})$:

**Encoder** reads the source sentence and produces a context representation:

$$\mathbf{h}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1})$$

The final hidden state $\mathbf{h}_T$ encodes the entire source sentence.

**Decoder** generates the target sentence token by token:

$$\mathbf{s}_t = \text{LSTM}([\mathbf{y}_{t-1}; \mathbf{c}], \mathbf{s}_{t-1})$$

$$P(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t)$$

where $\mathbf{c} = \mathbf{h}_T$ is the fixed context vector.

## Implementation

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        # Combine bidirectional states
        hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        cell = cell.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim * 2, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        embedded = self.dropout(self.embedding(trg))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell
```

## Teacher Forcing

During training, feed the ground-truth previous token rather than the model's own prediction:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* \mid y_{<t}^*, \mathbf{x})$$

At inference, use the model's own predictions (autoregressive generation). The mismatch between training and inference is called **exposure bias**.

## The Information Bottleneck

The fixed-length context vector $\mathbf{c} = \mathbf{h}_T$ must compress the entire source sentence into a single vector. This creates a bottleneck for long sentences — translation quality degrades as sentence length increases.

**Solution**: The attention mechanism (next section) allows the decoder to access all encoder hidden states, not just the final one.

## Practical Considerations

- **Reversing source sentence**: Sutskever et al. found that reversing the source sequence improves results by reducing the distance between corresponding words
- **Beam search decoding**: At inference, maintain top-$k$ hypotheses
- **Length normalization**: Divide log-probability by length to avoid favoring short translations

## References

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS*.
2. Cho, K., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder. *EMNLP*.
