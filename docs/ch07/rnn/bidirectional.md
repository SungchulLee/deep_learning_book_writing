# Bidirectional RNN

## Motivation

Standard RNNs process sequences in one direction, so each hidden state $h_t$ encodes only information from positions $1, 2, \ldots, t$. This is a fundamental limitation for tasks where interpretation depends on both past and future context.

Consider the sentence "The bank by the river was steep." A forward-only RNN processing "bank" has seen only "The"—insufficient to distinguish between a financial institution and a riverbank. A bidirectional model processes the sequence in both directions, giving "bank" access to both preceding and following context.

### Tasks Requiring Full Context

| Task | Why Bidirectional Helps |
|------|------------------------|
| Named Entity Recognition | "Apple announced..." vs. "I ate an apple..." |
| Part-of-Speech Tagging | "record" as noun vs. verb depends on surrounding words |
| Sentiment Analysis | Negation, intensifiers, and sarcasm require full context |
| Machine Translation (Encoder) | Word alignment requires full source context |
| Question Answering | Answer span depends on entire passage |

## Architecture

A Bidirectional RNN consists of two independent RNNs processing the sequence in opposite directions:

```
Forward:     h₁→ ─► h₂→ ─► h₃→ ─► h₄→ ─► h₅→
              ↑      ↑      ↑      ↑      ↑
Input:       x₁     x₂     x₃     x₄     x₅
              ↓      ↓      ↓      ↓      ↓
Backward:    h₁← ◄─ h₂← ◄─ h₃← ◄─ h₄← ◄─ h₅←

Output:     [h₁→;h₁←] [h₂→;h₂←] [h₃→;h₃←] [h₄→;h₄←] [h₅→;h₅←]
              2H        2H        2H        2H        2H
```

### Mathematical Formulation

**Forward direction** (left-to-right):

$$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1}), \quad t = 1, 2, \ldots, T$$

**Backward direction** (right-to-left):

$$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1}), \quad t = T, T-1, \ldots, 1$$

At each position $t$, $\overrightarrow{h}_t$ encodes information from positions $1, \ldots, t$ and $\overleftarrow{h}_t$ encodes information from positions $T, \ldots, t$.

### Output Combination Strategies

**Concatenation** (most common): $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{2H}$

**Summation** (preserves dimension): $h_t = \overrightarrow{h}_t + \overleftarrow{h}_t \in \mathbb{R}^{H}$

**Learned projection**: $h_t = W[\overrightarrow{h}_t; \overleftarrow{h}_t] + b \in \mathbb{R}^{H}$

## PyTorch Implementation

### Using the Built-in Flag

```python
import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM with proper output handling."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: Optional sequence lengths for packing
        Returns:
            output: (batch_size, seq_len, hidden_size * 2)
            h_n: (num_layers * 2, batch_size, hidden_size)
            c_n: (num_layers * 2, batch_size, hidden_size)
        """
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, c_n) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (h_n, c_n) = self.lstm(x)
        
        return output, h_n, c_n
    
    def get_final_states(self, h_n):
        """Extract and concatenate final states from both directions."""
        h_fwd = h_n[-2]  # Last layer, forward
        h_bwd = h_n[-1]  # Last layer, backward
        return torch.cat([h_fwd, h_bwd], dim=-1)  # (batch, hidden*2)
```

### Understanding the Output Structure

The output layout requires careful attention:

```python
def analyze_bidirectional_output():
    """Detailed analysis of bidirectional LSTM outputs."""
    hidden_size = 64
    num_layers = 2
    seq_len, batch_size = 10, 4
    
    bilstm = nn.LSTM(32, hidden_size, num_layers,
                     batch_first=True, bidirectional=True)
    x = torch.randn(batch_size, seq_len, 32)
    output, (h_n, c_n) = bilstm(x)
    
    # output: (batch, seq_len, hidden_size * 2)
    #   output[:, :, :64]  → forward direction
    #   output[:, :, 64:]  → backward direction
    
    # h_n: (num_layers * 2, batch, hidden_size)
    #   Layout: [L0_fwd, L0_bwd, L1_fwd, L1_bwd, ...]
    
    # Key correspondence:
    # Forward final output == forward final hidden state
    assert torch.allclose(output[:, -1, :hidden_size], h_n[-2], atol=1e-6)
    
    # Backward final output (at t=0) == backward final hidden state
    assert torch.allclose(output[:, 0, hidden_size:], h_n[-1], atol=1e-6)

analyze_bidirectional_output()
```

The forward direction's "final" state corresponds to processing the last token, while the backward direction's "final" state corresponds to processing the first token. For classification, concatenating both gives a representation informed by the entire sequence.

### Manual Implementation

Building a bidirectional RNN from two unidirectional ones clarifies the mechanics:

```python
class ManualBiLSTM(nn.Module):
    """Manual implementation for understanding."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.forward_lstms = nn.ModuleList()
        self.backward_lstms = nn.ModuleList()
        
        for layer in range(num_layers):
            layer_input = input_size if layer == 0 else hidden_size * 2
            self.forward_lstms.append(
                nn.LSTM(layer_input, hidden_size, 1, batch_first=True)
            )
            self.backward_lstms.append(
                nn.LSTM(layer_input, hidden_size, 1, batch_first=True)
            )
    
    def forward(self, x):
        h_n_list, c_n_list = [], []
        current_input = x
        
        for layer in range(self.num_layers):
            out_fwd, (h_fwd, c_fwd) = self.forward_lstms[layer](current_input)
            
            x_rev = torch.flip(current_input, dims=[1])
            out_bwd_rev, (h_bwd, c_bwd) = self.backward_lstms[layer](x_rev)
            out_bwd = torch.flip(out_bwd_rev, dims=[1])
            
            current_input = torch.cat([out_fwd, out_bwd], dim=-1)
            h_n_list.extend([h_fwd.squeeze(0), h_bwd.squeeze(0)])
            c_n_list.extend([c_fwd.squeeze(0), c_bwd.squeeze(0)])
        
        h_n = torch.stack(h_n_list, dim=0)
        c_n = torch.stack(c_n_list, dim=0)
        return current_input, (h_n, c_n)
```

## Practical Applications

### Text Classification with Pooling Strategies

```python
class BiLSTMClassifier(nn.Module):
    """BiLSTM classifier with configurable pooling."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3, pooling='last'):
        super().__init__()
        self.pooling = pooling
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        pool_dim = hidden_size * 2
        if pooling == 'concat_last_max_mean':
            pool_dim = hidden_size * 2 * 3
        
        self.fc = nn.Sequential(
            nn.Linear(pool_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, (h_n, _) = self.bilstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            output, (h_n, _) = self.bilstm(embedded)
        
        if self.pooling == 'last':
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        elif self.pooling == 'max':
            pooled = output.max(dim=1)[0]
        elif self.pooling == 'mean':
            pooled = output.mean(dim=1)
        elif self.pooling == 'concat_last_max_mean':
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            pooled = torch.cat([last, output.max(dim=1)[0], output.mean(dim=1)], dim=-1)
        
        return self.fc(self.dropout(pooled))
```

### Sequence Labeling (NER / POS Tagging)

For token-level predictions, the full bidirectional output at each position feeds into a per-token classifier:

```python
class BiLSTMTagger(nn.Module):
    """BiLSTM for sequence labeling."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_tags,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_tags)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.bilstm(embedded)
        emissions = self.fc(self.dropout(lstm_out))  # (batch, seq_len, num_tags)
        return emissions
```

### Self-Attention Pooling

Learned attention weights over BiLSTM outputs provide a more expressive pooling mechanism:

```python
class BiLSTMWithAttention(nn.Module):
    """BiLSTM with self-attention for weighted pooling."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x, mask=None):
        output, _ = self.bilstm(x)  # (batch, seq, hidden*2)
        
        scores = self.attention(output).squeeze(-1)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim=-1)  # (batch, seq)
        context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        
        return self.fc(context), weights
```

### Seq2Seq Encoder with Bridge

When a bidirectional encoder feeds a unidirectional decoder, the hidden states must be bridged—the $2H$-dimensional bidirectional state must be projected to the $H$-dimensional decoder state:

```python
class BiLSTMEncoder(nn.Module):
    """Bidirectional encoder with bridge to unidirectional decoder."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional states to decoder dimension
        self.bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_c = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, src, src_lengths=None):
        embedded = self.dropout(self.embedding(src))
        
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs_packed, (h_n, c_n) = self.bilstm(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs_packed, batch_first=True
            )
        else:
            encoder_outputs, (h_n, c_n) = self.bilstm(embedded)
        
        # Reshape: (num_layers*2, batch, hidden) → (num_layers, 2, batch, hidden)
        batch_size = src.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate directions and project for decoder
        h_combined = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        c_combined = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)
        
        h_init = torch.tanh(self.bridge_h(h_combined))
        c_init = torch.tanh(self.bridge_c(c_combined))
        
        return encoder_outputs, (h_init, c_init)
```

## When to Use Bidirectional

### Suitable

Full sequence available at inference and future context improves interpretation: text classification, NER, sentiment analysis, machine translation encoders, offline speech recognition, question answering.

### Not Suitable

| Task | Reason |
|------|--------|
| Language modeling | Cannot see future tokens by definition |
| Autoregressive generation | Must predict based on past only |
| Real-time streaming | Future context not yet available |
| Online speech recognition | Low-latency requirement precludes waiting for full input |

### Cost Analysis

Bidirectional processing roughly doubles both parameters and computation compared to unidirectional:

| Metric | Unidirectional | Bidirectional |
|--------|---------------|---------------|
| Parameters | $P$ | $2P$ |
| Output dimension | $H$ | $2H$ |
| Forward time | $t$ | $\sim 2t$ |
| Memory | $M$ | $\sim 2M$ |

For memory-constrained settings, gradient checkpointing can reduce the memory overhead:

```python
class MemoryEfficientBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True
        )
    
    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.bilstm, x, use_reentrant=False
            )
        return self.bilstm(x)
```

## Summary

Bidirectional RNNs provide richer sequence representations by processing in both directions, giving each position access to the full sequence context. The output dimension doubles to $2H$ (with concatenation), and parameters and computation roughly double as well. Use bidirectional processing whenever the full sequence is available at inference and the task benefits from future context—classification, labeling, and encoding tasks are the primary beneficiaries. For generation and streaming tasks, unidirectional processing remains necessary.

**Implementation checklist**: set `bidirectional=True`, handle the doubled output dimension in subsequent layers, extract final states correctly (`h_n[-2]` for forward, `h_n[-1]` for backward), use packed sequences for variable-length inputs, and bridge bidirectional encoder states when feeding a unidirectional decoder.
