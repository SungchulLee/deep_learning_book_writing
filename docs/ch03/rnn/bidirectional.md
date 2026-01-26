# Bidirectional RNNs

## Introduction

Standard RNNs process sequences in one direction, limiting their understanding to past context. **Bidirectional RNNs** (BiRNNs) process sequences in both forward and backward directions, allowing each position to incorporate information from the entire sequence. This proves essential for tasks where future context informs interpretation of current elements.

## Motivation

Consider the sentence: "The bank by the river was steep."

A forward-only RNN processing "bank" has only seen "The"—insufficient to distinguish between a financial institution and a riverbank. A bidirectional model sees both the preceding and following context, enabling correct interpretation.

Many NLP tasks benefit from full context:
- **Named Entity Recognition**: "Apple" (company vs fruit)
- **Part-of-Speech Tagging**: "record" (noun vs verb)
- **Sentiment Analysis**: Negation and modifiers affect meaning
- **Machine Translation**: Word alignment requires full source context

## Architecture

A BiRNN consists of two separate RNNs:
1. **Forward RNN**: Processes $x_1 \to x_2 \to \cdots \to x_T$
2. **Backward RNN**: Processes $x_T \to x_{T-1} \to \cdots \to x_1$

At each timestep, outputs are combined (typically concatenated):

```
Forward:   h₁→ → h₂→ → h₃→ → h₄→
             ↑     ↑     ↑     ↑
Input:      x₁    x₂    x₃    x₄
             ↓     ↓     ↓     ↓  
Backward: ←h₁  ←h₂  ←h₃  ←h₄

Output:   [h₁→;←h₁] [h₂→;←h₂] [h₃→;←h₃] [h₄→;←h₄]
```

## Mathematical Formulation

### Forward Pass
$$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})$$

### Backward Pass
$$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

### Combined Output
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \quad \text{(concatenation)}$$

or

$$h_t = \overrightarrow{h}_t + \overleftarrow{h}_t \quad \text{(summation)}$$

The concatenation approach doubles the hidden dimension but preserves all information.

## PyTorch Implementation from Scratch

```python
import torch
import torch.nn as nn

class BidirectionalRNN(nn.Module):
    """Bidirectional RNN implementation."""
    
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Select RNN type
        rnn_class = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        
        # Forward and backward RNNs
        self.forward_rnn = rnn_class(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.backward_rnn = rnn_class(
            input_size, hidden_size, num_layers, batch_first=True
        )
    
    def forward(self, x):
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
        
        Returns:
            outputs: Concatenated hidden states (batch, seq_len, hidden_size * 2)
            h_n: Final hidden states (num_layers * 2, batch, hidden_size)
        """
        # Forward pass
        forward_out, forward_h = self.forward_rnn(x)
        
        # Reverse sequence for backward pass
        x_reversed = torch.flip(x, dims=[1])
        backward_out, backward_h = self.backward_rnn(x_reversed)
        
        # Reverse backward outputs to align with forward
        backward_out = torch.flip(backward_out, dims=[1])
        
        # Concatenate forward and backward
        outputs = torch.cat([forward_out, backward_out], dim=-1)
        
        # Combine final hidden states
        if isinstance(forward_h, tuple):  # LSTM returns (h, c)
            h_n = (torch.cat([forward_h[0], backward_h[0]], dim=0),
                   torch.cat([forward_h[1], backward_h[1]], dim=0))
        else:  # GRU/RNN
            h_n = torch.cat([forward_h, backward_h], dim=0)
        
        return outputs, h_n
```

## Using PyTorch's Built-in Bidirectional

```python
# Bidirectional LSTM
bilstm = nn.LSTM(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # Enable bidirectionality
)

# Input
x = torch.randn(32, 50, 100)

# Forward pass
outputs, (h_n, c_n) = bilstm(x)

print(f"Outputs: {outputs.shape}")  # (32, 50, 256) - doubled hidden size
print(f"h_n: {h_n.shape}")          # (4, 32, 128) - 2 layers × 2 directions
print(f"c_n: {c_n.shape}")          # (4, 32, 128)
```

## Extracting Final States

For classification, you typically need the final hidden states from both directions:

```python
def get_final_bilstm_states(h_n, num_layers):
    """
    Extract and concatenate final states from bidirectional LSTM.
    
    Args:
        h_n: Hidden states (num_layers * 2, batch, hidden_size)
        num_layers: Number of layers
    
    Returns:
        final_hidden: (batch, hidden_size * 2)
    """
    # h_n layout: [layer0_forward, layer0_backward, layer1_forward, layer1_backward, ...]
    # Last layer's states are at indices -2 (forward) and -1 (backward)
    forward_final = h_n[-2]   # (batch, hidden)
    backward_final = h_n[-1]  # (batch, hidden)
    
    return torch.cat([forward_final, backward_final], dim=-1)

# Usage
outputs, (h_n, c_n) = bilstm(x)
final_states = get_final_bilstm_states(h_n, num_layers=2)  # (32, 256)
```

## BiLSTM for Text Classification

```python
class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM for sequence classification."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3):
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
        
        # *2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        
        lstm_out, (h_n, c_n) = self.bilstm(embedded)
        
        # Get final hidden states from both directions
        forward_final = h_n[-2]   # Last layer, forward direction
        backward_final = h_n[-1]  # Last layer, backward direction
        final_hidden = torch.cat([forward_final, backward_final], dim=-1)
        
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)
        
        return logits

# Usage
model = BiLSTMClassifier(
    vocab_size=10000,
    embed_dim=100,
    hidden_size=128,
    num_classes=2
)

x = torch.randint(0, 10000, (32, 100))
logits = model(x)  # (32, 2)
```

## BiLSTM for Sequence Labeling

For tasks like NER or POS tagging, predict at every position:

```python
class BiLSTMTagger(nn.Module):
    """BiLSTM for sequence labeling (NER, POS tagging)."""
    
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
        
        # Project from bidirectional hidden to tag space
        self.fc = nn.Linear(hidden_size * 2, num_tags)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        
        lstm_out, _ = self.bilstm(embedded)  # (batch, seq_len, hidden*2)
        
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # (batch, seq_len, num_tags)
        
        return logits

# Usage
tagger = BiLSTMTagger(
    vocab_size=10000,
    embed_dim=100,
    hidden_size=128,
    num_tags=9  # BIO tagging scheme
)

x = torch.randint(0, 10000, (32, 50))
tag_logits = tagger(x)  # (32, 50, 9)
```

## Attention over BiLSTM Outputs

Combine bidirectional outputs with attention:

```python
class BiLSTMWithAttention(nn.Module):
    """BiLSTM with self-attention pooling."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_size, num_layers=1,
            batch_first=True, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)  # (batch, seq_len, hidden*2)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided (padding)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted sum of hidden states
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        # context: (batch, hidden*2)
        
        logits = self.fc(self.dropout(context))
        return logits, attention_weights
```

## Multi-Layer Bidirectional Architecture

For deep bidirectional networks:

```python
class DeepBiLSTM(nn.Module):
    """Deep bidirectional LSTM with residual connections."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size * 2
            self.layers.append(nn.LSTM(
                input_dim, hidden_size, 1,
                batch_first=True, bidirectional=True
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        for i, (lstm, ln) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x if i > 0 and x.size(-1) == lstm.hidden_size * 2 else None
            
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
            
            if residual is not None:
                lstm_out = lstm_out + residual
            
            x = lstm_out
        
        return x
```

## Computational Considerations

| Aspect | Unidirectional | Bidirectional |
|--------|----------------|---------------|
| Parameters | $H$ | $2H$ (approximately) |
| Output dimension | $H$ | $2H$ |
| Computation | Forward only | Forward + Backward |
| Memory | $O(T \cdot H)$ | $O(T \cdot 2H)$ |
| Parallelization | Sequential | Two parallel passes |

**Important**: Bidirectional models cannot be used for:
- Real-time streaming (need future context)
- Autoregressive generation (prediction depends on future)

## When to Use Bidirectional

**Use BiRNN when**:
- Full sequence is available at inference time
- Task benefits from future context
- Classification or labeling over fixed sequences
- Encoding sequences for attention mechanisms

**Don't use BiRNN when**:
- Real-time/streaming processing required
- Autoregressive generation
- Memory constraints are severe

## Summary

Bidirectional RNNs provide richer representations by incorporating context from both directions:

1. **Forward RNN**: Captures past context
2. **Backward RNN**: Captures future context
3. **Concatenation**: Preserves both perspectives

Key applications:
- Text classification
- Named Entity Recognition
- Part-of-Speech tagging
- Sequence encoders for attention models

The architecture roughly doubles parameters and computation but often provides significant accuracy improvements for tasks with full sequence access.
