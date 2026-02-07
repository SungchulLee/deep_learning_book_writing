# Stacked (Multi-Layer) LSTM and GRU

## Introduction

Stacking multiple LSTM or GRU layers creates **deep recurrent networks** capable of learning hierarchical representations of sequential data. Just as deep feedforward networks learn increasingly abstract features, stacked RNNs allow lower layers to capture local patterns while upper layers model longer-range, more abstract dependencies.

---

## Architecture

In a stacked (multi-layer) RNN, the output sequence of each layer becomes the input to the next layer:

```
Layer 3:  h₃¹ → h₃² → h₃³ → ... → h₃ᵀ  (output layer)
           ↑      ↑      ↑              ↑
Layer 2:  h₂¹ → h₂² → h₂³ → ... → h₂ᵀ
           ↑      ↑      ↑              ↑
Layer 1:  h₁¹ → h₁² → h₁³ → ... → h₁ᵀ
           ↑      ↑      ↑              ↑
Input:    x₁    x₂    x₃    ...    xᵀ
```

For layer $l$ at timestep $t$:

$$h_t^{(l)} = \text{RNNCell}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

where $h_t^{(0)} = x_t$ is the input at the bottom layer.

### Key Properties

- **Output** is produced only from the final (top) layer
- **Hidden states** are available for all layers via `h_n[layer_idx]`
- **Dropout** is applied between layers (not within a layer or after the final layer)
- **First layer** takes `input_size`; subsequent layers take `hidden_size` as input dimension

---

## PyTorch Implementation

### Using Built-in nn.LSTM / nn.GRU

```python
import torch
import torch.nn as nn

# Multi-layer LSTM
lstm = nn.LSTM(
    input_size=32, 
    hidden_size=64, 
    num_layers=3, 
    batch_first=True, 
    dropout=0.2  # Applied between layers 1→2 and 2→3
)

x = torch.randn(16, 50, 32)  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)

print(f"Output: {output.shape}")   # (16, 50, 64) - from final layer only
print(f"Hidden: {h_n.shape}")      # (3, 16, 64) - one per layer
print(f"Cell:   {c_n.shape}")      # (3, 16, 64) - one per layer (LSTM only)

# Multi-layer GRU
gru = nn.GRU(
    input_size=32, 
    hidden_size=64, 
    num_layers=3, 
    batch_first=True, 
    dropout=0.2
)

output, h_n = gru(x)

print(f"Output: {output.shape}")   # (16, 50, 64) - from final layer only
print(f"Hidden: {h_n.shape}")      # (3, 16, 64) - one per layer
```

### Accessing Per-Layer Hidden States

```python
# h_n[i] gives the final hidden state of layer i
layer_0_final = h_n[0]  # (batch, hidden_size) — bottom layer
layer_1_final = h_n[1]  # (batch, hidden_size) — middle layer
layer_2_final = h_n[2]  # (batch, hidden_size) — top layer

# For classification, typically use the top layer
logits = classifier(h_n[-1])

# For richer representations, concatenate all layers
all_layers = torch.cat([h_n[i] for i in range(3)], dim=-1)  # (batch, 3 * hidden)
```

---

## Depth Guidelines

### How Many Layers?

| Layers | Use Case | Notes |
|--------|----------|-------|
| 1 | Simple tasks, limited data, fast prototyping | Sufficient for many practical tasks |
| 2 | Most applications (recommended default) | Good depth-performance trade-off |
| 3–4 | Complex tasks, large datasets | Diminishing returns beyond 3 layers |
| 5+ | Rare; risk of optimization difficulty | Consider residual connections |

### Why Deeper Helps

Each layer in a stacked RNN processes increasingly abstract temporal features:

- **Layer 1**: Captures local patterns (e.g., word-level features, short-term price movements)
- **Layer 2**: Captures phrase-level or medium-term patterns
- **Layer 3+**: Captures high-level, long-range structure

This hierarchical processing is analogous to how deeper CNN layers capture increasingly complex visual features.

---

## Practical Patterns

### Stacked LSTM Classifier

```python
class StackedLSTMClassifier(nn.Module):
    """Multi-layer LSTM for sequence classification."""
    
    def __init__(self, input_size, hidden_size, num_layers, 
                 num_classes, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # output: (batch, seq, hidden) from top layer
        # h_n: (num_layers, batch, hidden)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state of top layer
        final_hidden = self.dropout(h_n[-1])
        logits = self.fc(final_hidden)
        
        return logits
```

### Stacked GRU with Layer-wise Attention

```python
class StackedGRUWithAttention(nn.Module):
    """
    Multi-layer GRU where all layers contribute to the output
    via learned attention over layers.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, 
                 output_size, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention over layers
        self.layer_attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden)
        
        # Compute attention weights over layers
        h_stacked = h_n.permute(1, 0, 2)  # (batch, num_layers, hidden)
        attn_scores = self.layer_attention(h_stacked).squeeze(-1)  # (batch, num_layers)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, num_layers)
        
        # Weighted combination
        context = torch.bmm(
            attn_weights.unsqueeze(1), h_stacked
        ).squeeze(1)  # (batch, hidden)
        
        return self.fc(context)
```

### Residual Stacked LSTM

For deep stacks (3+ layers), residual connections help gradient flow:

```python
class ResidualStackedLSTM(nn.Module):
    """Stacked LSTM with residual connections between layers."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # First layer may have different input size
        self.layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_size, hidden_size,
                    num_layers=1, batch_first=True)
            for i in range(num_layers)
        ])
        
        # Projection for residual when dimensions differ
        self.input_proj = (nn.Linear(input_size, hidden_size) 
                          if input_size != hidden_size else nn.Identity())
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers - 1)
        ])
    
    def forward(self, x, states=None):
        layer_input = x
        all_h = []
        all_c = []
        
        for i, lstm in enumerate(self.layers):
            output, (h_n, c_n) = lstm(layer_input)
            all_h.append(h_n.squeeze(0))
            all_c.append(c_n.squeeze(0))
            
            # Residual connection (skip connection)
            if i == 0:
                # Project input to hidden size for residual
                residual = self.input_proj(layer_input)
            else:
                residual = layer_input
            
            layer_input = output + residual  # Residual connection
            
            # Dropout between layers
            if i < self.num_layers - 1:
                layer_input = self.dropouts[i](layer_input)
        
        h_n = torch.stack(all_h, dim=0)
        c_n = torch.stack(all_c, dim=0)
        
        return layer_input, (h_n, c_n)
```

---

## Training Considerations

### Dropout Behavior

```python
# Dropout is applied between layers, NOT within a layer
# Must disable during evaluation
model.train()   # Dropout active
model.eval()    # Dropout disabled (use for inference/validation)
```

### Truncated BPTT for Long Sequences

For very long sequences, process in chunks and detach hidden states:

```python
def train_with_tbptt(model, data, chunk_size=100, device='cpu'):
    """Training with truncated backpropagation through time."""
    hidden = None
    total_loss = 0
    
    for i in range(0, data.size(1), chunk_size):
        chunk = data[:, i:i+chunk_size, :].to(device)
        
        # Detach hidden states to stop gradient flow between chunks
        if hidden is not None:
            if isinstance(hidden, tuple):  # LSTM
                hidden = tuple(h.detach() for h in hidden)
            else:  # GRU
                hidden = hidden.detach()
        
        output, hidden = model(chunk, hidden)
        loss = compute_loss(output, targets[:, i:i+chunk_size])
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
```

### Variable-Length Sequences with Packing

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pack sequences (avoids computation on padding)
packed = pack_padded_sequence(
    x, lengths, batch_first=True, enforce_sorted=False
)
output_packed, hidden = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)
```

---

## Summary

Stacked RNNs add depth to recurrent models by processing sequences through multiple layers:

| Aspect | Details |
|--------|---------|
| Purpose | Hierarchical feature learning |
| Default depth | 2 layers for most tasks |
| Dropout | Between layers only (requires `num_layers > 1`) |
| Output | From top layer only; per-layer states accessible via `h_n` |
| Deep stacks (3+) | Consider residual connections |

**Best practices:**

- Start with 2 layers; add more only if performance justifies it
- Apply dropout between layers (0.2–0.5 depending on dataset size)
- Use `model.eval()` during inference to disable dropout
- For very long sequences, use truncated BPTT with hidden state detachment
- Consider residual connections for networks deeper than 3 layers

---

## References

1. Pascanu, R., Gulcehre, C., Cho, K., & Bengio, Y. (2014). How to Construct Deep Recurrent Neural Networks. *ICLR*.

2. Graves, A. (2013). Generating Sequences With Recurrent Neural Networks. *arXiv:1308.0850*.

3. Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent Neural Network Regularization. *arXiv:1409.2329*.
