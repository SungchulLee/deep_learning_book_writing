# Deep RNN

## Motivation

A single-layer RNN applies one nonlinear transformation per timestep. Stacking multiple RNN layers increases the model's capacity to learn hierarchical representations: lower layers capture local patterns (e.g., character n-grams, short-term price movements) while upper layers compose these into higher-level abstractions (e.g., semantic meaning, long-term trends).

## Multi-Layer Architecture

In a stacked RNN with $L$ layers, the output of layer $\ell$ becomes the input to layer $\ell + 1$:

$$h_t^{(\ell)} = \text{RNN}^{(\ell)}(h_t^{(\ell-1)}, h_{t-1}^{(\ell)})$$

where $h_t^{(0)} = x_t$ (the input) and $h_t^{(\ell)}$ is the hidden state of layer $\ell$ at timestep $t$.

```
Layer 3:  h₁³ → h₂³ → h₃³ → h₄³    (abstract patterns)
           ↑     ↑     ↑     ↑
Layer 2:  h₁² → h₂² → h₃² → h₄²    (compositional features)
           ↑     ↑     ↑     ↑
Layer 1:  h₁¹ → h₂¹ → h₃¹ → h₄¹    (local patterns)
           ↑     ↑     ↑     ↑
Input:    x₁    x₂    x₃    x₄
```

Each layer maintains its own hidden state and weight matrices. Information flows both vertically (between layers at the same timestep) and horizontally (within a layer across timesteps).

## PyTorch Multi-Layer RNN

PyTorch's `num_layers` parameter handles stacking automatically:

```python
import torch
import torch.nn as nn

# Three-layer LSTM
deep_lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=3,
    batch_first=True,
    dropout=0.3  # Applied between layers (not within recurrence)
)

x = torch.randn(32, 50, 100)  # (batch, seq_len, input_size)
outputs, (h_n, c_n) = deep_lstm(x)

# outputs: (32, 50, 256) - top layer's hidden states at all timesteps
# h_n: (3, 32, 256) - final hidden state from each layer
# c_n: (3, 32, 256) - final cell state from each layer (LSTM only)
```

The `outputs` tensor contains only the top layer's hidden states. To access intermediate layers, you must either process layer by layer manually or use hooks.

### Inter-Layer Dropout

The `dropout` parameter applies dropout **between** layers but not within the recurrence of any single layer. This is important: applying dropout to the recurrent connection would break the hidden state's ability to carry information across timesteps.

```python
# dropout=0.3 applies between layers 1→2 and 2→3
# No dropout within each layer's temporal recurrence
deep_lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=3,
                    batch_first=True, dropout=0.3)
```

For a single-layer network, `dropout` has no effect since there are no inter-layer connections.

## Residual Connections

As depth increases, gradients must flow through more layers to reach the bottom, exacerbating vanishing gradients. **Residual (skip) connections** provide direct gradient paths:

$$h_t^{(\ell)} = \text{RNN}^{(\ell)}(h_t^{(\ell-1)}, h_{t-1}^{(\ell)}) + h_t^{(\ell-1)}$$

The additive shortcut ensures gradients can bypass entire layers, enabling training of deeper networks.

```python
class DeepResidualLSTM(nn.Module):
    """Deep LSTM with residual connections and layer normalization."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # Project input to hidden dimension for residual compatibility
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        x = self.input_proj(x)
        
        for lstm, ln in zip(self.layers, self.layer_norms):
            residual = x
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
            x = lstm_out + residual  # Skip connection
        
        return x
```

The input projection is necessary when `input_size ≠ hidden_size`, since residual addition requires matching dimensions.

## Highway Connections

Highway networks generalize residual connections with a learned gate that controls how much of the layer output versus the input to pass through:

$$y = g \odot \text{RNN}(x) + (1 - g) \odot x$$

where $g = \sigma(W_g x + b_g)$ is a gate vector in $[0, 1]^H$.

```python
class HighwayLSTM(nn.Module):
    """LSTM with highway connections for controlled information flow."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        self.layers = nn.ModuleList()
        self.highway_gates = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
            )
            self.highway_gates.append(nn.Linear(hidden_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for lstm, gate_layer in zip(self.layers, self.highway_gates):
            lstm_out, _ = lstm(x)
            lstm_out = self.dropout(lstm_out)
            
            gate = torch.sigmoid(gate_layer(x))
            x = gate * lstm_out + (1 - gate) * x
        
        return x
```

When $g \to 1$, the highway behaves like a standard deep network; when $g \to 0$, it acts as an identity mapping. The network learns the appropriate balance per dimension and position.

## Deep Bidirectional RNN

Combining depth with bidirectionality requires care with dimension matching:

```python
class DeepBiLSTM(nn.Module):
    """Deep bidirectional LSTM with residual connections."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Project to bidirectional output dimension for residuals
        self.input_proj = nn.Linear(input_size, hidden_size * 2)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                nn.LSTM(hidden_size * 2, hidden_size, 1,
                        batch_first=True, bidirectional=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size * 2))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for lstm, ln in zip(self.layers, self.layer_norms):
            residual = x
            lstm_out, _ = lstm(x)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
            x = lstm_out + residual
        
        return x
```

Each bidirectional layer produces output of dimension $2H$ (forward $H$ concatenated with backward $H$), so all residual connections and subsequent layers must work with this doubled dimension.

## Design Guidelines

### How Many Layers?

| Depth | Use Case | Notes |
|-------|----------|-------|
| 1 | Simple tasks, small datasets | Usually sufficient for basic sequence classification |
| 2 | Most practical applications | Good balance of capacity and trainability |
| 3–4 | Complex tasks with large data | Requires residual connections and careful regularization |
| 5+ | Rarely beneficial for RNNs | Diminishing returns; consider Transformers instead |

Empirically, RNN performance plateaus or degrades beyond 2–3 layers unless residual connections are used. This contrasts with Transformers, which routinely use 12–96 layers.

### Dimension Choices

A common pattern reduces hidden size as depth increases, or keeps it constant:

```python
# Constant hidden size (simplest, most common)
nn.LSTM(100, 256, num_layers=3)

# Decreasing hidden size (for compression)
# Must be done manually, layer by layer
layers = [
    nn.LSTM(100, 512, 1, batch_first=True),
    nn.LSTM(512, 256, 1, batch_first=True),
    nn.LSTM(256, 128, 1, batch_first=True),
]
```

### Regularization for Deep RNNs

Deep RNNs require more aggressive regularization:

```python
class RegularizedDeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout=0.3, weight_decay=1e-5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.layer_norm(output)
        return output, (h_n, c_n)

# Use weight decay in optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-5
)
```

### Gradient Clipping for Deep Networks

Deeper networks amplify gradient instability. Use more conservative clipping thresholds:

```python
# Single-layer RNN: max_norm=5.0 is often sufficient
# Deep RNN (3+ layers): start with max_norm=0.5–1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Parameter Count Analysis

For a single-direction LSTM with $L$ layers, hidden size $H$, and input size $d$:

| Component | Parameters |
|-----------|-----------|
| Layer 1 (input) | $4H(d + H) + 8H$ |
| Layer $\ell > 1$ | $4H(H + H) + 8H = 8H^2 + 8H$ |
| Total ($L$ layers) | $4H(d + H) + 8H + (L-1)(8H^2 + 8H)$ |

For bidirectional, multiply by 2. The quadratic scaling in $H$ means that increasing hidden size is more expensive than adding layers (for $H > d$).

```python
def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,} ({param.shape})")
    
    return total
```

## Summary

Deep (multi-layer) RNNs increase representational capacity by stacking recurrent layers, enabling hierarchical feature extraction from sequences. Lower layers capture local patterns while upper layers compose higher-level abstractions. Practical deep RNNs require residual or highway connections to maintain gradient flow, inter-layer dropout for regularization, and conservative gradient clipping for stability. Two to three layers suffice for most tasks; beyond that, diminishing returns set in and Transformer architectures become more effective.
