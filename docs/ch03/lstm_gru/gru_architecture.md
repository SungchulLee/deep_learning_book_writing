# Gated Recurrent Unit (GRU)

## Introduction

The Gated Recurrent Unit, introduced by Cho et al. (2014), offers a streamlined alternative to LSTM. GRU achieves comparable performance with fewer parameters by combining the forget and input gates into a single **update gate** and merging the cell and hidden states. This simplification often leads to faster training while maintaining the ability to capture long-range dependencies.

## Architecture Comparison

| Component | LSTM | GRU |
|-----------|------|-----|
| States | 2 (hidden $h$, cell $c$) | 1 (hidden $h$) |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | More | ~25% fewer |
| Computation | Slower | Faster |

## Mathematical Formulation

At each timestep $t$, given input $x_t$ and previous hidden state $h_{t-1}$:

### Update Gate
Controls how much of the previous state to keep:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

When $z_t \approx 1$, keep previous state; when $z_t \approx 0$, use new candidate.

### Reset Gate
Controls how much of the previous state to forget when computing the candidate:

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

### Candidate Hidden State
Proposes new state, with reset gate controlling previous state influence:

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

### Hidden State Update
Interpolates between previous state and candidate:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

## Understanding the Gates

### Update Gate Intuition

The update gate acts as both forget and input gate combined:
- High $z_t$: Accept new information (like LSTM input gate)
- Low $z_t$: Retain old information (like LSTM forget gate)

The complementary weighting $(1 - z_t)$ and $z_t$ ensures information is either kept or replaced, never both amplified or both suppressed.

### Reset Gate Intuition

The reset gate allows the model to forget the previous hidden state when computing the candidate:
- High $r_t$: Consider previous state when computing candidate
- Low $r_t$: Ignore previous state, essentially "resetting" memory

This is particularly useful for detecting boundaries between unrelated sequences.

## PyTorch Implementation from Scratch

```python
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """Single GRU cell."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Gates: [z, r, h_candidate] - combined for efficiency
        self.W_ih = nn.Linear(input_size, 3 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
    
    def forward(self, x, h):
        """
        Args:
            x: Input (batch, input_size)
            h: Previous hidden state (batch, hidden_size)
        
        Returns:
            h_new: New hidden state (batch, hidden_size)
        """
        # Compute gate inputs
        gates_x = self.W_ih(x)
        gates_h = self.W_hh(h)
        
        # Split into components
        z_x, r_x, n_x = gates_x.chunk(3, dim=-1)
        z_h, r_h, n_h = gates_h.chunk(3, dim=-1)
        
        # Update gate
        z = torch.sigmoid(z_x + z_h)
        
        # Reset gate
        r = torch.sigmoid(r_x + r_h)
        
        # Candidate hidden state
        n = torch.tanh(n_x + r * n_h)
        
        # New hidden state
        h_new = (1 - z) * h + z * n
        
        return h_new


class GRU(nn.Module):
    """Full GRU processing sequences."""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            GRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, h_0=None):
        """
        Args:
            x: Input sequence (batch, seq_len, input_size)
            h_0: Initial hidden states (num_layers, batch, hidden_size)
        
        Returns:
            outputs: All hidden states from top layer (batch, seq_len, hidden)
            h_n: Final hidden states (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states
        if h_0 is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [h_0[i] for i in range(self.num_layers)]
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(layer_input, h[layer_idx])
                layer_input = h[layer_idx]
                
                if self.dropout and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        outputs = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        
        return outputs, h_n
```

## Using PyTorch's Built-in GRU

```python
import torch.nn as nn

# Create GRU layer
gru = nn.GRU(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.3,
    bidirectional=False
)

# Input: (batch=32, seq_len=50, features=100)
x = torch.randn(32, 50, 100)

# Forward pass
outputs, h_n = gru(x)

print(f"Outputs: {outputs.shape}")  # (32, 50, 256)
print(f"h_n: {h_n.shape}")          # (2, 32, 256) - only hidden state, no cell
```

## GRU for Text Classification

```python
class GRUClassifier(nn.Module):
    """GRU for sequence classification."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        gru_out, h_n = self.gru(embedded)
        
        # Use final hidden state
        final_hidden = self.dropout(h_n[-1])
        logits = self.fc(final_hidden)
        
        return logits
```

## GRU for Time Series Forecasting

```python
class GRUForecaster(nn.Module):
    """GRU for multi-step time series prediction."""
    
    def __init__(self, input_size, hidden_size, output_size, 
                 num_layers=2, dropout=0.2, forecast_horizon=1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size * forecast_horizon)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, h_n = self.gru(x)
        
        # Forecast from final hidden state
        predictions = self.fc(h_n[-1])
        predictions = predictions.view(-1, self.forecast_horizon, -1)
        
        return predictions
```

## Gradient Flow Comparison

### LSTM Gradient Path
Gradients flow through cell state with additive updates:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{(can stay close to 1)}$$

### GRU Gradient Path
Gradients flow through the interpolation:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

Both architectures allow gradients to flow through additive paths, preventing vanishing gradients.

## When to Choose GRU vs LSTM

| Consideration | Choose GRU | Choose LSTM |
|---------------|------------|-------------|
| Training speed | ✓ Faster | Slower |
| Memory usage | ✓ Less | More |
| Long sequences | Good | ✓ May be better |
| Dataset size | ✓ Small to medium | Large |
| Default choice | ✓ Try first | If GRU underperforms |

**Practical guidance**: Start with GRU. If performance is insufficient, try LSTM. The difference is often marginal, and GRU's efficiency advantage makes it a strong default.

## Bidirectional GRU

Process sequences in both directions:

```python
class BiGRU(nn.Module):
    """Bidirectional GRU."""
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
    
    def forward(self, x):
        # Output: (batch, seq_len, hidden_size * 2)
        outputs, h_n = self.gru(x)
        
        # h_n: (num_layers * 2, batch, hidden_size)
        # Reshape to (num_layers, batch, hidden_size * 2)
        batch_size = x.size(0)
        h_n = h_n.view(self.gru.num_layers, 2, batch_size, -1)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)
        
        return outputs, h_n
```

## Minimal GRU Variant

Even simpler variants exist with only one gate:

```python
class MinimalGRUCell(nn.Module):
    """Minimal GRU with single gate (from Heck & Salem, 2017)."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        
        # Single forget gate
        f = torch.sigmoid(self.W_f(combined))
        
        # Candidate state
        h_tilde = torch.tanh(self.W_h(combined))
        
        # Update
        h_new = f * h + (1 - f) * h_tilde
        
        return h_new
```

## Initialization

```python
def init_gru_weights(gru):
    """Initialize GRU weights."""
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            param.data.fill_(0.)
            # Optionally set reset gate bias to encourage remembering
            n = param.size(0)
            param.data[n // 3:2 * n // 3].fill_(1.)  # Reset gate bias
```

## Summary

GRU simplifies LSTM while maintaining effectiveness:

1. **Two gates instead of three**: Update and reset gates
2. **Single state vector**: No separate cell state
3. **Complementary gating**: $(1-z)$ and $z$ ensure balanced updates

Key advantages:
- Fewer parameters (faster training, less overfitting on small data)
- Comparable performance to LSTM on most tasks
- Simpler to understand and implement

The architecture choice between GRU and LSTM is often empirical—try both and pick based on validation performance.
