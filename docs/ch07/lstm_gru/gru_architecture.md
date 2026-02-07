# GRU Architecture

## Introduction

The Gated Recurrent Unit (GRU), introduced by Cho et al. in 2014, is a streamlined variant of LSTM that achieves comparable performance with fewer parameters. GRU simplifies the LSTM architecture by merging the cell state and hidden state into a single state vector and reducing the number of gates from three to two.

This architectural simplification offers practical advantages: faster training, lower memory footprint, and reduced risk of overfitting on smaller datasets—all while maintaining the ability to capture long-range dependencies that vanilla RNNs cannot learn.

**Core idea:** Where LSTM uses separate pathways for "what to remember" (cell state) and "what to output" (hidden state), GRU combines these into a single hidden state with two gates controlling information flow.

---

## Mathematical Formulation

For time step $t$, given input $x_t \in \mathbb{R}^d$ and previous hidden state $h_{t-1} \in \mathbb{R}^n$:

### Reset Gate

Controls how much of the previous state to forget when computing the candidate:

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

- When $r_t \approx 0$: Previous state is "reset" (ignored in candidate computation)
- When $r_t \approx 1$: Previous state fully influences the candidate

### Update Gate

Controls the interpolation between old and new states:

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- When $z_t \approx 0$: Keep the previous state (like LSTM forget gate ≈ 1)
- When $z_t \approx 1$: Use the new candidate (like LSTM input gate ≈ 1)

### Candidate Hidden State

New information computed with reset gate modulation:

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

The reset gate $r_t$ appears *inside* the tanh, multiplying $h_{t-1}$ before concatenation with $x_t$. This allows the model to completely ignore the previous state when computing new content.

### Hidden State Update

Interpolation between previous state and candidate:

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Where:

- $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\odot$ denotes element-wise (Hadamard) multiplication
- $[a, b]$ denotes concatenation along feature dimension
- $W_r, W_z, W_h \in \mathbb{R}^{n \times (n+d)}$ are weight matrices
- $b_r, b_z, b_h \in \mathbb{R}^n$ are bias vectors

---

## Understanding the Gates

### The Update Gate as Combined Forget/Input

The update equation reveals an elegant constraint:

$$h_t = \underbrace{(1 - z_t) \odot h_{t-1}}_{\text{retained information}} + \underbrace{z_t \odot \tilde{h}_t}_{\text{new information}}$$

This is a **convex combination** (weights sum to 1):

| $z_t$ value | Effect | LSTM equivalent |
|-------------|--------|-----------------|
| $z_t \approx 0$ | Keep old: $h_t \approx h_{t-1}$ | Forget gate ≈ 1, Input gate ≈ 0 |
| $z_t \approx 1$ | Use new: $h_t \approx \tilde{h}_t$ | Forget gate ≈ 0, Input gate ≈ 1 |
| $z_t = 0.5$ | Equal mix | Partial update |

**Critical insight:** In LSTM, forget gate $f_t$ and input gate $i_t$ are independent—you can simultaneously forget everything ($f_t = 0$) and add nothing ($i_t = 0$). In GRU, the complementary weighting $(1-z_t)$ and $z_t$ **enforces conservation**: the more you discard old information, the more you must replace it with new information.

### The Reset Gate: Enabling Sharp Transitions

The reset gate serves a different purpose than any single LSTM gate. When $r_t \approx 0$:

$$\tilde{h}_t \approx \tanh(W_h \cdot [\mathbf{0}, x_t] + b_h)$$

The candidate is computed as if there were no previous state—a "fresh start" based purely on current input. This is crucial for sentence boundaries, topic changes, and anomaly detection.

**Interaction between gates:** The reset gate affects *what candidate to propose*, while the update gate decides *how much of that candidate to use*:

1. Reset gate: "Should the new candidate consider history?"
2. Update gate: "Should we use this new candidate at all?"

---

## Gradient Flow Analysis

### GRU's Gradient Path

The gradient through the hidden state:

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - z_t) + z_t \odot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

The first term $(1 - z_t)$ provides a **direct gradient path**. When $z_t \approx 0$ (keep old state):

$$\frac{\partial h_t}{\partial h_{t-1}} \approx I$$

Gradients flow unchanged through this path, preventing vanishing gradients.

### Comparison with LSTM

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gradient path | Through cell state: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ | Through interpolation: $(1-z_t)$ |
| Mechanism | Additive update | Interpolative update |
| When gates ≈ 1 | $f_t \approx 1$: gradient preserved | $z_t \approx 0$: gradient preserved |
| Independence | Cell and hidden gradients separate | Single gradient path |

---

## PyTorch Implementation from Scratch

### GRU Cell (Educational Version)

```python
import torch
import torch.nn as nn
import numpy as np

class GRUCell(nn.Module):
    """
    Manual GRU cell implementation for educational purposes.
    
    Implements the standard GRU equations with separate weight matrices
    for clarity. Production implementations combine matrices for efficiency.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset gate parameters
        self.W_r = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # Update gate parameters
        self.W_z = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # Candidate parameters
        self.W_h = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size) / np.sqrt(input_size + hidden_size)
        )
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for a single timestep.
        
        Args:
            x: Input tensor (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
        
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Concatenate for gate computations
        combined = torch.cat([h_prev, x], dim=1)
        
        # Reset gate
        r = torch.sigmoid(combined @ self.W_r.t() + self.b_r)
        
        # Update gate
        z = torch.sigmoid(combined @ self.W_z.t() + self.b_z)
        
        # Candidate hidden state with reset modulation
        combined_reset = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(combined_reset @ self.W_h.t() + self.b_h)
        
        # Final hidden state: interpolation
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

### GRU Cell (Efficient Version)

```python
class GRUCellEfficient(nn.Module):
    """
    Efficient GRU cell with combined weight matrices.
    
    Combines all gate computations into single matrix multiplications
    for better GPU utilization.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(
            torch.randn(3 * hidden_size, input_size) / np.sqrt(input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        gi = x @ self.weight_ih.t() + self.bias_ih
        gh = h_prev @ self.weight_hh.t() + self.bias_hh
        
        n = self.hidden_size
        r_i, z_i, n_i = gi[:, :n], gi[:, n:2*n], gi[:, 2*n:]
        r_h, z_h, n_h = gh[:, :n], gh[:, n:2*n], gh[:, 2*n:]
        
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        
        # Candidate: reset gate applied to hidden contribution only
        h_tilde = torch.tanh(n_i + r * n_h)
        
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
```

### Full GRU Layer

```python
class GRU(nn.Module):
    """
    Full GRU layer processing sequences.
    
    Supports multiple stacked layers and dropout between layers.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            GRUCellEfficient(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process an entire sequence through the GRU.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden states (num_layers, batch_size, hidden_size)
        
        Returns:
            output: All hidden states from top layer (batch_size, seq_len, hidden_size)
            h_n: Final hidden states for all layers (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.size()
        
        if h_0 is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [h_0[i] for i in range(self.num_layers)]
        
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(layer_input, h[layer_idx])
                layer_input = h[layer_idx]
                
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        
        return output, h_n
```

---

## Using PyTorch's Built-in GRU

```python
import torch
import torch.nn as nn

# Create GRU layer
gru = nn.GRU(
    input_size=100,      # Dimension of input features
    hidden_size=256,     # Dimension of hidden state
    num_layers=2,        # Number of stacked GRU layers
    batch_first=True,    # Input shape: (batch, seq, features)
    dropout=0.3,         # Dropout between layers (only if num_layers > 1)
    bidirectional=False  # Unidirectional by default
)

# Input: (batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# Forward pass
output, h_n = gru(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) - one per layer

# Key difference from LSTM: no cell state!
# LSTM returns: output, (h_n, c_n)
# GRU returns:  output, h_n
```

---

## Practical Applications

### Text Classification

```python
class GRUClassifier(nn.Module):
    """GRU for sequence classification."""
    
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=2, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, h_n = self.gru(embedded)
        logits = self.fc(self.dropout(h_n[-1]))
        return logits
```

### Time Series Forecasting

```python
class GRUForecaster(nn.Module):
    """GRU for multi-step time series forecasting."""
    
    def __init__(self, input_size, hidden_size, num_layers, 
                 output_size, horizon=1, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size * horizon)
    
    def forward(self, x):
        _, h_n = self.gru(x)
        predictions = self.fc(h_n[-1])
        return predictions.view(-1, self.horizon, self.output_size)


# Example: Predict next 5 steps of 3 features from 20 historical steps
model = GRUForecaster(input_size=3, hidden_size=128, num_layers=2,
                      output_size=3, horizon=5)

x = torch.randn(32, 20, 3)
predictions = model(x)  # (32, 5, 3)
```

### Bidirectional GRU

```python
class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU for sequence encoding.
    
    Processes sequences in both directions, capturing both
    past and future context at each position.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=True)
    
    def forward(self, x):
        outputs, h_n = self.gru(x)
        # outputs: (batch, seq, hidden * 2) - already concatenated
        
        # h_n: (num_layers * 2, batch, hidden) → (num_layers, batch, hidden * 2)
        batch_size = x.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_n = h_n.permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, -1)
        
        return outputs, h_n
```

---

## Initialization Best Practices

```python
def init_gru_weights(gru: nn.GRU):
    """
    Initialize GRU weights for stable training.
    
    - Input-hidden weights: Xavier uniform
    - Hidden-hidden weights: Orthogonal (preserves gradient norm)
    - Biases: Zero (or optionally bias reset gate toward 1)
    """
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # Optional: bias reset gate toward "remember"
            # n = param.size(0) // 3
            # param.data[:n].fill_(1.0)  # Reset gate bias
```

**Why orthogonal initialization for hidden weights?** Orthogonal matrices have all singular values equal to 1, so gradient magnitude is preserved through matrix multiplication. This helps with gradient flow through the recurrence.

---

## GRU Variants

### Minimal GRU

Single gate variant that further reduces parameters:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$\tilde{h}_t = \tanh(W_h \cdot [h_{t-1}, x_t] + b_h)$$
$$h_t = f_t \odot h_{t-1} + (1 - f_t) \odot \tilde{h}_t$$

This removes the reset gate entirely, using only an update/forget gate. Even simpler but less expressive.

```python
class MinimalGRUCell(nn.Module):
    """Minimal GRU with single gate."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        combined_size = input_size + hidden_size
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_h = nn.Linear(combined_size, hidden_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h, x], dim=-1)
        f = torch.sigmoid(self.W_f(combined))
        h_tilde = torch.tanh(self.W_h(combined))
        return f * h + (1 - f) * h_tilde
```

### GRU with Coupled Gates

Further simplification where reset and update gates are coupled:

$$r_t = 1 - z_t$$

This enforces that when you're updating a lot ($z_t$ high), you also reset more ($r_t$ low), and vice versa.

---

## Parameter Efficiency

```python
def compare_parameters(input_size: int = 100, hidden_size: int = 256):
    """Compare parameter counts between architectures."""
    rnn_params = hidden_size * (input_size + hidden_size) + hidden_size
    gru_params = 3 * rnn_params
    lstm_params = 4 * rnn_params
    
    print(f"Parameter comparison (input={input_size}, hidden={hidden_size}):")
    print(f"  Vanilla RNN: {rnn_params:,} (1.0×)")
    print(f"  GRU:         {gru_params:,} (3.0×)")
    print(f"  LSTM:        {lstm_params:,} (4.0×)")
    print(f"\n  GRU savings vs LSTM: {(1 - gru_params/lstm_params)*100:.1f}%")

# compare_parameters()
```

---

## Summary

GRU offers an elegant simplification of LSTM:

| Feature | GRU Implementation |
|---------|-------------------|
| Memory management | Single hidden state (no separate cell) |
| Forget mechanism | Update gate: $(1-z_t)$ factor |
| Input mechanism | Update gate: $z_t$ factor (complementary) |
| Reset for transitions | Reset gate: $r_t$ modulates candidate |
| Gradient flow | Through $(1-z_t)$ direct path |

**Key advantages:**

- 25% fewer parameters than LSTM
- Faster training and inference
- Comparable performance on most tasks
- Simpler to understand and debug

**Key trade-off:**

- Less flexible than LSTM's independent gates
- Coupled forget/input may limit some use cases

**Bottom line:** GRU is an excellent default choice for sequence modeling. Its efficiency makes it preferable when computational resources are limited or when quick iteration is important. Reserve LSTM for tasks that clearly benefit from its additional complexity.

---

## References

1. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

2. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.

3. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

4. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

5. Heck, J., & Salem, F. M. (2017). Simplified Minimal Gated Unit Variations for Recurrent Neural Networks. *MWSCAS*.
