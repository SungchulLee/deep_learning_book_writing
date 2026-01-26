# GRU Architecture

## Introduction

The Gated Recurrent Unit (GRU), introduced by Cho et al. in 2014, is a streamlined variant of LSTM that achieves comparable performance with fewer parameters. GRU simplifies the LSTM architecture by merging the cell state and hidden state into a single state vector and reducing the number of gates from three to two.

This architectural simplification offers practical advantages: faster training, lower memory footprint, and reduced risk of overfitting on smaller datasets—all while maintaining the ability to capture long-range dependencies that vanilla RNNs cannot learn.

**Core idea:** Where LSTM uses separate pathways for "what to remember" (cell state) and "what to output" (hidden state), GRU combines these into a single hidden state with two gates controlling information flow.

---

## GRU vs LSTM: Structural Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| States | Hidden ($h$) + Cell ($c$) | Hidden ($h$) only |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Parameters | $4(n^2 + nd + n)$ | $3(n^2 + nd + n)$ |
| Memory update | Additive (via cell state) | Interpolative |
| Relative parameters | 1.0× | 0.75× |
| Output gating | Explicit output gate | No separate output gate |

**Key structural insight:** LSTM's cell state acts as a "memory highway" with additive updates. GRU achieves a similar effect through interpolation—the hidden state is a weighted average of the previous state and a candidate, with weights summing to 1.

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

## Understanding the Gates: Deep Dive

### The Update Gate as Combined Forget/Input

The update equation reveals an elegant constraint:

$$h_t = \underbrace{(1 - z_t) \odot h_{t-1}}_{\text{retained information}} + \underbrace{z_t \odot \tilde{h}_t}_{\text{new information}}$$

This is a **convex combination** (weights sum to 1):

| $z_t$ value | Effect | LSTM equivalent |
|-------------|--------|-----------------|
| $z_t \approx 0$ | Keep old: $h_t \approx h_{t-1}$ | Forget gate ≈ 1, Input gate ≈ 0 |
| $z_t \approx 1$ | Use new: $h_t \approx \tilde{h}_t$ | Forget gate ≈ 0, Input gate ≈ 1 |
| $z_t = 0.5$ | Equal mix: $h_t = 0.5 h_{t-1} + 0.5 \tilde{h}_t$ | Partial update |

**Critical insight:** In LSTM, forget gate $f_t$ and input gate $i_t$ are independent—you can simultaneously forget everything ($f_t = 0$) and add nothing ($i_t = 0$). In GRU, the complementary weighting $(1-z_t)$ and $z_t$ **enforces conservation**: the more you discard old information, the more you must replace it with new information.

This constraint can be:
- **Beneficial**: Prevents pathological states, provides implicit regularization
- **Limiting**: Less flexibility than LSTM's independent gating

### The Reset Gate: Enabling Sharp Transitions

The reset gate serves a different purpose than any single LSTM gate. Consider the candidate computation:

$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

When $r_t \approx 0$:

$$\tilde{h}_t \approx \tanh(W_h \cdot [\mathbf{0}, x_t] + b_h)$$

The candidate is computed as if there were no previous state—a "fresh start" based purely on current input. This is crucial for:

- **Sentence boundaries**: Starting a new sentence shouldn't be influenced by the previous one
- **Topic changes**: Detecting that context has shifted
- **Anomaly detection**: Recognizing that current input is unrelated to history

When $r_t \approx 1$:

$$\tilde{h}_t = \tanh(W_h \cdot [h_{t-1}, x_t] + b_h)$$

The candidate is computed normally, allowing smooth continuation.

**Interaction between gates:** The reset gate affects *what candidate to propose*, while the update gate decides *how much of that candidate to use*. These are complementary roles:

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

**Subtle difference:** LSTM's cell state gradient is $f_t$; when $f_t = 1$, gradients flow perfectly. GRU's direct path has factor $(1-z_t)$; when $z_t = 0$, gradients flow perfectly. The conditions are "opposite" in gate naming but achieve the same effect.

### Empirical Gradient Flow

```python
import numpy as np

def analyze_gradient_flow(sequence_length: int = 100, n_simulations: int = 1000):
    """Compare gradient flow between GRU and vanilla RNN."""
    
    gru_gradients = []
    rnn_gradients = []
    
    for _ in range(n_simulations):
        # GRU: update gate typically low (keep information)
        # z ~ Beta(2, 5) biases toward low values
        z_values = np.random.beta(2, 5, sequence_length)
        gru_grad = np.prod(1 - z_values)  # Direct path only
        gru_gradients.append(gru_grad)
        
        # Vanilla RNN: gradient through tanh and weight matrix
        # Effective multiplier typically < 1
        rnn_multiplier = np.random.uniform(0.5, 0.9, sequence_length)
        rnn_grad = np.prod(rnn_multiplier)
        rnn_gradients.append(rnn_grad)
    
    print(f"Gradient magnitude after {sequence_length} steps:")
    print(f"  GRU (median):  {np.median(gru_gradients):.6f}")
    print(f"  GRU (mean):    {np.mean(gru_gradients):.6f}")
    print(f"  RNN (median):  {np.median(rnn_gradients):.2e}")
    print(f"  RNN (mean):    {np.mean(rnn_gradients):.2e}")
    print(f"  GRU preserves ~{np.median(gru_gradients)/np.median(rnn_gradients):.0e}× more gradient")

# analyze_gradient_flow()
```

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
        combined = torch.cat([h_prev, x], dim=1)  # (batch, hidden + input)
        
        # Reset gate: r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        r = torch.sigmoid(combined @ self.W_r.t() + self.b_r)
        
        # Update gate: z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        z = torch.sigmoid(combined @ self.W_z.t() + self.b_z)
        
        # Candidate hidden state: h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
        combined_reset = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(combined_reset @ self.W_h.t() + self.b_h)
        
        # Final hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
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
        
        # Combined weights: [W_r; W_z; W_h] for input
        # Shape: (3 * hidden_size, input_size)
        self.weight_ih = nn.Parameter(
            torch.randn(3 * hidden_size, input_size) / np.sqrt(input_size)
        )
        
        # Combined weights for hidden state
        # Shape: (3 * hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(
            torch.randn(3 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        
        # Combined biases
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
    
    def forward(
        self, 
        x: torch.Tensor, 
        h_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
        
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Compute all gate pre-activations at once
        gi = x @ self.weight_ih.t() + self.bias_ih      # Input contribution
        gh = h_prev @ self.weight_hh.t() + self.bias_hh  # Hidden contribution
        
        # Split into gates
        n = self.hidden_size
        r_i, z_i, n_i = gi[:, :n], gi[:, n:2*n], gi[:, 2*n:]
        r_h, z_h, n_h = gh[:, :n], gh[:, n:2*n], gh[:, 2*n:]
        
        # Reset and update gates
        r = torch.sigmoid(r_i + r_h)
        z = torch.sigmoid(z_i + z_h)
        
        # Candidate (reset gate applied to hidden contribution only)
        h_tilde = torch.tanh(n_i + r * n_h)
        
        # Output
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
        
        # Create cells for each layer
        self.cells = nn.ModuleList([
            GRUCellEfficient(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Dropout between layers
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
        
        # Initialize hidden states
        if h_0 is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h = [h_0[i] for i in range(self.num_layers)]
        
        # Process sequence timestep by timestep
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(layer_input, h[layer_idx])
                layer_input = h[layer_idx]
                
                # Apply dropout between layers (not after final layer)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])
        
        # Stack outputs and hidden states
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        h_n = torch.stack(h, dim=0)           # (num_layers, batch, hidden)
        
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
    """
    GRU for sequence classification (sentiment, spam, topic, etc.)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int, 
        hidden_size: int, 
        num_classes: int,
        num_layers: int = 2, 
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch_size, seq_len)
        
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        embedded = self.embedding(x)          # (batch, seq, embed)
        embedded = self.dropout(embedded)
        
        gru_out, h_n = self.gru(embedded)     # h_n: (layers, batch, hidden)
        
        # Use final layer's hidden state for classification
        final_hidden = self.dropout(h_n[-1])  # (batch, hidden)
        logits = self.fc(final_hidden)        # (batch, num_classes)
        
        return logits
```

### Time Series Forecasting

```python
class GRUForecaster(nn.Module):
    """
    GRU for multi-step time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int,      # Number of input features
        hidden_size: int,
        num_layers: int,
        output_size: int,     # Number of output features
        horizon: int = 1,     # Forecast horizon (steps ahead)
        dropout: float = 0.2
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size * horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        
        Returns:
            predictions: Forecast (batch_size, horizon, output_size)
        """
        _, h_n = self.gru(x)
        
        # Use final hidden state for prediction
        predictions = self.fc(h_n[-1])  # (batch, output_size * horizon)
        predictions = predictions.view(-1, self.horizon, self.output_size)
        
        return predictions


# Example: Predict next 5 steps of 3 features from 20 historical steps
model = GRUForecaster(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    horizon=5
)

x = torch.randn(32, 20, 3)  # 32 samples, 20 timesteps, 3 features
predictions = model(x)       # (32, 5, 3)
```

### Sequence Generation

```python
class GRUGenerator(nn.Module):
    """
    GRU for autoregressive sequence generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Token indices (batch_size, seq_len)
            hidden: Previous hidden state (num_layers, batch_size, hidden_size)
        
        Returns:
            logits: Next-token logits (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc(self.dropout(output))
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
    
    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None
    ) -> list[int]:
        """
        Generate tokens autoregressively.
        
        Args:
            start_tokens: Initial token indices (seq_len,)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, sample only from top-k tokens
        
        Returns:
            List of all tokens (prompt + generated)
        """
        self.eval()
        device = next(self.parameters()).device
        
        tokens = start_tokens.tolist()
        hidden = self.init_hidden(1, device)
        
        # Process prompt
        x = torch.tensor([tokens], device=device)
        _, hidden = self.forward(x, hidden)
        
        # Generate new tokens
        current_token = torch.tensor([[tokens[-1]]], device=device)
        
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(current_token, hidden)
            logits = logits[0, -1, :] / temperature
            
            if top_k is not None:
                topk_logits, topk_indices = torch.topk(logits, top_k)
                probs = torch.softmax(topk_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = topk_indices[idx].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)
        
        return tokens
```

### Bidirectional GRU

```python
class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU for sequence encoding.
    
    Processes sequences in both directions, capturing both
    past and future context at each position.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        
        Returns:
            outputs: Concatenated forward/backward hidden states
                     (batch_size, seq_len, hidden_size * 2)
            h_n: Final hidden states, reshaped to
                 (num_layers, batch_size, hidden_size * 2)
        """
        outputs, h_n = self.gru(x)
        # outputs: (batch, seq, hidden * 2) - already concatenated
        
        # h_n: (num_layers * 2, batch, hidden)
        # Reshape to (num_layers, batch, hidden * 2)
        batch_size = x.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_n = h_n.permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, -1)
        
        return outputs, h_n
```

---

## Parameter Efficiency Analysis

```python
def compare_parameters(input_size: int = 100, hidden_size: int = 256):
    """Compare parameter counts between architectures."""
    
    # Formula: gates × (hidden × (input + hidden) + hidden)
    rnn_params = hidden_size * (input_size + hidden_size) + hidden_size
    gru_params = 3 * rnn_params
    lstm_params = 4 * rnn_params
    
    print(f"Parameter comparison (input={input_size}, hidden={hidden_size}):")
    print(f"  Vanilla RNN: {rnn_params:,} (1.0×)")
    print(f"  GRU:         {gru_params:,} (3.0×)")
    print(f"  LSTM:        {lstm_params:,} (4.0×)")
    print(f"\n  GRU savings vs LSTM: {(1 - gru_params/lstm_params)*100:.1f}%")
    
    # Real PyTorch comparison
    print("\nPyTorch verification:")
    gru = nn.GRU(input_size, hidden_size)
    lstm = nn.LSTM(input_size, hidden_size)
    
    gru_total = sum(p.numel() for p in gru.parameters())
    lstm_total = sum(p.numel() for p in lstm.parameters())
    
    print(f"  nn.GRU:  {gru_total:,}")
    print(f"  nn.LSTM: {lstm_total:,}")

# compare_parameters()
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
            # This helps early gradient flow
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

## When to Choose GRU vs LSTM

| Consideration | Choose GRU | Choose LSTM |
|---------------|------------|-------------|
| Training speed | ✓ ~25% faster | Slower |
| Memory usage | ✓ ~25% less | More |
| Small datasets | ✓ Less overfitting | May overfit |
| Very long sequences | Good | ✓ May be better |
| Fine-grained memory control | Limited | ✓ Independent gates |
| Default choice | ✓ Try first | If GRU insufficient |

**Practical guidance:**

1. **Start with GRU** for most tasks
2. **Switch to LSTM** if GRU underperforms
3. The difference is often marginal—benchmark on your specific task
4. Consider computational constraints (edge deployment, large-scale training)

### Empirical Findings

From Jozefowicz et al. (2015) and Greff et al. (2017):

- No consistent winner across all tasks
- GRU often matches LSTM with fewer parameters
- Task-specific tuning matters more than architecture choice
- Both significantly outperform vanilla RNN

---

## Debugging Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss doesn't decrease | Learning rate wrong | Try 1e-3, then tune |
| Loss explodes | Gradient explosion | Add gradient clipping |
| Outputs constant | Vanishing gradients or dead neurons | Check initialization |
| Poor long-range | Hidden size too small | Increase hidden_size |
| Overfitting | Model too large | Add dropout, reduce layers |

### Training Loop Template

```python
def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_norm: float = 5.0,
    device: torch.device = torch.device('cpu')
) -> float:
    """Training loop with gradient clipping."""
    
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (essential for RNNs!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # Update
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
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
