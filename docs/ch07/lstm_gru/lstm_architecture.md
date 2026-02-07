# LSTM Architecture

## Introduction

Long Short-Term Memory (LSTM) networks are a specialized form of recurrent neural networks designed to capture long-range dependencies in sequential data. Introduced by Hochreiter and Schmidhuber in 1997, LSTMs address the fundamental limitation of vanilla RNNs: the vanishing gradient problem that prevents learning from distant time steps.

The key innovation of LSTM lies in its **cell state**—a dedicated memory pathway that allows information to flow across many time steps with minimal transformation. This "memory highway" enables LSTMs to selectively remember information for hundreds or even thousands of time steps.

**Core architectural insight:** LSTM maintains two state vectors:

- **Hidden state** $h_t$: Short-term memory, the output of the cell
- **Cell state** $c_t$: Long-term memory, the information highway

Three learnable gates regulate information flow:

- **Forget gate** $f_t$: What to discard from cell state
- **Input gate** $i_t$: What new information to store
- **Output gate** $o_t$: What to output from cell state

---

## The Vanishing Gradient Problem

Before understanding LSTM architecture, we must appreciate the problem it solves.

### The Problem in Vanilla RNNs

In a vanilla RNN, the hidden state update follows:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

During backpropagation through time (BPTT), gradients flow through this recurrence:

$$\frac{\partial h_t}{\partial h_0} = \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=1}^{t} W_{hh}^\top \cdot \text{diag}(\tanh'(z_k))$$

**Two compounding problems:**

1. **Activation saturation:** Since $|\tanh'(x)| \leq 1$ (with equality only at $x=0$), repeated multiplication through saturated activations drives gradients toward zero.

2. **Spectral properties of $W_{hh}$:** If the largest singular value $\sigma_{\max}(W_{hh}) < 1$, gradients vanish exponentially. If $\sigma_{\max}(W_{hh}) > 1$, gradients explode.

**Consequence:** For a sequence of length $T$, the gradient contribution from time step $t$ to the loss at time $T$ decays as $O(\lambda^{T-t})$ where $\lambda < 1$. Information from distant past is effectively invisible during training.

### LSTM's Solution: Additive Updates

LSTM replaces the multiplicative recurrence with an **additive** cell state update:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The gradient through the cell state pathway:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

When $f_t \approx 1$ (forget gate open), gradients flow **unchanged**:

$$\frac{\partial \mathcal{L}}{\partial c_{t-k}} = \frac{\partial \mathcal{L}}{\partial c_t} \cdot \prod_{i=t-k}^{t-1} f_{i+1}$$

Unlike vanilla RNNs where gradients pass through weight matrices and saturating activations, LSTM gradients pass through learned gates that can remain near 1. This is the "constant error carousel"—gradients can flow for hundreds of steps without vanishing.

!!! note "The Key Insight"
    LSTM doesn't eliminate gradient issues entirely—it provides a **learnable bypass** around them. The network learns when to remember ($f_t \approx 1$) and when to forget ($f_t \approx 0$), allowing task-relevant gradients to flow while discarding irrelevant information.

---

## LSTM Cell Architecture

An LSTM cell contains four main components: three gates (forget, input, output) and a cell state update mechanism.

### Complete Mathematical Formulation

For time step $t$, given input $x_t \in \mathbb{R}^{d}$, previous hidden state $h_{t-1} \in \mathbb{R}^{n}$, and previous cell state $c_{t-1} \in \mathbb{R}^{n}$:

**Forget Gate** — determines what information to discard from the cell state:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

When $f_t \approx 0$, the gate forgets; when $f_t \approx 1$, it remembers.

**Input Gate** — determines what new information to store:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Cell Candidate** — creates candidate values to add to the cell state:

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Cell State Update** — combines old memory (gated by forget) with new information (gated by input):

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output Gate** — determines what to output based on the cell state:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State** — filtered version of the cell state:

$$h_t = o_t \odot \tanh(c_t)$$

Where:

- $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- $\odot$ denotes element-wise (Hadamard) product
- $[h_{t-1}, x_t]$ denotes concatenation along feature dimension
- $W_f, W_i, W_c, W_o \in \mathbb{R}^{n \times (n+d)}$ are weight matrices
- $b_f, b_i, b_c, b_o \in \mathbb{R}^{n}$ are bias vectors

### Why These Specific Nonlinearities?

The choice of activation functions is deliberate:

| Component | Activation | Range | Rationale |
|-----------|------------|-------|-----------|
| Gates ($f_t, i_t, o_t$) | Sigmoid | $(0, 1)$ | Soft binary switches; smooth interpolation between "off" and "on" |
| Cell candidate ($\tilde{c}_t$) | Tanh | $(-1, 1)$ | Centered at zero; allows both positive and negative updates |
| Output transform | Tanh | $(-1, 1)$ | Normalizes cell state before gating |

**Why sigmoid for gates?** Gates need to make soft binary decisions. Sigmoid provides smooth gradients everywhere while approximating a switch.

**Why tanh for content?** The cell state can accumulate over time. Tanh bounds new contributions to $[-1, 1]$, preventing unbounded growth while allowing both additive and subtractive updates.

---

## Information Flow Visualization

The LSTM can be understood as having two parallel pathways:

### Cell State Pathway (Memory Highway)

```
c_{t-1} ──[×f_t]──[+]── c_t ──→
                   ↑
            [×i_t]─┘
               ↑
           c̃_t (candidate)
```

The cell state undergoes only element-wise operations—no matrix multiplications, no activation functions in the main pathway. This is the "information highway" that preserves gradients.

### Hidden State Pathway (Output)

```
c_t ──[tanh]──[×o_t]── h_t ──→ output
```

The hidden state is a filtered, transformed view of the cell state. It goes through nonlinearities because it's used for:

1. Computing the current output/prediction
2. Informing the next timestep's gate computations

**Design insight:** The cell state stores raw information; the hidden state presents a task-relevant view of that information.

---

## Dimensionality Analysis

For an LSTM with input dimension $d$ and hidden dimension $n$:

| Component | Shape | Description |
|-----------|-------|-------------|
| $x_t$ | $(d,)$ | Input vector |
| $h_{t-1}, h_t$ | $(n,)$ | Hidden states |
| $c_{t-1}, c_t$ | $(n,)$ | Cell states |
| $[h_{t-1}, x_t]$ | $(n+d,)$ | Concatenated input |
| $W_f, W_i, W_c, W_o$ | $(n, n+d)$ | Weight matrices |
| $b_f, b_i, b_c, b_o$ | $(n,)$ | Bias vectors |
| $f_t, i_t, o_t$ | $(n,)$ | Gate activations |
| $\tilde{c}_t$ | $(n,)$ | Cell candidate |

### Parameter Count

**Total parameters:**

$$\text{Parameters} = 4 \times [n \times (n + d) + n] = 4n^2 + 4nd + 4n = 4n(n + d + 1)$$

**Example:** For $d = 100$ (input) and $n = 256$ (hidden):

$$\text{Parameters} = 4 \times 256 \times (256 + 100 + 1) = 4 \times 256 \times 357 = 365{,}568$$

### Comparison with Other Architectures

| Architecture | Parameters | Relative to RNN |
|--------------|------------|-----------------|
| Vanilla RNN | $n^2 + nd + n$ | 1× |
| GRU | $3(n^2 + nd + n)$ | 3× |
| LSTM | $4(n^2 + nd + n)$ | 4× |

The 4× increase comes from the four weight matrices (forget, input, cell, output), each with the same dimensions as the single RNN weight matrix.

!!! tip "Computational Trade-off"
    LSTM's 4× parameter increase yields dramatically better gradient flow. In practice, you can often use smaller hidden dimensions with LSTM than vanilla RNN while achieving better performance, partially offsetting the parameter cost.

---

## PyTorch Implementation from Scratch

### LSTM Cell

```python
import torch
import torch.nn as nn
import numpy as np

class LSTMCell(nn.Module):
    """
    Manual LSTM cell implementation for educational purposes.
    
    Implements the standard LSTM equations with combined weight matrices
    for computational efficiency.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for efficiency: [W_i, W_f, W_c, W_o] stacked
        # This allows computing all gates in a single matrix multiplication
        # Shape: (4 * hidden_size, input_size) for input weights
        # Shape: (4 * hidden_size, hidden_size) for hidden weights
        self.weight_ih = nn.Parameter(
            torch.randn(4 * hidden_size, input_size) / np.sqrt(input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        
        # Initialize forget gate bias to 1 (critical for gradient flow)
        self._init_forget_gate_bias()
    
    def _init_forget_gate_bias(self):
        """
        Initialize forget gate bias to 1.
        
        This encourages the model to remember by default at the start of 
        training, preventing early gradient vanishing before the network
        learns what to forget.
        """
        with torch.no_grad():
            # Gates are ordered: [input, forget, cell, output]
            # Forget gate bias is in the second quarter
            n = self.hidden_size
            self.bias[n:2*n].fill_(1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (h_prev, c_prev), each (batch_size, hidden_size)
                   If None, initializes to zeros.
        
        Returns:
            h_new: New hidden state (batch_size, hidden_size)
            c_new: New cell state (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize states if not provided
        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_prev, c_prev = state
        
        # Compute all gates in one matrix multiplication for efficiency
        # gates = W_ih @ x + W_hh @ h_prev + bias
        gates = (x @ self.weight_ih.t() + 
                 h_prev @ self.weight_hh.t() + 
                 self.bias)
        
        # Split into individual gates
        n = self.hidden_size
        i_gate = torch.sigmoid(gates[:, 0*n:1*n])      # Input gate
        f_gate = torch.sigmoid(gates[:, 1*n:2*n])      # Forget gate
        c_tilde = torch.tanh(gates[:, 2*n:3*n])        # Cell candidate
        o_gate = torch.sigmoid(gates[:, 3*n:4*n])      # Output gate
        
        # Cell state update: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        c_new = f_gate * c_prev + i_gate * c_tilde
        
        # Hidden state: h_t = o_t ⊙ tanh(c_t)
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class LSTM(nn.Module):
    """
    Full LSTM layer processing sequences.
    
    Supports multiple stacked layers and returns all hidden states
    for downstream processing.
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
        # First layer takes input_size, subsequent layers take hidden_size
        self.cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        # Dropout between layers (not after final layer)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Process an entire sequence through the LSTM.
        
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            state: Initial states as tuple of:
                   - h_0: (num_layers, batch_size, hidden_size)
                   - c_0: (num_layers, batch_size, hidden_size)
        
        Returns:
            output: All hidden states from top layer (batch_size, seq_len, hidden_size)
            (h_n, c_n): Final states for all layers, same shape as input states
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states for all layers
        if state is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h = [state[0][i] for i in range(self.num_layers)]
            c = [state[1][i] for i in range(self.num_layers)]
        
        # Process sequence timestep by timestep
        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            # Pass through each layer
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx], c[layer_idx] = cell(
                    layer_input, (h[layer_idx], c[layer_idx])
                )
                layer_input = h[layer_idx]
                
                # Apply dropout between layers (not after final layer)
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    layer_input = self.dropout(layer_input)
            
            outputs.append(h[-1])  # Collect output from top layer
        
        # Stack outputs and states
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        h_n = torch.stack(h, dim=0)           # (num_layers, batch, hidden)
        c_n = torch.stack(c, dim=0)           # (num_layers, batch, hidden)
        
        return output, (h_n, c_n)
```

### Verification Against PyTorch

```python
def verify_implementation():
    """Verify our implementation matches PyTorch's."""
    torch.manual_seed(42)
    
    batch_size, seq_len, input_size, hidden_size = 4, 10, 8, 16
    
    # Our implementation
    our_lstm = LSTM(input_size, hidden_size, num_layers=1)
    
    # PyTorch's implementation
    torch_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
    
    # Copy weights (accounting for different ordering)
    with torch.no_grad():
        # PyTorch orders gates as [i, f, g, o], we use same ordering
        torch_lstm.weight_ih_l0.copy_(our_lstm.cells[0].weight_ih)
        torch_lstm.weight_hh_l0.copy_(our_lstm.cells[0].weight_hh)
        torch_lstm.bias_ih_l0.copy_(our_lstm.cells[0].bias)
        torch_lstm.bias_hh_l0.zero_()  # We combine biases; PyTorch splits them
    
    # Test input
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    our_output, (our_h, our_c) = our_lstm(x)
    torch_output, (torch_h, torch_c) = torch_lstm(x)
    
    # Compare
    print(f"Output close: {torch.allclose(our_output, torch_output, atol=1e-5)}")
    print(f"Hidden close: {torch.allclose(our_h, torch_h, atol=1e-5)}")
    print(f"Cell close: {torch.allclose(our_c, torch_c, atol=1e-5)}")

# verify_implementation()
```

---

## Using PyTorch's Built-in LSTM

```python
import torch
import torch.nn as nn

# Create LSTM layer
lstm = nn.LSTM(
    input_size=100,      # Dimension of input features
    hidden_size=256,     # Dimension of hidden/cell state
    num_layers=2,        # Number of stacked LSTM layers
    batch_first=True,    # Input shape: (batch, seq, features)
    dropout=0.3,         # Dropout between layers (only if num_layers > 1)
    bidirectional=False  # Process forward only
)

# Input: (batch_size, sequence_length, input_size)
x = torch.randn(32, 50, 100)

# Forward pass
output, (h_n, c_n) = lstm(x)

print(f"Output shape: {output.shape}")     # (32, 50, 256)
print(f"Final hidden: {h_n.shape}")        # (2, 32, 256) - one per layer
print(f"Final cell: {c_n.shape}")          # (2, 32, 256) - one per layer

# Accessing outputs for downstream tasks
last_output = output[:, -1, :]    # Last timestep: (32, 256)
last_hidden = h_n[-1]             # Final layer's hidden: (32, 256)
# Note: last_output == last_hidden for unidirectional LSTM
```

---

## Practical Applications

### Sequence Classification

```python
class LSTMClassifier(nn.Module):
    """
    LSTM for sequence classification (sentiment analysis, spam detection, etc.)
    
    Uses the final hidden state as a fixed-dimensional representation
    of the entire sequence for classification.
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
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
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
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use final hidden state for classification
        # h_n shape: (num_layers, batch, hidden_size)
        # Take the last layer's hidden state
        final_hidden = h_n[-1]  # (batch, hidden_size)
        final_hidden = self.dropout(final_hidden)
        
        # Classify
        logits = self.fc(final_hidden)  # (batch, num_classes)
        
        return logits


# Example usage
model = LSTMClassifier(
    vocab_size=30000,
    embed_dim=128,
    hidden_size=256,
    num_classes=2,  # Binary classification
    num_layers=2,
    dropout=0.5
)

# Input: batch of 32 sequences, each with 100 tokens
x = torch.randint(0, 30000, (32, 100))
logits = model(x)  # (32, 2)
probs = torch.softmax(logits, dim=-1)
```

### Sequence Generation (Language Model)

```python
class LSTMLanguageModel(nn.Module):
    """
    LSTM for autoregressive text generation.
    
    Predicts the next token given previous tokens, enabling
    text generation through iterative sampling.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        tie_weights: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project hidden state to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Optionally tie embedding and output weights
        if tie_weights and embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Token indices (batch_size, seq_len)
            hidden: Previous (h, c) states
        
        Returns:
            logits: Next-token logits (batch_size, seq_len, vocab_size)
            hidden: Updated (h, c) states
        """
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden states to zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
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
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample only from top-k most likely tokens
        
        Returns:
            List of generated token indices
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
            logits = logits[0, -1, :] / temperature  # (vocab_size,)
            
            # Optional top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[idx].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
            current_token = torch.tensor([[next_token]], device=device)
        
        return tokens
```

### Time Series Forecasting

```python
class LSTMForecaster(nn.Module):
    """
    LSTM for multivariate time series forecasting.
    
    Given a sequence of observations, predicts future values
    for a specified forecast horizon.
    """
    
    def __init__(
        self,
        input_size: int,     # Number of input features
        hidden_size: int,
        num_layers: int,
        output_size: int,    # Number of output features
        horizon: int = 1,    # Forecast horizon (steps ahead)
        dropout: float = 0.2
    ):
        super().__init__()
        self.horizon = horizon
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Predict all horizon steps at once
        self.fc = nn.Linear(hidden_size, output_size * horizon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
        
        Returns:
            predictions: Forecast (batch_size, horizon, output_size)
        """
        # Process sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state for prediction
        final_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Predict all steps
        predictions = self.fc(final_hidden)  # (batch, output_size * horizon)
        predictions = predictions.view(-1, self.horizon, self.output_size)
        
        return predictions


# Example: Predict next 5 steps of 3 features from 20 historical steps
model = LSTMForecaster(
    input_size=3,
    hidden_size=128,
    num_layers=2,
    output_size=3,
    horizon=5
)

x = torch.randn(32, 20, 3)  # 32 samples, 20 timesteps, 3 features
predictions = model(x)       # (32, 5, 3)
```

---

## Initialization and Training Best Practices

### Weight Initialization

```python
def init_lstm_weights(lstm: nn.LSTM):
    """
    Initialize LSTM weights following best practices.
    
    - Input-hidden weights: Xavier/Glorot uniform
    - Hidden-hidden weights: Orthogonal (preserves gradient norm)
    - Biases: Zero, except forget gate bias = 1
    """
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden weights: Xavier uniform
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-hidden weights: Orthogonal
            # This helps maintain gradient magnitude through recurrence
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Biases: mostly zero
            nn.init.zeros_(param)
            # But set forget gate bias to 1
            n = param.size(0) // 4
            param.data[n:2*n].fill_(1.0)
```

### Why Forget Gate Bias = 1?

At initialization with random weights, gate activations are approximately 0.5 (sigmoid of small random values). This means the model starts by forgetting ~50% of information at each timestep—problematic for learning long-range dependencies.

Setting $b_f = 1$ biases the forget gate sigmoid input, so:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + 1) \approx \sigma(1) \approx 0.73$$

This "remember by default" initialization allows gradients to flow early in training. The network can then learn to forget specific information as needed.

### Gradient Clipping

```python
# During training, clip gradients to prevent explosion
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

LSTM addresses vanishing gradients but can still suffer from exploding gradients, especially early in training. Gradient clipping is standard practice.

### Learning Rate Scheduling

```python
# Reduce learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

# After each epoch
scheduler.step(val_loss)
```

---

## LSTM Variants

### Coupled Forget-Input Gates

Observation: If we're forgetting old information, we're usually adding new information, and vice versa. Coupling the gates:

$$c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t$$

This reduces parameters and enforces the constraint that total "information mass" is conserved.

### GRU (Gated Recurrent Unit)

GRU simplifies LSTM by:

1. Merging cell state and hidden state
2. Combining forget and input gates into an "update gate"
3. Using a "reset gate" instead of output gate

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**Trade-off:** Fewer parameters (3× vs 4×), similar performance on many tasks, but less flexible memory control.

---

## When to Use LSTM

### LSTM Strengths

- **Long sequences**: Effective up to hundreds of timesteps
- **Variable-length sequences**: Natural handling via recurrence
- **Online/streaming processing**: Can process one timestep at a time
- **Sequential inductive bias**: Explicit modeling of temporal order

### When to Consider Alternatives

| Situation | Alternative | Reason |
|-----------|-------------|--------|
| Very long sequences (1000+) | Transformer | Better parallelization, attention over full context |
| Simple short sequences | GRU | Fewer parameters, faster |
| Massive parallelism needed | 1D CNN, Transformer | LSTM is inherently sequential |
| Global dependencies | Transformer | Self-attention captures long-range directly |
| Edge deployment | GRU or custom | Lower memory footprint |

### Modern Context

While Transformers have largely superseded LSTMs for NLP tasks, LSTMs remain competitive or superior for:

- Time series with strong sequential structure
- Low-latency streaming applications
- Small dataset regimes (less prone to overfitting)
- Edge deployment (predictable memory usage)

---

## Debugging Common Issues

### Symptoms and Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss doesn't decrease | Learning rate too high/low | Try 1e-3, then adjust |
| Loss oscillates wildly | Exploding gradients | Add gradient clipping |
| Model outputs constant | Dead neurons or vanishing gradients | Check forget gate bias = 1 |
| Poor long-range performance | Insufficient hidden size | Increase hidden_size |
| Overfitting | Model too large or no regularization | Add dropout, reduce size |
| Slow training | Sequence too long | Use truncated BPTT |

### Diagnostic Code

```python
def diagnose_lstm(model, dataloader, device):
    """Run diagnostics on LSTM training."""
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # Check gradient magnitudes
            model.train()
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"{name}: grad_norm = {grad_norm:.6f}")
            
            break
```

---

## Summary

LSTM networks solve the vanishing gradient problem through:

1. **Cell state**: Additive information highway that preserves gradients
2. **Forget gate**: Learns what information to discard
3. **Input gate**: Learns what new information to store
4. **Output gate**: Learns what information to output

**Key properties:**

- Gradients flow through cell state without multiplicative decay
- Gates learn task-specific information flow patterns
- Captures dependencies over hundreds of timesteps
- 4× parameters vs vanilla RNN, but dramatically better gradient flow

**Best practices:**

- Initialize forget gate bias to 1
- Use gradient clipping
- Apply dropout between layers
- Consider GRU if parameters are constrained

The architecture adds computational cost but enables learning from much longer sequences—essential for most practical sequence modeling tasks.

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

3. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.

4. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

5. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.
