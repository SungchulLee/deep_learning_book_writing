# RNN Architecture and Hidden States

## The Recurrence Principle

Recurrent Neural Networks (RNNs) solve the sequential modeling problem through a elegantly simple idea: maintain a hidden state that accumulates information across timesteps. At each step, the network combines new input with its memory of past inputs to produce both an output and an updated memory.

## Core Architecture

The fundamental RNN computation at timestep $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

Where:
- $x_t \in \mathbb{R}^{d}$: Input at timestep $t$ (dimension $d$)
- $h_t \in \mathbb{R}^{H}$: Hidden state at timestep $t$ (dimension $H$)
- $y_t \in \mathbb{R}^{o}$: Output at timestep $t$ (dimension $o$)
- $W_{xh} \in \mathbb{R}^{H \times d}$: Input-to-hidden weights
- $W_{hh} \in \mathbb{R}^{H \times H}$: Hidden-to-hidden weights
- $W_{hy} \in \mathbb{R}^{o \times H}$: Hidden-to-output weights
- $b_h, b_y$: Bias vectors

## Unrolling Through Time

The recurrence creates a computational graph when "unrolled" across timesteps:

```
    x₁       x₂       x₃       x₄
    ↓        ↓        ↓        ↓
h₀ → RNN → h₁ → RNN → h₂ → RNN → h₃ → RNN → h₄
           ↓        ↓        ↓        ↓
           y₁       y₂       y₃       y₄
```

Critically, the **same weights** $(W_{xh}, W_{hh}, W_{hy})$ are shared across all timesteps—this parameter sharing enables processing sequences of arbitrary length.

## PyTorch Implementation from Scratch

```python
import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    """Single RNN cell computing one timestep."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weight matrix for efficiency
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, h_prev):
        """
        Args:
            x: Input tensor (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h


class SimpleRNN(nn.Module):
    """Complete RNN processing full sequences."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = SimpleRNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h_0=None):
        """
        Args:
            x: Input sequence (batch_size, seq_len, input_size)
            h_0: Initial hidden state (batch_size, hidden_size)
        Returns:
            outputs: All hidden states (batch_size, seq_len, hidden_size)
            h_n: Final hidden state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process sequence
        h = h_0
        outputs = []
        
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        return outputs, h
```

## Using PyTorch's Built-in RNN

```python
import torch.nn as nn

# Create RNN layer
rnn = nn.RNN(
    input_size=10,      # Feature dimension of input
    hidden_size=20,     # Dimension of hidden state
    num_layers=1,       # Number of stacked RNN layers
    batch_first=True,   # Input shape: (batch, seq, features)
    nonlinearity='tanh' # Activation function
)

# Input: (batch_size=32, seq_len=15, input_size=10)
x = torch.randn(32, 15, 10)

# Forward pass
outputs, h_n = rnn(x)

print(f"Outputs shape: {outputs.shape}")  # (32, 15, 20)
print(f"Final hidden shape: {h_n.shape}") # (1, 32, 20)
```

## Understanding Hidden State Dimensions

The hidden state serves multiple roles:

**Memory**: Encodes accumulated information from $x_1, \ldots, x_{t-1}$

**Context**: Provides context for interpreting current input $x_t$

**Representation**: Serves as the sequence's learned representation for downstream tasks

Hidden state dimension $H$ is a hyperparameter balancing:
- **Larger $H$**: More capacity to store complex patterns, but more parameters and computation
- **Smaller $H$**: Faster training, potential information bottleneck

## Multi-Layer RNNs

Stacking RNN layers increases model capacity:

```python
# Two-layer RNN
rnn_stacked = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(32, 15, 10)
outputs, h_n = rnn_stacked(x)

# outputs: (32, 15, 20) - top layer's hidden states
# h_n: (2, 32, 20) - final hidden state from each layer
```

In multi-layer RNNs:
- Layer 1 processes the input sequence
- Layer 2 receives Layer 1's outputs as its input
- Each layer maintains its own hidden state

```
Layer 2:  h₁² → h₂² → h₃² → h₄²  (higher-level patterns)
           ↑     ↑     ↑     ↑
Layer 1:  h₁¹ → h₂¹ → h₃¹ → h₄¹  (lower-level patterns)
           ↑     ↑     ↑     ↑
Input:    x₁    x₂    x₃    x₄
```

## Information Flow Analysis

At each timestep, information flows through three paths:

**1. Direct Input Path**: $W_{xh} x_t$

New information from current input directly influences the hidden state.

**2. Recurrent Path**: $W_{hh} h_{t-1}$

Historical information carried forward through the hidden state.

**3. Nonlinear Combination**: $\tanh(\cdot)$

The activation function creates complex interactions between current and past information.

## Computational Graph and Memory

For a sequence of length $T$:

**Time Complexity**: $O(T)$ - must process sequentially
**Space Complexity**: $O(T \cdot H)$ - store all hidden states for backpropagation

This sequential nature prevents parallelization—a key limitation addressed by Transformers.

## Classification Example

For sequence classification, we typically use the final hidden state:

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        outputs, h_n = self.rnn(embedded)  # h_n: (1, batch, hidden)
        
        # Use final hidden state for classification
        h_final = h_n.squeeze(0)  # (batch, hidden)
        logits = self.fc(h_final)  # (batch, num_classes)
        return logits
```

## Generation Example

For text generation, we produce outputs at each timestep:

```python
class RNNGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded, hidden)
        logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
        return logits, hidden
    
    def generate(self, start_token, max_length, temperature=1.0):
        """Autoregressive generation."""
        tokens = [start_token]
        hidden = None
        
        for _ in range(max_length - 1):
            x = torch.tensor([[tokens[-1]]])
            logits, hidden = self.forward(x, hidden)
            
            # Sample from distribution
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            
        return tokens
```

## Hyperparameter Guidelines

| Parameter | Typical Range | Considerations |
|-----------|--------------|----------------|
| hidden_size | 64-512 | Larger for complex tasks |
| num_layers | 1-3 | Diminishing returns beyond 2-3 |
| dropout | 0.0-0.5 | Between layers, not within |

## Summary

RNNs introduce recurrence to neural networks, enabling sequential modeling through:

- **Hidden states**: Compress historical information into fixed-size vectors
- **Parameter sharing**: Same weights process each timestep
- **Unrolling**: Creates computational graph for backpropagation

The architecture elegantly solves sequential modeling but faces challenges with long sequences due to vanishing gradients—motivating LSTM and GRU architectures covered next.
