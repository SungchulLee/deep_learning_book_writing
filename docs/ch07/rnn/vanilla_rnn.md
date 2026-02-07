# Vanilla RNN

## Architecture Overview

The vanilla (Elman) RNN is the simplest recurrent architecture, defined by a single hidden state updated at each timestep through a linear transformation followed by a $\tanh$ nonlinearity. Despite its simplicity, understanding the vanilla RNN thoroughly is essential—it establishes the computational patterns that all more sophisticated recurrent architectures build upon.

The update equations:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

The $\tanh$ activation bounds hidden state values to $[-1, 1]$, preventing unbounded growth while permitting both positive and negative activations. This choice has important consequences for gradient flow that we will examine in the BPTT section.

## Implementation from Scratch

### RNN Cell

The cell computes a single timestep of the recurrence:

```python
import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """Single RNN cell computing one timestep."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Separate linear layers for clarity
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
```

The bias is included only in `W_xh` since adding bias to both would be redundant—the two bias vectors would simply be summed.

### Complete RNN Module

The full module iterates the cell over a sequence:

```python
class VanillaRNN(nn.Module):
    """Complete RNN processing full sequences."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = VanillaRNNCell(input_size, hidden_size)
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
        
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        h = h_0
        outputs = []
        
        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)
            outputs.append(h)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        return outputs, h
```

The explicit loop over timesteps makes the sequential nature clear—each hidden state depends on the previous one, preventing parallelization.

## Using PyTorch's Built-in RNN

PyTorch provides an optimized `nn.RNN` with fused CUDA kernels:

```python
rnn = nn.RNN(
    input_size=10,      # Feature dimension of input
    hidden_size=20,     # Dimension of hidden state
    num_layers=1,       # Number of stacked RNN layers
    batch_first=True,   # Input shape: (batch, seq, features)
    nonlinearity='tanh' # Activation function ('tanh' or 'relu')
)

# Input: (batch_size=32, seq_len=15, input_size=10)
x = torch.randn(32, 15, 10)

# Forward pass
outputs, h_n = rnn(x)

print(f"Outputs shape: {outputs.shape}")  # (32, 15, 20)
print(f"Final hidden: {h_n.shape}")       # (1, 32, 20)
```

The `outputs` tensor contains hidden states at every timestep while `h_n` contains only the final hidden state. The leading dimension of `h_n` corresponds to `num_layers`—for a single-layer RNN this is 1.

### Weight Access

```python
# PyTorch names weights differently from our notation
print(rnn.weight_ih_l0.shape)  # (20, 10) - W_xh (input-to-hidden)
print(rnn.weight_hh_l0.shape)  # (20, 20) - W_hh (hidden-to-hidden)
print(rnn.bias_ih_l0.shape)    # (20,)    - b_xh
print(rnn.bias_hh_l0.shape)    # (20,)    - b_hh
```

## Sequence Classification

For classifying an entire sequence into a category, we typically use the final hidden state as the sequence representation:

```python
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len) - token indices
        embedded = self.embedding(x)        # (batch, seq_len, embed_dim)
        outputs, h_n = self.rnn(embedded)   # h_n: (1, batch, hidden)
        
        h_final = h_n.squeeze(0)            # (batch, hidden)
        logits = self.fc(h_final)           # (batch, num_classes)
        return logits
```

The final hidden state $h_T$ theoretically encodes all sequence information, though in practice vanilla RNNs struggle to retain information from early timesteps.

## Autoregressive Generation

For text generation, the model produces outputs at each timestep and feeds predictions back as inputs:

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
            
            # Sample from temperature-scaled distribution
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            
        return tokens
```

During generation, the hidden state persists across calls—each new token is conditioned on the full history compressed into $h_t$.

## Activation Function Choice

The vanilla RNN supports two activation functions:

**$\tanh$** (default): outputs in $[-1, 1]$ with derivative $1 - \tanh^2(x) \in [0, 1]$. The bounded derivative contributes to vanishing gradients but ensures stable activations.

**ReLU**: outputs in $[0, \infty)$ with derivative either 0 or 1. Can mitigate vanishing gradients but risks exploding activations since hidden states are no longer bounded. ReLU RNNs require careful initialization—Le et al. (2015) showed that initializing $W_{hh}$ as the identity matrix with ReLU activation can match LSTM performance on certain tasks.

```python
# ReLU variant
rnn_relu = nn.RNN(
    input_size=10, hidden_size=20,
    batch_first=True, nonlinearity='relu'
)
```

## Hyperparameter Guidelines

| Parameter | Typical Range | Considerations |
|-----------|--------------|----------------|
| `hidden_size` | 64–512 | Larger for complex tasks; diminishing returns beyond input complexity |
| `num_layers` | 1–3 | See Deep RNN section |
| `dropout` | 0.0–0.5 | Between layers only; not within recurrence |
| `nonlinearity` | `tanh` | `relu` requires identity initialization for $W_{hh}$ |

## Limitations of Vanilla RNNs

The vanilla RNN's simplicity brings fundamental limitations:

**Short effective memory**: the repeated multiplication by $W_{hh}$ causes gradients to vanish (or explode), limiting the practical memory horizon to roughly 10–20 timesteps.

**No selective memory**: every timestep overwrites the hidden state through the same $\tanh$ transformation. There is no mechanism to selectively preserve, update, or forget information—a capability that gated architectures (LSTM, GRU) explicitly provide.

**Sequential computation**: the $h_t \to h_{t+1}$ dependency prevents parallelization across timesteps, making training slower than architectures like Transformers that process all positions simultaneously.

These limitations motivate the progression from vanilla RNNs to LSTM and GRU architectures, which introduce gating mechanisms to control information flow through the hidden state.

## Summary

The vanilla RNN establishes the foundational recurrent computation: a hidden state updated by combining current input with previous memory through a $\tanh$ nonlinearity. Its from-scratch implementation reveals the core loop that all RNN variants share. While PyTorch's `nn.RNN` provides optimized execution, understanding the underlying mechanics—weight sharing, sequential processing, and the role of the hidden state as compressed history—is essential for diagnosing training issues and motivating more advanced architectures.
