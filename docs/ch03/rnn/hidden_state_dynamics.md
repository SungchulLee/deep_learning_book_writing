# Hidden State Dynamics

## Introduction

The hidden state is the defining feature of recurrent neural networks—a vector that carries information across timesteps, serving as the network's memory. Understanding how hidden states evolve, what information they encode, and how to interpret them is essential for effective RNN design and debugging.

## The Hidden State as Memory

At each timestep $t$, the hidden state $h_t$ theoretically encodes all relevant information from inputs $x_1, x_2, \ldots, x_t$:

$$h_t = f(x_t, h_{t-1}) = f(x_t, f(x_{t-1}, f(x_{t-2}, \ldots)))$$

This recursive definition means $h_t$ is a compressed representation of the entire history—a lossy compression that retains information useful for the task at hand.

## Hidden State Update Equation

For a vanilla RNN:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Breaking this down:

| Term | Meaning |
|------|---------|
| $W_{xh} x_t$ | How current input influences hidden state |
| $W_{hh} h_{t-1}$ | How previous memory influences current state |
| $\tanh(\cdot)$ | Squashes values to $[-1, 1]$, introduces nonlinearity |

## Visualizing Hidden State Evolution

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def visualize_hidden_states(model, sequence, vocab=None):
    """
    Visualize how hidden states evolve over a sequence.
    
    Args:
        model: RNN model with embedding and rnn layers
        sequence: Input tensor (1, seq_len)
        vocab: Optional vocabulary for token labels
    """
    model.eval()
    
    hidden_states = []
    
    # Hook to capture hidden states
    with torch.no_grad():
        embedded = model.embedding(sequence)
        h = None
        
        for t in range(sequence.size(1)):
            out, h = model.rnn(embedded[:, t:t+1, :], h)
            hidden_states.append(h.squeeze().cpu().numpy())
    
    hidden_states = np.array(hidden_states)  # (seq_len, hidden_size)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hidden_states.T, aspect='auto', cmap='RdBu_r', 
                   vmin=-1, vmax=1)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Hidden Dimension')
    ax.set_title('Hidden State Evolution')
    
    # Add token labels if vocabulary provided
    if vocab is not None:
        tokens = [vocab.idx2token.get(idx.item(), '?') for idx in sequence[0]]
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.colorbar(im, label='Activation')
    plt.tight_layout()
    plt.show()
    
    return hidden_states
```

## Information Encoding in Hidden States

### Positional Information

Hidden states implicitly encode position through accumulated transformations:

```python
def analyze_position_encoding(rnn, seq_length=50, hidden_size=128):
    """Check if hidden states encode positional information."""
    
    # Feed identical inputs at each position
    constant_input = torch.ones(1, seq_length, 1)
    
    outputs, _ = rnn(constant_input)
    
    # Hidden states should differ despite identical inputs
    # due to different "positions" in the sequence
    hidden_states = outputs.squeeze().detach().numpy()
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(hidden_states))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(distances, cmap='viridis')
    plt.colorbar(label='Euclidean Distance')
    plt.xlabel('Timestep')
    plt.ylabel('Timestep')
    plt.title('Hidden State Distances (Constant Input)')
    plt.show()
```

### Content-Based Representations

Hidden states also encode semantic content:

```python
def compare_hidden_states(model, sentences, vocab):
    """Compare final hidden states for different sentences."""
    
    final_states = []
    labels = []
    
    for sentence in sentences:
        # Tokenize and encode
        tokens = [vocab.token2idx.get(w, vocab.token2idx['<UNK>']) 
                  for w in sentence.lower().split()]
        x = torch.tensor([tokens])
        
        # Get final hidden state
        with torch.no_grad():
            _, (h_n, _) = model.lstm(model.embedding(x))
            final_states.append(h_n[-1].squeeze().numpy())
            labels.append(sentence[:20] + '...' if len(sentence) > 20 else sentence)
    
    # Visualize with PCA or t-SNE
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(final_states))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
    plt.title('Final Hidden States (PCA)')
    plt.show()
```

## Hidden State Initialization

### Zero Initialization (Default)

```python
def init_hidden_zero(batch_size, hidden_size, num_layers, device):
    """Standard zero initialization."""
    h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    return (h, c)
```

### Learned Initialization

Learn optimal initial hidden state:

```python
class RNNWithLearnedInit(nn.Module):
    """RNN with learnable initial hidden state."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Learnable initial states
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Expand learned init to batch size
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        
        return self.rnn(x, (h0, c0))
```

### Encoder-Initialized (Seq2Seq)

Initialize decoder with encoder's final state:

```python
class Seq2SeqDecoder(nn.Module):
    def __init__(self, ...):
        # ... initialization ...
        pass
    
    def forward(self, x, encoder_hidden):
        """
        Args:
            x: Decoder input
            encoder_hidden: (h_n, c_n) from encoder
        """
        # Use encoder's final state as decoder's initial state
        return self.lstm(x, encoder_hidden)
```

## Hidden State Dynamics Analysis

### Saturation Detection

Monitor for saturated activations:

```python
def check_saturation(hidden_states, threshold=0.95):
    """
    Check if hidden states are saturating (close to ±1 for tanh).
    
    Args:
        hidden_states: Tensor of hidden states
        threshold: Values above this are considered saturated
    
    Returns:
        Saturation statistics
    """
    saturated = (hidden_states.abs() > threshold).float()
    saturation_rate = saturated.mean().item()
    
    print(f"Saturation rate: {saturation_rate:.2%}")
    print(f"Mean |h|: {hidden_states.abs().mean():.4f}")
    print(f"Max |h|: {hidden_states.abs().max():.4f}")
    
    return {
        'saturation_rate': saturation_rate,
        'mean_abs': hidden_states.abs().mean().item(),
        'max_abs': hidden_states.abs().max().item()
    }
```

### Effective Memory Length

Measure how far back information propagates:

```python
def measure_effective_memory(model, seq_length=100, num_trials=50):
    """
    Estimate effective memory by measuring gradient magnitude
    from output to each input position.
    """
    model.train()
    
    gradient_magnitudes = torch.zeros(seq_length)
    
    for _ in range(num_trials):
        x = torch.randn(1, seq_length, model.input_size, requires_grad=True)
        
        outputs, _ = model(x)
        
        # Backprop from final output
        loss = outputs[0, -1, :].sum()
        loss.backward()
        
        # Measure gradient at each timestep
        grad_norms = x.grad[0].norm(dim=-1)
        gradient_magnitudes += grad_norms.detach()
    
    gradient_magnitudes /= num_trials
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_length), gradient_magnitudes.numpy())
    plt.xlabel('Timestep (distance from output)')
    plt.ylabel('Gradient Magnitude')
    plt.title('Effective Memory: Gradient Flow to Past Inputs')
    plt.yscale('log')
    plt.show()
    
    # Find effective memory length (where gradient drops below threshold)
    threshold = gradient_magnitudes.max() * 0.01
    effective_length = (gradient_magnitudes > threshold).sum().item()
    print(f"Effective memory length: ~{effective_length} timesteps")
    
    return gradient_magnitudes
```

## Hidden State Regularization

### Activity Regularization

Penalize large hidden state activations:

```python
def activity_regularization_loss(hidden_states, alpha=1e-4):
    """
    L2 penalty on hidden state activations.
    Encourages sparse, moderate activations.
    """
    return alpha * hidden_states.pow(2).mean()

# Usage in training
outputs, hidden = model(x)
task_loss = criterion(outputs, targets)
reg_loss = activity_regularization_loss(outputs)
total_loss = task_loss + reg_loss
```

### Temporal Activation Regularization

Penalize differences between consecutive hidden states:

```python
def temporal_activation_regularization(hidden_states, beta=1e-4):
    """
    Penalize large changes between consecutive hidden states.
    Encourages smooth temporal dynamics.
    """
    # hidden_states: (batch, seq_len, hidden_size)
    diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    return beta * diff.pow(2).mean()
```

## Stateful vs Stateless RNNs

### Stateless (Default)

Hidden state resets for each sequence:

```python
# Each call starts fresh
for batch in dataloader:
    outputs, _ = model(batch)  # h_0 = 0
```

### Stateful

Carry hidden state across batches (for very long sequences):

```python
class StatefulRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden = None
    
    def forward(self, x):
        # Detach to prevent backprop through previous batches
        if self.hidden is not None:
            self.hidden = tuple(h.detach() for h in self.hidden)
        
        outputs, self.hidden = self.rnn(x, self.hidden)
        return outputs
    
    def reset_hidden(self):
        """Call at sequence boundaries."""
        self.hidden = None

# Usage
model = StatefulRNN(input_size=10, hidden_size=64)

for epoch in range(epochs):
    model.reset_hidden()  # Reset at start of each epoch/document
    
    for batch in dataloader:
        outputs = model(batch)  # Hidden state carries over
```

## Debugging Hidden States

### Common Issues and Diagnostics

| Symptom | Possible Cause | Diagnostic |
|---------|---------------|------------|
| All zeros | Vanishing gradients | Check gradient norms |
| All ±1 | Saturation | Check saturation rate |
| Identical across batch | Bug in batching | Print h for different samples |
| NaN values | Exploding gradients | Add gradient clipping |
| No learning | Dead neurons | Check activation distribution |

```python
def diagnose_hidden_states(model, dataloader, device):
    """Run diagnostics on hidden state behavior."""
    
    all_hidden = []
    
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            _, (h_n, _) = model.lstm(model.embedding(x))
            all_hidden.append(h_n[-1])  # Final layer
    
    hidden = torch.cat(all_hidden, dim=0)
    
    print("Hidden State Diagnostics")
    print("=" * 40)
    print(f"Shape: {hidden.shape}")
    print(f"Mean: {hidden.mean():.6f}")
    print(f"Std: {hidden.std():.6f}")
    print(f"Min: {hidden.min():.6f}")
    print(f"Max: {hidden.max():.6f}")
    print(f"% near zero (|h| < 0.01): {(hidden.abs() < 0.01).float().mean():.2%}")
    print(f"% saturated (|h| > 0.99): {(hidden.abs() > 0.99).float().mean():.2%}")
    print(f"NaN count: {hidden.isnan().sum().item()}")
    print(f"Inf count: {hidden.isinf().sum().item()}")
```

## Summary

Hidden state dynamics determine what information RNNs can capture and propagate:

1. **Memory mechanism**: $h_t$ compresses history into fixed-size vector
2. **Update equation**: Combines current input with previous memory
3. **Initialization**: Zero, learned, or encoder-initialized
4. **Capacity limits**: Effective memory decays with sequence length

Key diagnostics:
- Monitor saturation rates
- Measure effective memory length via gradients
- Check for dead or exploding activations

Understanding hidden state behavior helps diagnose training issues and design better architectures—motivating gated mechanisms (LSTM, GRU) that explicitly control information flow.
