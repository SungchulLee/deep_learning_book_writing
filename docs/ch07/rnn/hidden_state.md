# Hidden State Dynamics

## The Hidden State as Memory

The hidden state is the defining feature of recurrent neural networks—a vector that carries information across timesteps, serving as the network's memory. At each timestep $t$, the hidden state $h_t$ theoretically encodes all relevant information from inputs $x_1, x_2, \ldots, x_t$:

$$h_t = f(x_t, h_{t-1}) = f(x_t, f(x_{t-1}, f(x_{t-2}, \ldots)))$$

This recursive definition means $h_t$ is a compressed representation of the entire history—a lossy compression that retains information useful for the task at hand. Understanding how hidden states evolve, what information they encode, and how to interpret them is essential for effective RNN design and debugging.

## Hidden State Update Mechanics

For a vanilla RNN, the update equation is:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Breaking this into its constituent contributions:

| Term | Role |
|------|------|
| $W_{xh} x_t$ | How current input influences hidden state |
| $W_{hh} h_{t-1}$ | How previous memory influences current state |
| $\tanh(\cdot)$ | Squashes values to $[-1, 1]$, introduces nonlinearity |

The hidden state dimension $H$ is a hyperparameter balancing capacity against efficiency. Larger $H$ provides more capacity to store complex patterns but increases parameters ($W_{hh}$ alone has $H^2$ parameters) and computation. Smaller $H$ trains faster but may create an information bottleneck.

## What Hidden States Encode

### Positional Information

Hidden states implicitly encode position through accumulated transformations. Even when fed identical inputs at every position, the hidden states diverge because each additional application of $W_{hh}$ and $\tanh$ produces a different vector:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def analyze_position_encoding(rnn, seq_length=50):
    """Check if hidden states encode positional information."""
    constant_input = torch.ones(1, seq_length, 1)
    
    with torch.no_grad():
        outputs, _ = rnn(constant_input)
    
    hidden_states = outputs.squeeze().numpy()
    
    # Pairwise distances reveal position-dependent structure
    distances = squareform(pdist(hidden_states))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(distances, cmap='viridis')
    plt.colorbar(label='Euclidean Distance')
    plt.xlabel('Timestep')
    plt.ylabel('Timestep')
    plt.title('Hidden State Distances (Constant Input)')
    plt.show()
```

Nearby timesteps produce similar hidden states, while distant timesteps diverge—an emergent positional encoding.

### Content-Based Representations

Hidden states also encode semantic content. After training, sequences with similar meaning produce similar final hidden states, enabling downstream tasks like classification and retrieval:

```python
def compare_hidden_states(model, sentences, vocab):
    """Compare final hidden states for different sentences."""
    from sklearn.decomposition import PCA
    
    final_states = []
    labels = []
    
    for sentence in sentences:
        tokens = [vocab.token2idx.get(w, vocab.token2idx['<UNK>']) 
                  for w in sentence.lower().split()]
        x = torch.tensor([tokens])
        
        with torch.no_grad():
            _, h_n = model.rnn(model.embedding(x))
            final_states.append(h_n.squeeze().numpy())
            labels.append(sentence[:30])
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(final_states))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
    plt.title('Final Hidden States (PCA Projection)')
    plt.show()
```

## Visualizing Hidden State Evolution

A heatmap of hidden state activations across timesteps reveals which dimensions respond to which inputs:

```python
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
    
    with torch.no_grad():
        embedded = model.embedding(sequence)
        h = None
        for t in range(sequence.size(1)):
            out, h = model.rnn(embedded[:, t:t+1, :], h)
            hidden_states.append(h.squeeze().cpu().numpy())
    
    hidden_states = np.array(hidden_states)  # (seq_len, hidden_size)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(hidden_states.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Hidden Dimension')
    ax.set_title('Hidden State Evolution')
    
    if vocab is not None:
        tokens = [vocab.idx2token.get(idx.item(), '?') for idx in sequence[0]]
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.colorbar(im, label='Activation')
    plt.tight_layout()
    plt.show()
    
    return hidden_states
```

## Hidden State Initialization

### Zero Initialization (Default)

The simplest and most common approach initializes $h_0 = \mathbf{0}$:

```python
def init_hidden_zero(batch_size, hidden_size, num_layers, device):
    """Standard zero initialization."""
    return torch.zeros(num_layers, batch_size, hidden_size, device=device)
```

This works well in practice because the network learns to "warm up" its hidden state from zeros within the first few timesteps.

### Learned Initialization

A learnable initial hidden state can improve performance, especially when the beginning of a sequence carries important information:

```python
class RNNWithLearnedInit(nn.Module):
    """RNN with learnable initial hidden state."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Learnable initial state
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        return self.rnn(x, h0)
```

### Encoder-Initialized (Seq2Seq)

In encoder-decoder architectures, the decoder's initial hidden state comes from the encoder's final state, transferring the compressed source representation:

```python
class Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
    def forward(self, x, encoder_hidden):
        """Use encoder's final state as decoder's initial state."""
        return self.rnn(x, encoder_hidden)
```

## Stateful vs. Stateless RNNs

### Stateless (Default)

The hidden state resets to zero for each new sequence. Each forward call is independent:

```python
for batch in dataloader:
    outputs, _ = model(batch)  # h_0 = 0 each time
```

### Stateful

For processing very long sequences that are split across batches, the hidden state carries over. The critical detail is detaching the hidden state to prevent backpropagation across batch boundaries:

```python
class StatefulRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.hidden = None
    
    def forward(self, x):
        if self.hidden is not None:
            self.hidden = self.hidden.detach()  # Truncate gradient flow
        
        outputs, self.hidden = self.rnn(x, self.hidden)
        return outputs
    
    def reset_hidden(self):
        """Call at document/sequence boundaries."""
        self.hidden = None

# Usage
model = StatefulRNN(input_size=10, hidden_size=64)

for epoch in range(num_epochs):
    model.reset_hidden()
    for batch in dataloader:
        outputs = model(batch)  # Hidden state persists
```

The `.detach()` call is essential: without it, the backward pass would attempt to propagate gradients through all previous batches, consuming unbounded memory.

## Effective Memory Length

The practical memory horizon of an RNN can be measured by examining how gradient magnitude decays with distance from the loss:

```python
def measure_effective_memory(model, seq_length=100, input_size=10, num_trials=50):
    """
    Estimate effective memory by measuring gradient magnitude
    from output to each input position.
    """
    model.train()
    gradient_magnitudes = torch.zeros(seq_length)
    
    for _ in range(num_trials):
        x = torch.randn(1, seq_length, input_size, requires_grad=True)
        outputs, _ = model(x)
        
        loss = outputs[0, -1, :].sum()
        loss.backward()
        
        grad_norms = x.grad[0].norm(dim=-1)
        gradient_magnitudes += grad_norms.detach()
    
    gradient_magnitudes /= num_trials
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_length), gradient_magnitudes.numpy())
    plt.xlabel('Timestep (distance from output)')
    plt.ylabel('Gradient Magnitude')
    plt.title('Effective Memory: Gradient Flow to Past Inputs')
    plt.yscale('log')
    plt.show()
    
    # Effective length: where gradient drops below 1% of maximum
    threshold = gradient_magnitudes.max() * 0.01
    effective_length = (gradient_magnitudes > threshold).sum().item()
    print(f"Effective memory length: ~{effective_length} timesteps")
    
    return gradient_magnitudes
```

For vanilla RNNs, this effective memory is typically 10–20 timesteps—much shorter than the actual sequence length.

## Hidden State Regularization

### Activity Regularization

An $L_2$ penalty on hidden state activations encourages moderate, well-distributed activations:

$$\mathcal{L}_{\text{reg}} = \alpha \cdot \frac{1}{T} \sum_{t=1}^{T} \| h_t \|^2$$

```python
def activity_regularization_loss(hidden_states, alpha=1e-4):
    """L2 penalty on hidden state activations."""
    return alpha * hidden_states.pow(2).mean()

# In training loop
outputs, hidden = model(x)
task_loss = criterion(outputs, targets)
reg_loss = activity_regularization_loss(outputs)
total_loss = task_loss + reg_loss
```

### Temporal Activation Regularization

Penalizing large differences between consecutive hidden states encourages smooth temporal dynamics, acting as a form of temporal smoothing:

$$\mathcal{L}_{\text{TAR}} = \beta \cdot \frac{1}{T-1} \sum_{t=1}^{T-1} \| h_{t+1} - h_t \|^2$$

```python
def temporal_activation_regularization(hidden_states, beta=1e-4):
    """Penalize large changes between consecutive hidden states."""
    diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
    return beta * diff.pow(2).mean()
```

## Saturation Detection

When hidden states saturate (approach $\pm 1$ for $\tanh$), gradients vanish locally because $\tanh'(x) \to 0$ for large $|x|$. Monitoring saturation rates is a key diagnostic:

```python
def check_saturation(hidden_states, threshold=0.95):
    """
    Check if hidden states are saturating (close to ±1 for tanh).
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

## Debugging Hidden States

Common issues and their diagnostics:

| Symptom | Possible Cause | Diagnostic |
|---------|---------------|------------|
| All zeros | Vanishing gradients | Check gradient norms |
| All ±1 | Saturation | Check saturation rate |
| Identical across batch | Batching bug | Print $h$ for different samples |
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
            _, h_n = model.rnn(model.embedding(x))
            all_hidden.append(h_n[-1])
    
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

Hidden state dynamics determine what information RNNs can capture and propagate. The hidden state $h_t$ compresses the full input history into a fixed-size vector through the update equation that combines current input with previous memory. Initialization (zero, learned, or encoder-initialized) and statefulness (stateless vs. stateful across batches) are design choices with practical consequences for convergence and long-range modeling.

Key diagnostics—saturation monitoring, effective memory measurement via gradients, and activation distribution analysis—help identify training issues early. Understanding these dynamics motivates gated mechanisms (LSTM, GRU) that explicitly control what information to retain, update, and forget.
