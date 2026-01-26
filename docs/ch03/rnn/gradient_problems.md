# Vanishing and Exploding Gradients

## The Core Problem

Recurrent Neural Networks face a fundamental challenge: gradients must flow backward through many timesteps during training. This repeated multiplication through the recurrence relation causes gradients to either shrink exponentially (vanish) or grow exponentially (explode), making learning over long sequences extremely difficult.

## Mathematical Analysis

### Gradient Flow Through Time

Recall that the gradient of the loss with respect to an early hidden state $h_t$ depends on:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Each term in the product is the Jacobian:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(\sigma'(z_k)) \cdot W_{hh}$$

Where $\sigma'$ is the derivative of the activation function (e.g., $\tanh' = 1 - \tanh^2$) and $z_k$ is the pre-activation.

### Gradient Magnitude Analysis

The norm of the gradient product satisfies:

$$\left\| \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \right\| \leq \prod_{k=t}^{T-1} \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$$

Let $\gamma = \max_k \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$. Then:

$$\left\| \frac{\partial \mathcal{L}}{\partial h_t} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial h_T} \right\| \cdot \gamma^{T-t}$$

**Three regimes emerge**:
- $\gamma < 1$: Gradients vanish exponentially ($\gamma^{T-t} \to 0$)
- $\gamma > 1$: Gradients explode exponentially ($\gamma^{T-t} \to \infty$)
- $\gamma = 1$: Gradients remain stable (ideal but difficult to achieve)

### Eigenvalue Perspective

The recurrent weight matrix $W_{hh}$ determines gradient behavior. If $W_{hh}$ has singular value decomposition:

$$W_{hh} = U \Sigma V^T$$

Then repeated multiplication by $W_{hh}$ amplifies components along eigenvectors with eigenvalues $|\lambda| > 1$ and suppresses components with $|\lambda| < 1$.

For a typical randomly initialized matrix:
- Most eigenvalues satisfy $|\lambda| < 1$
- Gradients decay as $|\lambda|^T$ through time
- Information from early timesteps becomes inaccessible

## Vanishing Gradients in Detail

### Why Gradients Vanish

**Activation Function Saturation**: The $\tanh$ function has derivative:

$$\tanh'(x) = 1 - \tanh^2(x)$$

This derivative is at most 1 (at $x=0$) and approaches 0 as $|x|$ increases. When neurons saturate (large activations), gradients shrink dramatically.

**Repeated Multiplication**: Even with $\tanh'(x) \approx 1$, if $\|W_{hh}\| < 1$, gradients decay:

$$\|W_{hh}^{T-t}\| \to 0 \quad \text{as } T-t \to \infty$$

### Consequences

1. **Cannot Learn Long-Range Dependencies**: The model cannot associate early inputs with later outputs
2. **Early Layers Don't Update**: Weights for processing early timesteps receive near-zero gradients
3. **Effective Memory is Short**: The model behaves as if it only sees recent inputs

### Visualization

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def demonstrate_vanishing_gradients():
    """Show gradient decay through timesteps."""
    
    # Simple RNN
    hidden_size = 100
    seq_length = 50
    
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    
    # Initialize with small weights (promotes vanishing)
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, -0.1, 0.1)
    
    # Input sequence
    x = torch.randn(1, seq_length, hidden_size, requires_grad=True)
    
    # Forward pass
    outputs, h_n = rnn(x)
    
    # Compute gradients for each timestep
    gradient_norms = []
    for t in range(seq_length):
        if x.grad is not None:
            x.grad.zero_()
        outputs[0, t].sum().backward(retain_graph=True)
        gradient_norms.append(x.grad[0, 0].norm().item())
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_length), gradient_norms[::-1])
    plt.xlabel('Distance from Loss (timesteps)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Vanishing in Simple RNN')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

demonstrate_vanishing_gradients()
```

## Exploding Gradients in Detail

### Why Gradients Explode

When $\|W_{hh}\| > 1$ or when eigenvalues exceed 1, gradients grow exponentially:

$$\|W_{hh}^{T-t}\| \to \infty \quad \text{as } T-t \to \infty$$

### Consequences

1. **Numerical Overflow**: Gradient values exceed floating-point range (NaN)
2. **Unstable Updates**: Large gradients cause parameters to oscillate wildly
3. **Training Divergence**: Loss increases instead of decreasing

### Detection

Exploding gradients are easier to detect than vanishing:

```python
def check_gradients(model):
    """Monitor gradient health during training."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > 100:
        print(f"Warning: Large gradient norm {total_norm:.2f}")
    if torch.isnan(torch.tensor(total_norm)):
        print("Error: NaN gradients detected!")
    
    return total_norm
```

## Solutions and Mitigations

### 1. Gradient Clipping

The most direct solution for exploding gradients:

```python
import torch.nn.utils as utils

# During training
loss.backward()

# Clip gradients before optimizer step
max_norm = 1.0
utils.clip_grad_norm_(model.parameters(), max_norm)

optimizer.step()
```

**Clipping strategies**:

```python
# Clip by global norm (recommended)
utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Manual implementation
def clip_gradients(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2) ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

### 2. Weight Initialization

Careful initialization keeps gradients in stable range:

```python
def init_rnn_weights(rnn):
    """Initialize RNN weights for stable gradients."""
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            # Input-to-hidden: Xavier initialization
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-to-hidden: Orthogonal initialization
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Biases: Zero or small constant
            nn.init.zeros_(param)
```

**Orthogonal initialization** for $W_{hh}$ ensures eigenvalues have magnitude 1, preserving gradient norm.

### 3. Use LSTM or GRU

Gated architectures fundamentally solve vanishing gradients through additive gradient paths:

```python
# LSTM provides stable gradient flow
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# GRU: Simpler alternative with similar benefits
gru = nn.GRU(input_size, hidden_size, batch_first=True)
```

The cell state in LSTM allows gradients to flow unchanged through time (see LSTM section).

### 4. Skip Connections

Residual connections provide direct gradient paths:

```python
class ResidualRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        # Skip connection
        return rnn_out + self.projection(x), hidden
```

### 5. Batch Normalization Variants

Normalizing activations stabilizes training:

```python
class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h):
        combined = self.W_xh(x) + self.W_hh(h)
        normalized = self.layer_norm(combined)
        return torch.tanh(normalized)
```

## Diagnosing Gradient Problems

```python
class GradientMonitor:
    """Track gradient statistics during training."""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
    
    def log_gradients(self):
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item(),
                    'norm': param.grad.norm().item()
                }
        self.gradient_history.append(grad_stats)
    
    def plot_gradient_norms(self, layer_name):
        norms = [h[layer_name]['norm'] for h in self.gradient_history]
        plt.plot(norms)
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title(f'Gradient Norm: {layer_name}')
        plt.show()
```

## Practical Training Guidelines

| Issue | Symptom | Solution |
|-------|---------|----------|
| Vanishing gradients | Loss plateaus early; poor long-sequence performance | Use LSTM/GRU; skip connections |
| Exploding gradients | NaN loss; unstable training | Gradient clipping; lower learning rate |
| Mixed issues | Inconsistent training | Orthogonal initialization + clipping + gated units |

## Summary

The vanishing and exploding gradient problems arise from:
1. Repeated multiplication through the recurrence
2. Activation function derivatives
3. Weight matrix eigenvalue structure

**Vanishing gradients** prevent learning long-range dependencies; **exploding gradients** destabilize training entirely.

Solutions include:
- Gradient clipping (primarily for exploding)
- Careful initialization
- Gated architectures (LSTM, GRU)
- Skip connections
- Normalization

Understanding these issues motivates the sophisticated gating mechanisms in LSTM and GRU, which we explore in the next sections.
