# Vanishing Gradients

## The Core Problem

During backpropagation through time, gradients must flow backward through many timesteps. The repeated multiplication through the recurrence relation causes gradients to shrink exponentially, making it impossible for the network to learn dependencies between distant elements in a sequence. This is the **vanishing gradient problem**—the primary limitation of vanilla RNNs.

## Mathematical Analysis

### Gradient Flow Through Time

The gradient of the loss with respect to an early hidden state $h_t$ involves a product of Jacobians:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Each Jacobian term is:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(\sigma'(z_k)) \cdot W_{hh}$$

where $\sigma'$ is the derivative of the activation function and $z_k$ is the pre-activation value.

### Bounding the Gradient Norm

The norm of the Jacobian product satisfies:

$$\left\| \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \right\| \leq \prod_{k=t}^{T-1} \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$$

Let $\gamma = \max_k \left\| \frac{\partial h_{k+1}}{\partial h_k} \right\|$. When $\gamma < 1$:

$$\left\| \frac{\partial \mathcal{L}}{\partial h_t} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial h_T} \right\| \cdot \gamma^{T-t} \xrightarrow{T-t \to \infty} 0$$

The gradient decays exponentially with the temporal distance $T - t$.

### Why $\gamma < 1$ Is the Common Case

Two factors conspire to make $\gamma < 1$ the default behavior:

**Activation function saturation.** The $\tanh$ derivative is:

$$\tanh'(x) = 1 - \tanh^2(x)$$

This derivative achieves its maximum of 1 only at $x = 0$ and approaches 0 as $|x|$ increases. Whenever hidden state activations are not centered near zero—which becomes increasingly common as networks train—the $\text{diag}(\sigma'(z_k))$ factor shrinks the gradient.

**Recurrent weight spectral norm.** Even when $\tanh'(x) \approx 1$, if $\|W_{hh}\| < 1$ (i.e., the largest singular value of $W_{hh}$ is less than 1), repeated multiplication drives gradients to zero:

$$\|W_{hh}^{T-t}\| \to 0 \quad \text{as } T-t \to \infty$$

### Eigenvalue Perspective

The eigendecomposition of $W_{hh}$ provides deeper insight. If $W_{hh} = U \Lambda U^{-1}$, then $W_{hh}^n = U \Lambda^n U^{-1}$. Components of the gradient along eigenvectors with $|\lambda_i| < 1$ decay as $|\lambda_i|^n$, while components with $|\lambda_i| > 1$ grow. For typical random initialization, most eigenvalues satisfy $|\lambda_i| < 1$, so gradient information from early timesteps is systematically destroyed.

## Consequences

**Cannot learn long-range dependencies.** The model cannot associate early inputs with later outputs. If a word at position 5 determines the correct label at position 50, the gradient signal arriving at position 5 is negligibly small.

**Early timesteps receive near-zero updates.** Weights responsible for processing early parts of the sequence are effectively frozen—they receive gradients too small to produce meaningful parameter changes.

**Effective memory is short.** Despite theoretically having access to the entire history, the model in practice behaves as if it only sees the most recent 10–20 timesteps.

## Empirical Demonstration

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def demonstrate_vanishing_gradients():
    """Show gradient decay through timesteps."""
    hidden_size = 100
    seq_length = 50
    
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
    
    # Small weight initialization promotes vanishing
    for name, param in rnn.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param, -0.1, 0.1)
    
    x = torch.randn(1, seq_length, hidden_size, requires_grad=True)
    outputs, h_n = rnn(x)
    
    # Measure gradient from final output to each input position
    gradient_norms = []
    for t in range(seq_length):
        if x.grad is not None:
            x.grad.zero_()
        outputs[0, t].sum().backward(retain_graph=True)
        gradient_norms.append(x.grad[0, 0].norm().item())
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(seq_length), gradient_norms[::-1])
    plt.xlabel('Distance from Output (timesteps)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Vanishing in Vanilla RNN')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_vanishing_gradients()
```

The log-scale plot reveals exponential decay: gradients at timestep 0 can be $10^{6}$ to $10^{10}$ times smaller than at the final timestep.

## Diagnosis

### Gradient Norm Monitoring

```python
class VanishingGradientDetector:
    """Track gradient norms across layers during training."""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
    
    def log_gradients(self):
        grad_stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_stats[name] = {
                    'norm': param.grad.norm().item(),
                    'mean_abs': param.grad.abs().mean().item(),
                    'max_abs': param.grad.abs().max().item()
                }
        self.gradient_history.append(grad_stats)
    
    def check_vanishing(self, threshold=1e-7):
        """Check if recurrent weight gradients are vanishing."""
        if not self.gradient_history:
            return False
        latest = self.gradient_history[-1]
        for name, stats in latest.items():
            if 'weight_hh' in name and stats['norm'] < threshold:
                print(f"⚠️  Vanishing gradient: {name} norm = {stats['norm']:.2e}")
                return True
        return False
```

### Symptoms

| Indicator | What to Check |
|-----------|---------------|
| Loss plateaus early | Training loss stops decreasing after initial drop |
| Poor long-sequence accuracy | Model performs well on short sequences but fails on long ones |
| Near-zero `weight_hh` gradients | Gradient norms for recurrent weights are orders of magnitude smaller than input weights |
| Hidden states collapse | Final hidden states become similar regardless of early input content |

## Mitigations

### Weight Initialization

Orthogonal initialization for $W_{hh}$ ensures all singular values equal 1 at initialization, providing the best starting point for gradient flow:

```python
def init_for_gradient_preservation(rnn):
    """Initialize RNN weights to mitigate vanishing gradients."""
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

Orthogonal matrices satisfy $\|Qx\| = \|x\|$ for all $x$, so repeated multiplication preserves gradient norms—at least at initialization before training changes the weights.

### Skip Connections

Residual connections provide a direct gradient path that bypasses the recurrence:

```python
class ResidualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, x, h):
        h_new = self.rnn_cell(x, h)
        return h_new + self.projection(x)  # Residual path
```

The skip connection ensures that gradients can flow directly from output to input without passing through the entire chain of $W_{hh}$ multiplications.

### Layer Normalization

Normalizing hidden state activations prevents saturation and keeps values in the region where $\tanh'$ is large:

```python
class LayerNormRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, h):
        combined = self.W_xh(x) + self.W_hh(h)
        normalized = self.layer_norm(combined)
        return torch.tanh(normalized)
```

### Gated Architectures (The Fundamental Solution)

LSTM and GRU architectures solve the vanishing gradient problem through additive update paths. The LSTM cell state update:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

creates a gradient highway: $\frac{\partial c_t}{\partial c_{t-1}} = f_t$, which is a simple element-wise multiplication by values in $[0, 1]$. When $f_t \approx 1$ (forget gate open), gradients flow unchanged—no repeated multiplication by $W_{hh}$.

```python
# Drop-in replacement: LSTM provides stable gradient flow
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

# GRU: simpler alternative with similar gradient benefits
gru = nn.GRU(input_size, hidden_size, batch_first=True)
```

## Summary

The vanishing gradient problem arises from repeated multiplication through the recurrence: each timestep multiplies the gradient by $\text{diag}(\tanh') \cdot W_{hh}$, and when this product has spectral norm less than 1, gradients decay exponentially with temporal distance. This prevents learning long-range dependencies, making vanilla RNNs effectively short-memory models. While initialization, skip connections, and normalization provide partial relief, gated architectures (LSTM, GRU) offer the fundamental solution through additive gradient paths.
