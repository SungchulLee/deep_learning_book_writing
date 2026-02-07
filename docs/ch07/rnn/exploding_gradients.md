# Exploding Gradients

## The Problem

While vanishing gradients silently degrade learning, **exploding gradients** cause dramatic training failures. When the spectral norm of the Jacobian product exceeds 1 at each timestep, gradients grow exponentially during backpropagation through time:

$$\left\| \frac{\partial \mathcal{L}}{\partial h_t} \right\| \leq \left\| \frac{\partial \mathcal{L}}{\partial h_T} \right\| \cdot \gamma^{T-t} \xrightarrow{\gamma > 1} \infty$$

Unlike vanishing gradients, which produce subtle degradation, exploding gradients manifest as sudden, catastrophic training failures.

## Mathematical Mechanism

### When Gradients Explode

The Jacobian at each timestep is:

$$\frac{\partial h_{k+1}}{\partial h_k} = \text{diag}(\sigma'(z_k)) \cdot W_{hh}$$

For $\tanh$, $\sigma'(z_k) \in [0, 1]$, which acts as a damping factor. However, if $W_{hh}$ has eigenvalues with magnitude significantly greater than 1, the amplification from $W_{hh}$ overwhelms the damping from $\sigma'$:

$$\|W_{hh}\|_2 > \frac{1}{\max_k \|\sigma'(z_k)\|} \implies \text{gradient explosion possible}$$

In practice, even moderate spectral norms like $\|W_{hh}\|_2 = 1.5$ produce explosive growth: $1.5^{50} \approx 6.4 \times 10^8$ after 50 timesteps.

### Eigenvalue Perspective

If $W_{hh}$ has eigenvalue $\lambda$ with $|\lambda| > 1$, the component of the gradient along the corresponding eigenvector grows as $|\lambda|^{T-t}$. For a randomly initialized matrix with hidden size $H$, the largest eigenvalue scales approximately as $\sqrt{H}$ under standard initialization schemes—which means larger hidden sizes are more prone to explosion without careful scaling.

## Consequences

**Numerical overflow.** Gradient values exceed floating-point range, producing NaN or Inf values that propagate through all parameters.

**Unstable parameter updates.** Even before overflow, excessively large gradients cause parameter updates that overshoot wildly, moving parameters far from any reasonable solution.

**Training divergence.** Loss increases rather than decreasing, often oscillating violently before collapsing to NaN.

## Detection

Exploding gradients are easier to detect than vanishing gradients because their effects are immediate and dramatic:

```python
import torch
import torch.nn as nn


def check_gradient_health(model):
    """Monitor gradient norms during training."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    if torch.isnan(torch.tensor(total_norm)):
        print("ERROR: NaN gradients detected!")
    elif total_norm > 100:
        print(f"WARNING: Large gradient norm {total_norm:.2f}")
    
    return total_norm
```

### Symptoms

| Symptom | Description |
|---------|-------------|
| NaN loss | Loss becomes NaN, typically within a few training steps |
| Loss spikes | Sudden large increases in loss followed by instability |
| Parameter divergence | Weight values grow unbounded |
| Oscillating loss | Loss oscillates wildly instead of converging |

### Proactive Monitoring

```python
class ExplodingGradientDetector:
    """Track gradient statistics to detect explosion early."""
    
    def __init__(self, model, alert_threshold=100.0):
        self.model = model
        self.alert_threshold = alert_threshold
        self.norm_history = []
    
    def check(self):
        total_norm = 0
        max_param_grad = 0
        
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_param_grad = max(max_param_grad, param_norm)
        
        total_norm = total_norm ** 0.5
        self.norm_history.append(total_norm)
        
        if total_norm > self.alert_threshold:
            print(f"⚠️  Gradient explosion: total norm = {total_norm:.2e}")
            print(f"   Max single-param gradient norm = {max_param_grad:.2e}")
            return True
        
        # Check for rapid growth
        if len(self.norm_history) > 10:
            recent = self.norm_history[-10:]
            growth_rate = recent[-1] / (recent[0] + 1e-8)
            if growth_rate > 100:
                print(f"⚠️  Rapid gradient growth: {growth_rate:.1f}x over 10 steps")
                return True
        
        return False
```

## Solutions

### Gradient Clipping (Primary Defense)

The most direct and widely used solution scales gradients when their combined norm exceeds a threshold. This is covered in detail in the Gradient Clipping section, but the essential idea:

```python
# After loss.backward(), before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Weight Initialization

Careful initialization keeps eigenvalues of $W_{hh}$ near 1 at the start of training:

```python
def init_for_stability(rnn):
    """Initialize RNN weights to prevent explosion."""
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Orthogonal: all singular values = 1
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
```

Orthogonal initialization sets all singular values to exactly 1, placing $W_{hh}$ at the boundary between vanishing and exploding—the ideal starting point.

### Spectral Normalization

Constrain $W_{hh}$ to have spectral norm at most 1 throughout training:

```python
from torch.nn.utils import spectral_norm

class SpectralNormRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Linear(input_size, hidden_size)
        # Constrain spectral norm of recurrent weight
        self.W_hh = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
    
    def forward(self, x, h):
        return torch.tanh(self.W_xh(x) + self.W_hh(h))
```

This guarantees $\|W_{hh}\|_2 \leq 1$, preventing explosion but potentially limiting the model's expressiveness.

### Learning Rate Reduction

Lower learning rates reduce the impact of large gradients on parameter updates:

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

Even if $\|g_t\|$ is large, a small $\eta$ prevents catastrophic updates. However, this slows learning globally rather than targeting the problematic gradients specifically—gradient clipping is usually preferred.

### Truncated BPTT

Limiting the backpropagation horizon to $k$ timesteps caps the number of Jacobian multiplications, bounding gradient growth:

```python
# Detach hidden state every k steps
if hidden is not None:
    hidden = hidden.detach()
```

This prevents gradients from accumulating over more than $k$ multiplications, but at the cost of being unable to learn dependencies longer than $k$ steps.

## Relationship to Vanishing Gradients

Vanishing and exploding gradients are two sides of the same coin—both arise from the repeated Jacobian product $\prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$. The three regimes are:

| Spectral norm $\gamma$ | Behavior | Result |
|------------------------|----------|--------|
| $\gamma < 1$ | Gradients shrink exponentially | Vanishing (slow, silent) |
| $\gamma = 1$ | Gradients remain stable | Ideal but fragile |
| $\gamma > 1$ | Gradients grow exponentially | Exploding (fast, catastrophic) |

In practice, both problems can coexist: different components of the gradient (along different eigenvectors of $W_{hh}$) may vanish or explode simultaneously. Gradient clipping addresses the exploding components while gated architectures address the vanishing components.

## Practical Guidelines

| Situation | Recommended Action |
|-----------|-------------------|
| Training any RNN | Always use gradient clipping (`max_norm=1.0`) |
| NaN/Inf loss | Reduce learning rate, verify initialization, add clipping |
| Loss spikes | Lower clipping threshold, check data for outliers |
| Deep RNN (3+ layers) | More aggressive clipping (`max_norm=0.5`), use LSTM/GRU |
| Long sequences ($T > 100$) | Combine clipping with truncated BPTT |

## Summary

Exploding gradients cause catastrophic training failure through exponential gradient growth during BPTT. The primary defense is gradient clipping—scaling gradients when their norm exceeds a threshold—supplemented by orthogonal initialization, spectral normalization, and truncated BPTT. Unlike vanishing gradients, which require architectural changes (LSTM/GRU) for a fundamental solution, exploding gradients are effectively controlled by the relatively simple technique of gradient norm clipping.
