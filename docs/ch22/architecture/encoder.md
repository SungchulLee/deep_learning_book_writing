# Encoder Network

Amortized variational inference: learning to map data to approximate posterior distributions.

---

## Learning Objectives

By the end of this section, you will be able to:

- Design encoder architectures that output distribution parameters
- Implement the dual-head architecture for $\mu$ and $\log\sigma^2$
- Understand amortized inference and its computational advantages
- Build both MLP and convolutional encoders for different data types

---

## The Encoder's Role

### From Inference to Amortization

In classical variational inference, we optimize separate variational parameters for each data point $x_i$. The VAE encoder performs **amortized inference**: a single neural network $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$ maps any input to its approximate posterior. This trades per-sample optimality for computational efficiency — inference requires only a single forward pass.

### What the Encoder Outputs

The encoder network takes input $x$ and produces two vectors:

- **Mean** $\mu_\phi(x) \in \mathbb{R}^d$: the center of the approximate posterior
- **Log-variance** $\log\sigma^2_\phi(x) \in \mathbb{R}^d$: the spread (in log-space for numerical stability)

```
Input x ──► [Shared Hidden Layers] ──┬──► fc_mu ──► μ ∈ ℝ^d
                                      │
                                      └──► fc_logvar ──► log(σ²) ∈ ℝ^d
```

---

## MLP Encoder

### Architecture

```python
import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    """
    Fully connected encoder for vector inputs (e.g., flattened MNIST).
    
    Architecture: input_dim → hidden_dim → hidden_dim → (μ, logvar)
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for distribution parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, input_dim]
        Returns:
            mu: Mean [batch_size, latent_dim]
            logvar: Log-variance [batch_size, latent_dim]
        """
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### Why Separate Heads?

The mean and log-variance require different output characteristics. The mean $\mu$ is unbounded and benefits from the full expressiveness of the hidden features. The log-variance $\log\sigma^2$ controls uncertainty and typically converges to a different scale. Sharing all layers except the final projection allows the network to learn a common feature representation while specializing the final mapping.

---

## Convolutional Encoder

### For Image Data

```python
class ConvEncoder(nn.Module):
    """
    Convolutional encoder for image inputs.
    
    Progressively downsamples spatial dimensions while 
    increasing channel depth, then projects to latent parameters.
    """
    
    def __init__(self, in_channels=1, hidden_channels=32, latent_dim=32):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            
            # 14x14 → 7x7
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels * 2),
            
            # 7x7 → 4x4
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels * 4),
        )
        
        # Flatten and project
        flat_dim = hidden_channels * 4 * 4 * 4  # depends on input size
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input image [batch_size, channels, H, W]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)  # flatten
        return self.fc_mu(h), self.fc_logvar(h)
```

---

## Log-Variance Parameterization

### Why Not Output $\sigma$ Directly?

The standard deviation $\sigma$ must be positive. If we output $\sigma$ directly, we need an activation function (like softplus) to enforce positivity. Instead, outputting $\log\sigma^2$ and computing $\sigma = \exp(0.5 \cdot \log\sigma^2)$ provides automatic positivity without any activation function.

| Parameterization | Positivity | Gradient Behavior | Standard Choice |
|-----------------|------------|-------------------|-----------------|
| $\sigma$ directly | Requires softplus/exp | Can have vanishing gradients | No |
| $\sigma^2$ | Requires softplus | Squared scale issues | No |
| $\log\sigma^2$ | Automatic via exp | Well-behaved | **Yes** |

### Numerical Stability

For very large or small log-variances, clamping prevents numerical overflow:

```python
def safe_reparameterize(mu, logvar, max_logvar=10.0):
    """Reparameterize with clamped log-variance."""
    logvar = torch.clamp(logvar, min=-max_logvar, max=max_logvar)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

---

## Amortization Gap

### What Is It?

The **amortization gap** is the difference between the ELBO achieved by the amortized encoder and the ELBO achievable by per-sample optimization:

$$\text{Amortization gap} = \mathcal{L}^*(\theta; x) - \mathcal{L}(\theta, \phi; x)$$

where $\mathcal{L}^*$ uses the optimal $q^*(z|x)$ for each $x$ individually.

### Why It Matters

The encoder must generalize across all data points, so it cannot perfectly approximate the posterior for every single input. This is the price of amortization — fast inference at the cost of per-sample optimality.

### Reducing the Gap

Strategies to reduce the amortization gap include using more expressive encoder architectures, applying normalizing flows to the encoder output ($q_\phi(z|x)$ becomes a flow-based distribution), and performing a few steps of gradient-based refinement at test time (semi-amortized inference).

---

## Design Guidelines

### Choosing Hidden Dimensions

The encoder should have sufficient capacity to learn the mapping from data to posterior parameters. As a rule of thumb, hidden dimensions should be at least 2–4× the latent dimension, and the number of hidden layers should scale with data complexity.

### Initialization

Standard initialization (Xavier/He) works well for most cases. Some practitioners initialize the log-variance head's bias to a small negative value (e.g., -1) to start with moderate variance rather than unit variance:

```python
# Initialize logvar head bias for moderate initial variance
nn.init.constant_(encoder.fc_logvar.bias, -1.0)
```

### Batch Normalization

Batch normalization in the encoder can help training stability but should **not** be applied after the $\mu$ and $\log\sigma^2$ heads, as it would interfere with the distribution parameterization.

---

## Summary

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Shared layers** | Extract features from input | FC or Conv layers with ReLU |
| **Mean head** | Output $\mu_\phi(x)$ | Linear projection, unbounded |
| **Log-var head** | Output $\log\sigma^2_\phi(x)$ | Linear projection, unconstrained |
| **Reparameterization** | Sample $z = \mu + \sigma \cdot \epsilon$ | Enables backpropagation |

---

## Exercises

### Exercise 1: Architecture Ablation

Compare encoder architectures with 1, 2, and 4 hidden layers on MNIST. Measure reconstruction quality and training speed.

### Exercise 2: Latent Dimension Sweep

Train encoders with latent dimensions $d \in \{2, 8, 16, 32, 64, 128\}$. Plot reconstruction error and KL divergence vs $d$.

### Exercise 3: Amortization Gap

For a trained VAE, compare the ELBO from the encoder against the ELBO after 100 steps of gradient-based optimization of $(μ, \log σ²)$ for individual samples. Report the gap.

---

## What's Next

The next section covers the [Decoder Network](decoder.md), which maps latent codes back to data space.
