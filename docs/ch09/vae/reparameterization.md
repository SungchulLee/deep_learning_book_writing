# Section 41.2: The Reparameterization Trick

Making stochastic sampling differentiable for backpropagation.

---

## Overview

**What you'll learn:**

- Why sampling blocks gradients
- The reparameterization trick
- Implementation details
- Numerical stability considerations

**Time:** ~30 minutes  
**Level:** Intermediate

---

## The Problem: Backpropagating Through Sampling

### Standard Autoencoder Flow

In a standard autoencoder, gradients flow smoothly:

$$x \xrightarrow{\nabla} h \xrightarrow{\nabla} z \xrightarrow{\nabla} h' \xrightarrow{\nabla} \hat{x} \xrightarrow{\nabla} \mathcal{L}$$

Every operation is deterministic and differentiable.

### VAE with Naive Sampling

In a VAE, we sample from the encoder's distribution:

$$x \xrightarrow{\nabla} h \xrightarrow{\nabla} (\mu, \sigma) \xrightarrow{\color{red}{\times}} z \sim \mathcal{N}(\mu, \sigma^2) \xrightarrow{\nabla} \hat{x} \xrightarrow{\nabla} \mathcal{L}$$

**Problem:** The sampling operation $z \sim \mathcal{N}(\mu, \sigma^2)$ is **stochastic** and **not differentiable**!

We cannot compute:

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot \underbrace{\frac{\partial z}{\partial \mu}}_{\text{undefined!}}$$

---

## The Solution: Reparameterization

### Key Insight

Instead of sampling $z$ directly from $\mathcal{N}(\mu, \sigma^2)$, we:

1. Sample noise from a **fixed** distribution: $\epsilon \sim \mathcal{N}(0, 1)$
2. Transform it deterministically: $z = \mu + \sigma \cdot \epsilon$

### Why This Works

The distribution of $z$ is still $\mathcal{N}(\mu, \sigma^2)$:

- If $\epsilon \sim \mathcal{N}(0, 1)$
- Then $\sigma \cdot \epsilon \sim \mathcal{N}(0, \sigma^2)$
- And $\mu + \sigma \cdot \epsilon \sim \mathcal{N}(\mu, \sigma^2)$

But now $z$ is a **deterministic function** of $\mu$, $\sigma$, and $\epsilon$!

### Gradient Flow with Reparameterization

$$x \xrightarrow{\nabla} h \xrightarrow{\nabla} (\mu, \sigma) \xrightarrow{\nabla} z = \mu + \sigma \cdot \epsilon \xrightarrow{\nabla} \hat{x} \xrightarrow{\nabla} \mathcal{L}$$

Now we can compute:

$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

---

## Visual Explanation

### Without Reparameterization

```
           Sampling
    μ ───────┐
             ├──→ z ~ N(μ, σ²) ──→ Decoder ──→ Loss
    σ ───────┘    ↑
              (stochastic, 
               no gradient)
```

### With Reparameterization

```
    μ ───────────────┐
                     │
                     ▼
    ε ~ N(0,1) ──→ z = μ + σ·ε ──→ Decoder ──→ Loss
                     ▲
                     │
    σ ───────────────┘
    
    (deterministic function of μ, σ, ε)
    (gradients flow through μ and σ)
```

---

## Mathematical Details

### Deriving the Gradients

Given $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$:

$$\frac{\partial z}{\partial \mu} = 1$$

$$\frac{\partial z}{\partial \sigma} = \epsilon$$

For the loss $\mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z}$$

$$\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \sigma} = \frac{\partial \mathcal{L}}{\partial z} \cdot \epsilon$$

### Monte Carlo Estimation

In practice, we estimate the expectation with a single sample:

$$\mathbb{E}_{q(z|x)}[\mathcal{L}] \approx \mathcal{L}(z), \quad z = \mu + \sigma \cdot \epsilon$$

This is unbiased and has reasonable variance for training.

---

## Implementation

### Using Standard Deviation

```python
def reparameterize(mu, std):
    """
    Reparameterization trick using standard deviation.
    
    Args:
        mu: Mean of the distribution [batch_size, latent_dim]
        std: Standard deviation [batch_size, latent_dim]
        
    Returns:
        z: Sampled latent vector [batch_size, latent_dim]
    """
    # Sample noise from standard normal
    eps = torch.randn_like(std)
    
    # Reparameterized sample
    z = mu + std * eps
    
    return z
```

### Using Log-Variance (Recommended)

```python
def reparameterize_logvar(mu, logvar):
    """
    Reparameterization trick using log-variance.
    
    Using log-variance is numerically more stable because:
    - σ = exp(0.5 * logvar) is always positive
    - Avoids issues with negative variance
    - Better gradient behavior for small variances
    
    Args:
        mu: Mean of the distribution [batch_size, latent_dim]
        logvar: Log-variance [batch_size, latent_dim]
        
    Returns:
        z: Sampled latent vector [batch_size, latent_dim]
    """
    # Convert log-variance to standard deviation
    # logvar = log(σ²) → σ = exp(0.5 * logvar)
    std = torch.exp(0.5 * logvar)
    
    # Sample noise from standard normal
    eps = torch.randn_like(std)
    
    # Reparameterized sample
    z = mu + std * eps
    
    return z
```

### Why Log-Variance?

| Approach | Pros | Cons |
|----------|------|------|
| **Variance (σ²)** | Direct interpretation | Can be negative (problematic) |
| **Std dev (σ)** | Simple formula | Requires positivity constraint |
| **Log-variance** | Always gives positive σ | Slightly more complex |

Log-variance is the standard choice because:
- $\sigma = \exp(0.5 \cdot \log\sigma^2)$ is **always positive**
- No need for softplus or other positivity constraints
- Numerically stable gradients

---

## Complete VAE Encoder Implementation

```python
class VAEEncoder(nn.Module):
    """
    VAE Encoder that outputs distribution parameters.
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Shared hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for mu and logvar
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        """
        Encode input to distribution parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            mu: Mean [batch_size, latent_dim]
            logvar: Log-variance [batch_size, latent_dim]
        """
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def sample(self, mu, logvar):
        """
        Sample z using reparameterization trick.
        
        Args:
            mu: Mean [batch_size, latent_dim]
            logvar: Log-variance [batch_size, latent_dim]
            
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
```

---

## Numerical Stability Considerations

### Issue 1: Exploding Standard Deviation

If logvar is very large, $\sigma = \exp(0.5 \cdot \text{logvar})$ explodes.

**Solution:** Clamp logvar

```python
def reparameterize_stable(mu, logvar, max_logvar=10):
    """Numerically stable reparameterization."""
    logvar = torch.clamp(logvar, max=-max_logvar, min=max_logvar)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps
```

### Issue 2: Zero Variance

If logvar is very negative, variance approaches zero.

**Implication:** Encoder becomes deterministic (like standard AE)

**Solution:** Use KL annealing or minimum variance threshold

```python
def reparameterize_min_std(mu, logvar, min_std=1e-4):
    """Reparameterization with minimum standard deviation."""
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=min_std)
    eps = torch.randn_like(std)
    return mu + std * eps
```

### Issue 3: Gradient Magnitude

Gradient w.r.t. σ is proportional to ε, which can be large.

**Solution:** Gradient clipping during training

```python
# In training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Training vs. Inference

### During Training

We **always** sample using the reparameterization trick:

```python
def forward_train(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)  # Sample with noise
    return self.decode(z), mu, logvar
```

### During Inference (Deterministic)

For reproducible results, we can use just the mean:

```python
def forward_inference(self, x):
    mu, logvar = self.encode(x)
    z = mu  # Use mean, no sampling
    return self.decode(z)
```

Or sample with fixed seed:

```python
def forward_inference_sample(self, x, seed=42):
    torch.manual_seed(seed)
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    return self.decode(z)
```

---

## Exercises

### Exercise 1: Verify Reparameterization

Empirically verify that $z = \mu + \sigma \cdot \epsilon$ has distribution $\mathcal{N}(\mu, \sigma^2)$:

```python
mu = torch.tensor([2.0])
sigma = torch.tensor([0.5])
samples = []

for _ in range(10000):
    eps = torch.randn_like(sigma)
    z = mu + sigma * eps
    samples.append(z.item())

# Compute sample mean and std
# Should be close to μ=2.0 and σ=0.5
```

### Exercise 2: Gradient Verification

Compute gradients analytically and verify with autograd:

```python
mu = torch.tensor([1.0], requires_grad=True)
logvar = torch.tensor([0.0], requires_grad=True)

# Fixed epsilon for reproducibility
torch.manual_seed(42)
eps = torch.randn_like(mu)

# Reparameterization
std = torch.exp(0.5 * logvar)
z = mu + std * eps

# Simple loss
loss = z.sum()
loss.backward()

print(f"∂L/∂μ = {mu.grad.item()}")  # Should be 1
print(f"∂L/∂logvar = {logvar.grad.item()}")  # Should be 0.5 * eps * std
```

### Exercise 3: Sampling Visualization

Create a visualization showing:
1. The encoder output (μ, σ) for several inputs
2. Multiple samples for each input
3. How samples cluster around the mean

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Problem** | Cannot backpropagate through stochastic sampling |
| **Solution** | Reparameterize: $z = \mu + \sigma \cdot \epsilon$ |
| **Benefit** | Gradients flow through $\mu$ and $\sigma$ |
| **Implementation** | Use log-variance for numerical stability |

### The Reparameterization Formula

$$z = \mu + \exp(0.5 \cdot \log\sigma^2) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## Next: VAE Architecture and Implementation

The next section covers the complete VAE architecture and implementation in PyTorch.
