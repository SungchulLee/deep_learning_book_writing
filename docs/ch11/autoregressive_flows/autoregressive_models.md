# Autoregressive Models

## Introduction

Autoregressive models decompose a joint distribution into a product of conditional distributions, modeling each dimension sequentially given all previous dimensions. This factorization provides a principled approach to density estimation with tractable likelihoods and forms the foundation for powerful normalizing flow architectures.

## The Autoregressive Factorization

### Chain Rule of Probability

Any joint distribution can be factored using the chain rule:

$$p(\mathbf{x}) = p(x_1, x_2, \ldots, x_D) = \prod_{d=1}^{D} p(x_d | x_1, \ldots, x_{d-1})$$

This factorization is exact—no approximation is involved. The key insight is that we can model each conditional $p(x_d | x_{<d})$ with a neural network.

### Autoregressive Property

A model is **autoregressive** if each output depends only on previous outputs:

$$x_d = f(x_1, x_2, \ldots, x_{d-1}; \theta_d)$$

The ordering creates a directed acyclic structure where information flows in one direction.

## Density Estimation with Autoregressive Models

### Parametric Conditionals

Each conditional distribution is parameterized by a neural network. For continuous data with Gaussian conditionals:

$$p(x_d | x_{<d}) = \mathcal{N}(x_d | \mu_d(x_{<d}), \sigma_d^2(x_{<d}))$$

where $\mu_d$ and $\sigma_d$ are outputs of neural networks that take $x_{<d}$ as input.

### Log-Likelihood

The log-likelihood decomposes as a sum:

$$\log p(\mathbf{x}) = \sum_{d=1}^{D} \log p(x_d | x_{<d})$$

For Gaussian conditionals:

$$\log p(\mathbf{x}) = -\frac{1}{2} \sum_{d=1}^{D} \left[ \frac{(x_d - \mu_d)^2}{\sigma_d^2} + \log \sigma_d^2 + \log 2\pi \right]$$

### Implementation

```python
import torch
import torch.nn as nn

class AutoregressiveGaussian(nn.Module):
    """
    Simple autoregressive model with Gaussian conditionals.
    Each dimension has its own network for mu and sigma.
    """
    def __init__(self, dim, hidden_size=64):
        super().__init__()
        self.dim = dim
        
        # First dimension: unconditional
        self.mu_1 = nn.Parameter(torch.zeros(1))
        self.log_sigma_1 = nn.Parameter(torch.zeros(1))
        
        # Subsequent dimensions: conditional on previous
        self.networks = nn.ModuleList()
        for d in range(1, dim):
            net = nn.Sequential(
                nn.Linear(d, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2)  # mu and log_sigma
            )
            self.networks.append(net)
    
    def get_params(self, x):
        """Get mu and sigma for each dimension."""
        batch_size = x.shape[0]
        
        # First dimension (unconditional)
        mu = [self.mu_1.expand(batch_size)]
        log_sigma = [self.log_sigma_1.expand(batch_size)]
        
        # Subsequent dimensions
        for d in range(1, self.dim):
            context = x[:, :d]  # x_1, ..., x_{d-1}
            params = self.networks[d-1](context)
            mu.append(params[:, 0])
            log_sigma.append(params[:, 1])
        
        mu = torch.stack(mu, dim=1)
        log_sigma = torch.stack(log_sigma, dim=1)
        
        return mu, log_sigma
    
    def log_prob(self, x):
        """Compute log p(x)."""
        mu, log_sigma = self.get_params(x)
        sigma = torch.exp(log_sigma)
        
        # Gaussian log-likelihood
        log_prob = -0.5 * (
            ((x - mu) / sigma) ** 2 
            + 2 * log_sigma 
            + torch.log(torch.tensor(2 * torch.pi))
        )
        
        return log_prob.sum(dim=1)
    
    def sample(self, n_samples):
        """Generate samples (sequential)."""
        samples = torch.zeros(n_samples, self.dim)
        
        for d in range(self.dim):
            if d == 0:
                mu = self.mu_1
                sigma = torch.exp(self.log_sigma_1)
            else:
                context = samples[:, :d]
                params = self.networks[d-1](context)
                mu = params[:, 0]
                sigma = torch.exp(params[:, 1])
            
            # Sample from conditional
            samples[:, d] = mu + sigma * torch.randn(n_samples)
        
        return samples
```

## Connection to Normalizing Flows

### Autoregressive Transformations

Autoregressive models naturally define invertible transformations. Given base noise $\mathbf{z} \sim \mathcal{N}(0, I)$:

**Forward (sampling direction):**
$$x_d = \mu_d(x_{<d}) + \sigma_d(x_{<d}) \cdot z_d$$

**Inverse (encoding direction):**
$$z_d = \frac{x_d - \mu_d(x_{<d})}{\sigma_d(x_{<d})}$$

### Triangular Jacobian

The transformation has a **lower triangular** Jacobian:

$$\frac{\partial x_i}{\partial z_j} = \begin{cases} 
\sigma_i(x_{<i}) & \text{if } i = j \\
\text{depends on path} & \text{if } i > j \\
0 & \text{if } i < j
\end{cases}$$

Since $x_i$ depends only on $z_1, \ldots, z_i$, the Jacobian is lower triangular.

### Efficient Determinant

For triangular matrices, the determinant is the product of diagonal elements:

$$\det J = \prod_{d=1}^{D} \sigma_d(x_{<d})$$

$$\log |\det J| = \sum_{d=1}^{D} \log \sigma_d(x_{<d})$$

This is **O(D)** instead of **O(D³)** for general matrices.

## Historical Context

### NADE (2011)

Neural Autoregressive Distribution Estimator was an early neural autoregressive model:

$$p(x_d = 1 | x_{<d}) = \sigma(W_{d,:} \cdot \text{hidden}_d + b_d)$$

Key contribution: Shared hidden representations across dimensions.

### RNADE (2013)

Real-valued NADE extended to continuous data using mixture of Gaussians:

$$p(x_d | x_{<d}) = \sum_{k=1}^{K} \pi_{dk} \mathcal{N}(x_d | \mu_{dk}, \sigma_{dk}^2)$$

### PixelCNN (2016)

Applied autoregressive modeling to images using masked convolutions for efficient training.

## Computational Trade-offs

### Density Evaluation

Computing $\log p(\mathbf{x})$ requires:
- One forward pass through the model
- Access to all dimensions simultaneously
- **Parallelizable** across dimensions (with proper architecture)

### Sampling

Generating samples requires:
- Sequential computation: $x_1 \to x_2 \to \ldots \to x_D$
- Each dimension depends on all previous
- **Not parallelizable** - O(D) sequential steps

This asymmetry is fundamental and motivates different flow variants:

| Direction | Autoregressive | Inverse Autoregressive |
|-----------|---------------|----------------------|
| Density   | Parallel (fast) | Sequential (slow) |
| Sampling  | Sequential (slow) | Parallel (fast) |

## Beyond Gaussian Conditionals

### Mixture of Gaussians

More flexible conditionals using mixtures:

$$p(x_d | x_{<d}) = \sum_{k=1}^{K} \pi_k(x_{<d}) \mathcal{N}(x_d | \mu_k(x_{<d}), \sigma_k^2(x_{<d}))$$

### Logistic Mixtures

Used in PixelCNN++:

$$p(x_d | x_{<d}) = \sum_{k=1}^{K} \pi_k \cdot \text{Logistic}(x_d | \mu_k, s_k)$$

### Discretized Distributions

For discrete data (e.g., 8-bit images):

$$P(x_d = v | x_{<d}) = \text{CDF}(v + 0.5) - \text{CDF}(v - 0.5)$$

## Dimension Ordering

### Importance of Order

The choice of dimension ordering affects model quality:

$$p(x_1)p(x_2|x_1)p(x_3|x_1,x_2) \neq p(x_3)p(x_2|x_3)p(x_1|x_2,x_3)$$

While both factorizations are valid, the model's ability to capture dependencies differs.

### Strategies

1. **Natural order**: Use domain-specific ordering (raster scan for images)
2. **Learned order**: Learn optimal ordering during training
3. **Random order**: Train with random orderings for robustness
4. **Multiple orders**: Ensemble models with different orderings

```python
def random_order_training(model, x, n_orders=4):
    """Train with multiple random orderings."""
    total_loss = 0
    
    for _ in range(n_orders):
        # Random permutation
        perm = torch.randperm(x.shape[1])
        x_perm = x[:, perm]
        
        # Compute loss with this ordering
        loss = -model.log_prob(x_perm).mean()
        total_loss += loss
    
    return total_loss / n_orders
```

## Efficient Architectures

### Weight Sharing

Instead of separate networks per dimension, share parameters:

$$[\mu_d, \log\sigma_d] = f_\theta(x_{<d}, d)$$

The network takes dimension index as additional input.

### MADE (Masked Autoencoder)

Use masking to compute all conditionals in one forward pass:

```python
class SharedAutoregressive(nn.Module):
    """Autoregressive with weight sharing via masking."""
    def __init__(self, dim, hidden_size=256):
        super().__init__()
        self.dim = dim
        
        # Single network with masks
        self.hidden = nn.Linear(dim, hidden_size)
        self.output = nn.Linear(hidden_size, dim * 2)  # mu and log_sigma
        
        # Create masks
        self.register_buffer('mask_hidden', self._create_mask(dim, hidden_size))
        self.register_buffer('mask_output', self._create_mask(hidden_size, dim * 2))
    
    def _create_mask(self, in_dim, out_dim):
        # Mask ensures autoregressive property
        # Implementation depends on specific architecture
        pass
    
    def forward(self, x):
        h = torch.relu(self.hidden(x) * self.mask_hidden)
        out = self.output(h) * self.mask_output
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma
```

## Connections to Other Models

### RNNs and Language Models

Language models are autoregressive:

$$p(\text{sentence}) = \prod_{t=1}^{T} p(\text{word}_t | \text{word}_{<t})$$

RNNs compress history into hidden state; autoregressive flows use explicit conditioning.

### Transformers

GPT-style transformers are autoregressive with masked self-attention ensuring each position only attends to previous positions.

### Diffusion Models

Diffusion models can be viewed as continuous-time autoregressive models over noise levels rather than dimensions.

## Summary

Autoregressive models:

1. **Exact factorization** via chain rule of probability
2. **Tractable likelihood** through sum of conditional log-probs
3. **Triangular Jacobian** enables O(D) determinant computation
4. **Asymmetric computation**: Fast density, slow sampling (or vice versa)
5. **Foundation for flows**: MAF and IAF build on autoregressive structure

The autoregressive principle—modeling distributions sequentially—underlies many successful generative models and provides the mathematical foundation for efficient normalizing flows.

## References

1. Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. *AISTATS*.
2. Uria, B., et al. (2013). RNADE: The Real-valued Neural Autoregressive Density-estimator. *NIPS*.
3. van den Oord, A., et al. (2016). Pixel Recurrent Neural Networks. *ICML*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
