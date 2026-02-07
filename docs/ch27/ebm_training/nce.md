# Noise Contrastive Estimation

## Learning Objectives

After completing this section, you will be able to:

1. Reformulate density estimation as a binary classification problem between data and noise
2. Derive the NCE objective and show how it estimates the partition function as a byproduct
3. Implement NCE training for energy-based models
4. Understand the role of the noise distribution and its impact on estimation quality

## Introduction

Noise Contrastive Estimation (NCE), introduced by Gutmann and Hyvärinen (2010), takes a radically different approach to the partition function problem. Rather than approximating the intractable normalization constant (as in CD) or bypassing it through score functions (as in score matching), NCE treats $\log Z$ as a learnable parameter and estimates it by training a classifier to distinguish data from noise.

The key insight is that density estimation can be reduced to classification: if we can distinguish real data from artificial noise, we implicitly know the density ratio between the two distributions—and this ratio encodes the model density up to the noise density, which is known.

## The NCE Framework

### Setup

Given a data distribution $p_{\text{data}}(x)$ and a known noise distribution $p_n(x)$, NCE creates a binary classification problem:

- Draw $x$ from data with label $y=1$ (with probability $\nu/(1+\nu)$)
- Draw $x$ from noise with label $y=0$ (with probability $1/(1+\nu)$)

where $\nu$ is the ratio of noise samples to data samples.

### The Classifier

The optimal classifier for this binary task uses the log-density ratio:

$$\log \frac{P(y=1|x)}{P(y=0|x)} = \log p_{\text{data}}(x) - \log p_n(x) + \log \nu$$

NCE parameterizes the model's log-density as:

$$\log p_\theta(x) = -E_\theta(x) + c$$

where $c$ is a learnable scalar that implicitly estimates $-\log Z(\theta)$.

### NCE Objective

The NCE loss function is the binary cross-entropy of the induced classifier:

$$J_{\text{NCE}}(\theta, c) = -\mathbb{E}_{p_{\text{data}}}[\log h(x; \theta, c)] - \nu \cdot \mathbb{E}_{p_n}[\log(1 - h(x; \theta, c))]$$

where:

$$h(x; \theta, c) = \sigma\left(-E_\theta(x) + c - \log p_n(x) + \log \nu\right)$$

and $\sigma$ is the sigmoid function.

### Consistency

Gutmann and Hyvärinen (2010) proved that as the number of noise samples approaches infinity, the NCE estimator is consistent: $\hat{\theta} \to \theta^*$ and $\hat{c} \to -\log Z(\theta^*)$. The partition function is estimated as a byproduct of training.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class EnergyNetNCE(nn.Module):
    """Energy network with learnable log-partition function for NCE."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable log-partition function estimate
        self.log_Z = nn.Parameter(torch.tensor(0.0))
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x)."""
        return self.energy_net(x).squeeze(-1)
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) = -E(x) - log Z."""
        return -self.energy(x) - self.log_Z


def nce_loss(model, x_data, x_noise, log_pn_data, log_pn_noise, nu=1.0):
    """
    Compute NCE loss.
    
    Parameters
    ----------
    model : EnergyNetNCE
        Energy model with learnable log Z
    x_data : torch.Tensor
        Samples from data distribution
    x_noise : torch.Tensor
        Samples from noise distribution
    log_pn_data : torch.Tensor
        Log noise density at data points
    log_pn_noise : torch.Tensor  
        Log noise density at noise points
    nu : float
        Noise-to-data ratio
    
    Returns
    -------
    torch.Tensor
        NCE loss
    """
    # Log-density ratio for data points
    log_ratio_data = model.log_prob(x_data) - log_pn_data + np.log(nu)
    
    # Log-density ratio for noise points
    log_ratio_noise = model.log_prob(x_noise) - log_pn_noise + np.log(nu)
    
    # Binary cross-entropy
    loss_data = -F.logsigmoid(log_ratio_data).mean()
    loss_noise = -nu * F.logsigmoid(-log_ratio_noise).mean()
    
    return loss_data + loss_noise


def train_ebm_with_nce():
    """
    Train a 2D energy model using NCE.
    """
    # Generate data: mixture of 3 Gaussians
    n_data = 2000
    data = torch.cat([
        torch.randn(n_data // 3, 2) * 0.3 + torch.tensor([-2.0, 0.0]),
        torch.randn(n_data // 3, 2) * 0.3 + torch.tensor([2.0, 0.0]),
        torch.randn(n_data // 3, 2) * 0.3 + torch.tensor([0.0, 2.0]),
    ])
    
    # Noise distribution: Gaussian with wider spread
    noise_mean = data.mean(0)
    noise_std = data.std() * 2
    
    def sample_noise(n):
        return torch.randn(n, 2) * noise_std + noise_mean
    
    def log_noise_density(x):
        diff = (x - noise_mean) / noise_std
        return -0.5 * (diff ** 2).sum(dim=1) - np.log(2 * np.pi) - 2 * np.log(noise_std)
    
    # Model
    model = EnergyNetNCE(input_dim=2, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    nu = 5.0  # 5 noise samples per data sample
    n_epochs = 300
    losses = []
    log_Z_history = []
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(data))
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(data), 128):
            x_data = data[perm[i:i+128]]
            batch_size = x_data.shape[0]
            
            # Sample noise
            x_noise = sample_noise(int(batch_size * nu))
            
            # Compute noise log-densities
            log_pn_d = log_noise_density(x_data)
            log_pn_n = log_noise_density(x_noise)
            
            # NCE loss
            loss = nce_loss(model, x_data, x_noise, log_pn_d, log_pn_n, nu)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        losses.append(epoch_loss / n_batches)
        log_Z_history.append(model.log_Z.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: loss = {losses[-1]:.4f}, "
                  f"log Z = {model.log_Z.item():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Energy landscape
    grid_x = torch.linspace(-4, 4, 100)
    grid_y = torch.linspace(-2, 4, 100)
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        energies = model.energy(grid).reshape(100, 100)
    
    axes[0].contourf(X.numpy(), Y.numpy(), energies.numpy(), 
                     levels=30, cmap='viridis')
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(),
                   s=1, alpha=0.1, c='white')
    axes[0].set_title('Learned Energy Landscape')
    
    # Training loss
    axes[1].plot(losses, linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('NCE Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Log Z convergence
    axes[2].plot(log_Z_history, linewidth=2, color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Estimated log Z')
    axes[2].set_title('Partition Function Estimate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model

model = train_ebm_with_nce()
```

## Choosing the Noise Distribution

The choice of noise distribution $p_n$ critically affects NCE performance:

**Requirements**: $p_n$ must have support wherever $p_{\text{data}}$ has support. Otherwise, the classifier can trivially distinguish data from noise based on regions where $p_n = 0$, learning nothing about the data density in those regions.

**Practical guidelines**:

- $p_n$ should roughly match the spread of the data (too narrow: poor coverage; too wide: wasted noise samples)
- Gaussian noise centered on the data mean with standard deviation 2–3× the data spread is a reasonable default
- More noise samples ($\nu > 1$) generally improve estimation quality at the cost of computation

**Optimal noise**: In theory, the optimal noise distribution is $p_n = p_{\text{data}}$ itself—but this is circular since we don't know the data distribution. In practice, a reasonable approximation to the data distribution (e.g., a Gaussian fitted to the data) works well.

## Comparison with Other Methods

| Property | CD | Score Matching | NCE |
|----------|-----|---------------|-----|
| Avoids $Z$ | Approximates | Completely | Estimates $Z$ |
| Needs MCMC | Yes | No | No |
| Discrete data | Yes | No | Yes |
| Likelihood estimate | No | No | Yes (via $\hat{Z}$) |
| Noise distribution | N/A | N/A | Required |
| Consistency | Biased | Consistent | Consistent |

NCE is unique in providing a direct estimate of the partition function, enabling approximate likelihood evaluation. This is particularly valuable for model comparison and evaluation.

## NCE for Word Embeddings

NCE gained widespread adoption in NLP through its use in word2vec (Mikolov et al., 2013), where it trains word embeddings by distinguishing true word-context pairs from randomly sampled "negative" pairs. The energy function is the negative dot product $E(w, c) = -\mathbf{w}^T \mathbf{c}$, and the noise distribution samples random words from the vocabulary. This application demonstrated that NCE scales to very large vocabularies (effectively very large state spaces) where the partition function would be prohibitively expensive.

## Key Takeaways

!!! success "Core Concepts"
    1. NCE reduces density estimation to binary classification: data vs. noise
    2. The partition function $\log Z$ is estimated as a learnable parameter, a byproduct of training
    3. NCE is consistent: with enough noise samples, it recovers the true parameters
    4. The noise distribution must cover the data support and should roughly match the data scale
    5. NCE is the only standard EBM training method that provides an estimate of the partition function

!!! info "Historical Impact"
    NCE's influence extends far beyond EBMs. The "negative sampling" technique in word2vec, contrastive learning in self-supervised representation learning (SimCLR, MoCo), and InfoNCE in mutual information estimation are all descendants of the NCE framework. The principle that useful representations emerge from distinguishing positive examples from negatives has become one of the most productive ideas in modern machine learning.

## Exercises

1. **Noise ratio**: Train NCE models with $\nu \in \{1, 5, 10, 50\}$ noise samples per data point. How does the quality of the learned energy function and the accuracy of the $\log Z$ estimate change?

2. **Noise distribution ablation**: Compare NCE performance with Gaussian noise of different variances. What happens when the noise is too concentrated or too diffuse?

3. **Ranking NCE**: Implement Ranking NCE (Ma & Collins, 2018), which replaces binary classification with ranking, and compare against standard NCE.

## References

- Gutmann, M. U., & Hyvärinen, A. (2010). Noise-Contrastive Estimation: A new estimation principle for unnormalized statistical models. *AISTATS*.
- Mnih, A., & Teh, Y. W. (2012). A Fast and Simple Algorithm for Training Neural Probabilistic Language Models. *ICML*.
- Mikolov, T., et al. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS*.
- Ma, Z., & Collins, M. (2018). Noise Contrastive Estimation and Negative Sampling for Conditional Models. *EMNLP*.
