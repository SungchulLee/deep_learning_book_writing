# Contrastive Divergence

## Learning Objectives

After completing this section, you will be able to:

1. Derive the maximum likelihood gradient for EBMs and identify its intractable component
2. Explain the CD-k approximation and why starting MCMC from data reduces bias
3. Implement CD-k training for RBMs with different values of $k$
4. Analyze the bias-variance trade-off in CD approximations

## Introduction

Contrastive Divergence (CD), introduced by Geoffrey Hinton in 2002, revolutionized the training of energy-based models by providing a practical approximation to the intractable maximum likelihood gradient. This algorithm made RBMs trainable on real datasets and helped spark the deep learning renaissance of the late 2000s. Understanding CD is essential because it established the template—approximate the negative phase with short-run MCMC—that modern EBM training methods continue to follow.

## Maximum Likelihood for EBMs

### The Log-Likelihood Gradient

For an EBM with energy $E_\theta(\mathbf{x})$, the log-likelihood of data $\mathbf{x}$ is:

$$\log P_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$$

where $Z(\theta) = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ is the partition function.

The gradient with respect to model parameters $\theta$ is:

$$\nabla_\theta \log P_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) + \mathbb{E}_{P_\theta}[\nabla_\theta E_\theta(\mathbf{x})]$$

This gradient has two terms with intuitive interpretations:

1. **Positive phase** (data term): $-\nabla_\theta E_\theta(\mathbf{x})$ — decrease energy on observed data
2. **Negative phase** (model term): $+\mathbb{E}_{P_\theta}[\nabla_\theta E_\theta]$ — increase energy on model samples

Training pushes data into energy valleys while pushing everything else out—a contrastive process that shapes the energy landscape to match the data distribution.

### The Intractability Problem

The negative phase requires computing:

$$\mathbb{E}_{P_\theta}[\nabla_\theta E_\theta(\mathbf{x})] = \int P_\theta(\mathbf{x}) \nabla_\theta E_\theta(\mathbf{x})\, d\mathbf{x}$$

This expectation is intractable because:

1. $P_\theta(\mathbf{x})$ requires computing the partition function $Z(\theta)$
2. The integral is over a potentially high-dimensional space
3. Monte Carlo estimation requires samples from $P_\theta$, which itself requires MCMC

Even though we cannot evaluate $P_\theta$ directly, we can generate approximate samples from it using MCMC methods like Gibbs sampling. The question becomes: how many MCMC steps are sufficient?

## The Contrastive Divergence Approximation

### Key Idea

Instead of running MCMC to equilibrium (which may require thousands of steps), CD starts the chain from the training data and runs only $k$ steps:

$$\text{CD-}k: \quad \mathbf{x}^{(0)} = \mathbf{x}_{\text{data}} \xrightarrow{\text{k Gibbs steps}} \mathbf{x}^{(k)}$$

### Gradient Approximation

The CD-k gradient approximation is:

$$\nabla_\theta \log P \approx -\nabla_\theta E_\theta(\mathbf{x}^{(0)}) + \nabla_\theta E_\theta(\mathbf{x}^{(k)})$$

For RBMs with weights $W$, visible biases $a$, and hidden biases $b$:

$$\Delta W \propto \langle \mathbf{v} \mathbf{h}^T \rangle_{\text{data}} - \langle \mathbf{v} \mathbf{h}^T \rangle_{k}$$

$$\Delta \mathbf{a} \propto \langle \mathbf{v} \rangle_{\text{data}} - \langle \mathbf{v} \rangle_{k}$$

$$\Delta \mathbf{b} \propto \langle \mathbf{h} \rangle_{\text{data}} - \langle \mathbf{h} \rangle_{k}$$

### Why CD Works

1. **Fast mixing near data**: Starting from data places the chain in a high-probability region, so it is already close to equilibrium
2. **Small parameter changes**: With small learning rates, the model distribution changes slowly between updates, so the previous sample serves as a good initialization
3. **Low variance**: Few MCMC steps produce low-variance gradient estimates, even though the estimates are biased

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

class RBMWithCDVariants(nn.Module):
    """
    RBM with multiple CD training variants for comparison.
    """
    
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))
        
        # For PCD: persistent chains
        self.register_buffer('persistent_chains', None)
    
    def sample_hidden(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """P(h=1|v) = σ(Wv + b)"""
        prob = torch.sigmoid(F.linear(v, self.W, self.b))
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def sample_visible(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """P(v=1|h) = σ(W^T h + a)"""
        prob = torch.sigmoid(F.linear(h, self.W.t(), self.a))
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def gibbs_step(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """One full Gibbs step: v → h → v'"""
        _, h = self.sample_hidden(v)
        _, v_new = self.sample_visible(h)
        return v_new, h
    
    def cd_k(self, v_data: torch.Tensor, k: int = 1, lr: float = 0.01) -> dict:
        """
        Contrastive Divergence with k Gibbs steps.
        
        Starts chain from data, runs k steps.
        """
        batch_size = v_data.shape[0]
        
        # Positive phase (from data)
        prob_h_data, _ = self.sample_hidden(v_data)
        
        # Negative phase (k Gibbs steps from data)
        v_model = v_data
        for _ in range(k):
            v_model, h_model = self.gibbs_step(v_model)
        
        # Compute gradients
        pos_grad_W = torch.matmul(prob_h_data.t(), v_data) / batch_size
        neg_grad_W = torch.matmul(h_model.t(), v_model) / batch_size
        
        # Update parameters
        self.W.data += lr * (pos_grad_W - neg_grad_W)
        self.a.data += lr * (v_data - v_model).mean(dim=0)
        self.b.data += lr * (prob_h_data - h_model).mean(dim=0)
        
        # Metrics
        recon_error = ((v_data - v_model) ** 2).mean().item()
        
        return {
            'recon_error': recon_error,
            'v_model': v_model,
            'h_model': h_model
        }


def compare_cd_k_values():
    """
    Compare CD-1, CD-10, and demonstrate the bias-accuracy trade-off.
    """
    print("=" * 70)
    print("COMPARING CD-k VALUES")
    print("=" * 70)
    
    # Create synthetic data (mixture of patterns)
    n_visible = 100
    n_hidden = 50
    n_samples = 1000
    
    pattern1 = torch.zeros(n_visible)
    pattern1[:50] = 1
    pattern2 = torch.zeros(n_visible)
    pattern2[50:] = 1
    
    data = torch.zeros(n_samples, n_visible)
    for i in range(n_samples):
        base = pattern1 if i % 2 == 0 else pattern2
        noise = torch.bernoulli(torch.ones(n_visible) * 0.1)
        data[i] = (base + noise) % 2
    
    # Train with different k values
    k_values = {'CD-1': 1, 'CD-5': 5, 'CD-10': 10}
    results = {name: {'errors': []} for name in k_values}
    
    n_epochs = 50
    batch_size = 64
    
    for name, k in k_values.items():
        print(f"\nTraining {name}...")
        rbm = RBMWithCDVariants(n_visible, n_hidden)
        
        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_error = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch = data[perm[i:i+batch_size]]
                result = rbm.cd_k(batch, k=k, lr=0.01)
                epoch_error += result['recon_error']
                n_batches += 1
            
            results[name]['errors'].append(epoch_error / n_batches)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: error = {results[name]['errors'][-1]:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for name, data_dict in results.items():
        plt.plot(data_dict['errors'], linewidth=2, label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('Comparison of CD-k for Different k Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

compare_cd_k_values()
```

## Analysis of CD Bias

### Theoretical Analysis

CD-k provides a biased estimate of the true MLE gradient. The bias decreases exponentially as $k$ increases:

$$\text{Bias}[\nabla_\theta^{\text{CD-}k}] = \mathcal{O}(\rho^k)$$

where $0 < \rho < 1$ is related to the spectral gap of the Markov chain transition operator. Larger spectral gaps (faster mixing) mean the bias decays more rapidly with $k$.

### Practical Trade-offs

| CD-k Value | Bias | Variance | Speed | Best For |
|------------|------|----------|-------|----------|
| $k=1$ | Higher | Low | Fast | Initial training, large datasets |
| $k=5-10$ | Moderate | Moderate | Medium | Fine-tuning, better models |
| $k=\infty$ | Zero | High | Infeasible | Theoretical reference only |

In practice, CD-1 is sufficient for most RBM applications. The bias of CD-1 tends to slow down learning rather than cause it to converge to a wrong solution—the learned model is slightly worse than the MLE solution but still useful.

```python
def analyze_cd_bias(n_steps_list: List[int] = [1, 5, 10, 20, 50]):
    """
    Analyze how CD bias changes with number of steps.
    """
    n_visible, n_hidden = 20, 10
    rbm = RBMWithCDVariants(n_visible, n_hidden)
    
    v_data = torch.bernoulli(torch.ones(1, n_visible) * 0.5)
    
    fig, axes = plt.subplots(1, len(n_steps_list), figsize=(15, 3))
    
    for idx, k in enumerate(n_steps_list):
        # Run many chains and collect endpoint statistics
        n_chains = 500
        v_chains = v_data.repeat(n_chains, 1)
        
        for _ in range(k):
            v_chains, _ = rbm.gibbs_step(v_chains)
        
        # Visualize average endpoint
        avg_endpoint = v_chains.mean(dim=0).numpy().reshape(4, 5)
        axes[idx].imshow(avg_endpoint, cmap='hot', vmin=0, vmax=1)
        axes[idx].set_title(f'k = {k}')
        axes[idx].axis('off')
    
    plt.suptitle('Average CD Endpoint for Different k')
    plt.tight_layout()
    plt.show()

analyze_cd_bias()
```

## Advanced Training Techniques

### Momentum

Adding momentum to parameter updates accelerates convergence:

```python
def cd_with_momentum(rbm, v_data, k=1, lr=0.01, momentum=0.9, 
                     W_velocity=None):
    """CD with momentum for faster convergence."""
    if W_velocity is None:
        W_velocity = torch.zeros_like(rbm.W)
    
    result = rbm.cd_k(v_data, k=k, lr=0.0)  # Compute gradients only
    
    # The gradient is implicitly (pos_grad - neg_grad)
    # Apply momentum to the update
    W_velocity = momentum * W_velocity + lr * (result['pos_grad'] - result['neg_grad'])
    rbm.W.data += W_velocity
    
    return W_velocity
```

### Weight Decay

L2 regularization prevents weights from growing too large, which improves mixing of the Gibbs chain and reduces overfitting:

```python
# Add weight decay to CD update
weight_decay = 0.0001
rbm.W.data -= lr * weight_decay * rbm.W.data
```

### Sparsity Regularization

Encouraging sparse hidden activations improves feature quality and makes learned representations more interpretable:

```python
def apply_sparsity_penalty(rbm, prob_h_data, sparsity_target=0.05, 
                           sparsity_cost=0.1, lr=0.01):
    """Penalize hidden units that fire too often."""
    avg_activation = prob_h_data.mean(dim=0)
    sparsity_penalty = sparsity_cost * (avg_activation - sparsity_target)
    rbm.b.data -= lr * sparsity_penalty
```

## Key Takeaways

!!! success "Core Concepts"
    1. The MLE gradient for EBMs has two phases: lower energy on data (positive) and raise energy on model samples (negative)
    2. CD-k approximates the intractable negative phase by running $k$ Gibbs steps starting from data
    3. The bias of CD-k decreases exponentially with $k$, but CD-1 is often sufficient in practice
    4. Starting MCMC from data exploits the proximity to the model distribution, reducing the number of steps needed
    5. Advanced techniques (momentum, weight decay, sparsity) improve training stability and model quality

!!! warning "Common Pitfalls"
    - CD-1 bias can cause the model to place probability mass slightly outside the data support—this is usually benign but worth monitoring
    - Too high a learning rate causes divergence because the model distribution changes faster than the MCMC chain can track
    - Reconstruction error is a useful diagnostic but does not directly measure log-likelihood

## Exercises

1. **Bias measurement**: Empirically measure the bias of CD-k by computing the true gradient (via exhaustive enumeration in a small system) and comparing it to the CD-k estimate for various $k$.

2. **Learning dynamics**: Track the evolution of the weight matrix during CD-1 training. How quickly do receptive fields emerge? How does this change with $k$?

3. **Financial data**: Train an RBM with CD-1 on binarized stock return data (above/below median). What features do the hidden units learn?

## References

- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. *Neural Computation*.
- Bengio, Y., & Delalleau, O. (2009). Justifying and Generalizing Contrastive Divergence. *Neural Computation*.
- Carreira-Perpiñán, M. A., & Hinton, G. E. (2005). On Contrastive Divergence Learning. *AISTATS*.
