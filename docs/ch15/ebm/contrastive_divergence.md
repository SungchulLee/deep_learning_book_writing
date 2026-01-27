# Contrastive Divergence

## Learning Objectives

After completing this section, you will be able to:

1. Derive the maximum likelihood gradient for EBMs
2. Understand why exact gradient computation is intractable
3. Implement CD-k and Persistent CD algorithms
4. Analyze the bias-variance trade-off in CD approximations
5. Apply advanced training techniques for stable learning

## Introduction

Contrastive Divergence (CD), introduced by Geoffrey Hinton in 2002, revolutionized the training of energy-based models by providing a practical approximation to the intractable maximum likelihood gradient. This algorithm made RBMs trainable on real datasets and helped spark the deep learning renaissance of the late 2000s.

## Maximum Likelihood for EBMs

### The Log-Likelihood Gradient

For an EBM with energy $E_\theta(\mathbf{x})$, the log-likelihood of data $\mathbf{x}$ is:

$$\log P_\theta(\mathbf{x}) = -E_\theta(\mathbf{x}) - \log Z(\theta)$$

where $Z(\theta) = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ is the partition function.

The gradient is:

$$\nabla_\theta \log P_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) + \mathbb{E}_{P_\theta}[\nabla_\theta E_\theta(\mathbf{x})]$$

**Key insight**: The gradient has two terms:
1. **Positive phase** (data): $-\nabla_\theta E_\theta(\mathbf{x})$ — lower energy on data
2. **Negative phase** (model): $+\mathbb{E}_{P_\theta}[\nabla_\theta E_\theta]$ — raise energy elsewhere

### The Intractability Problem

The negative phase requires:
$$\mathbb{E}_{P_\theta}[\nabla_\theta E_\theta(\mathbf{x})] = \int P_\theta(\mathbf{x}) \nabla_\theta E_\theta(\mathbf{x}) d\mathbf{x}$$

This expectation is intractable because:
1. $P_\theta(\mathbf{x})$ requires computing $Z(\theta)$
2. The integral is over potentially high-dimensional space
3. Monte Carlo estimation requires samples from $P_\theta$

## The Contrastive Divergence Approximation

### Key Idea

Instead of running MCMC to equilibrium, CD starts the chain from the data and runs only $k$ steps:

$$\text{CD-}k: \quad \mathbf{x}^{(0)} = \mathbf{x}_{\text{data}} \xrightarrow{\text{k steps}} \mathbf{x}^{(k)}$$

### Gradient Approximation

$$\nabla_\theta \log P \approx -\nabla_\theta E_\theta(\mathbf{x}^{(0)}) + \nabla_\theta E_\theta(\mathbf{x}^{(k)})$$

For RBMs with weights $W$:
$$\Delta W \propto \langle \mathbf{v} \mathbf{h}^T \rangle_{\text{data}} - \langle \mathbf{v} \mathbf{h}^T \rangle_{k}$$

### Why It Works

1. **Fast mixing near data**: Starting from data, the chain is already in a high-probability region
2. **Good approximation for small changes**: For small learning rates, the model changes slowly
3. **Low variance**: Few steps means low variance compared to long chains

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
    
    def pcd(self, v_data: torch.Tensor, k: int = 1, lr: float = 0.01) -> dict:
        """
        Persistent Contrastive Divergence.
        
        Maintains persistent chains that continue between updates.
        Better approximation but can diverge if chains get stuck.
        """
        batch_size = v_data.shape[0]
        
        # Initialize persistent chains if needed
        if self.persistent_chains is None or self.persistent_chains.shape[0] != batch_size:
            self.persistent_chains = torch.bernoulli(
                torch.ones(batch_size, self.n_visible) * 0.5
            )
        
        # Positive phase
        prob_h_data, _ = self.sample_hidden(v_data)
        
        # Negative phase (continue from persistent chains)
        v_model = self.persistent_chains
        for _ in range(k):
            v_model, h_model = self.gibbs_step(v_model)
        
        # Update persistent chains
        self.persistent_chains = v_model.detach()
        
        # Compute gradients
        pos_grad_W = torch.matmul(prob_h_data.t(), v_data) / batch_size
        neg_grad_W = torch.matmul(h_model.t(), v_model) / batch_size
        
        # Update parameters
        self.W.data += lr * (pos_grad_W - neg_grad_W)
        self.a.data += lr * (v_data - v_model).mean(dim=0)
        self.b.data += lr * (prob_h_data - h_model).mean(dim=0)
        
        recon_error = ((v_data - v_model) ** 2).mean().item()
        
        return {
            'recon_error': recon_error,
            'v_model': v_model,
            'h_model': h_model
        }


def compare_cd_variants():
    """
    Compare CD-1, CD-10, and PCD on a simple task.
    """
    print("="*70)
    print("COMPARING CD VARIANTS")
    print("="*70)
    
    # Create simple synthetic data (mixture of patterns)
    n_visible = 100
    n_hidden = 50
    n_samples = 1000
    
    # Generate data from two patterns
    pattern1 = torch.zeros(n_visible)
    pattern1[:50] = 1
    pattern2 = torch.zeros(n_visible)
    pattern2[50:] = 1
    
    data = torch.zeros(n_samples, n_visible)
    for i in range(n_samples):
        base = pattern1 if i % 2 == 0 else pattern2
        noise = torch.bernoulli(torch.ones(n_visible) * 0.1)
        data[i] = (base + noise) % 2  # XOR for noise
    
    # Train with different methods
    methods = {
        'CD-1': lambda rbm, v: rbm.cd_k(v, k=1, lr=0.01),
        'CD-10': lambda rbm, v: rbm.cd_k(v, k=10, lr=0.01),
        'PCD-1': lambda rbm, v: rbm.pcd(v, k=1, lr=0.01)
    }
    
    results = {name: {'errors': []} for name in methods}
    
    n_epochs = 50
    batch_size = 64
    
    for name, train_fn in methods.items():
        print(f"\nTraining {name}...")
        rbm = RBMWithCDVariants(n_visible, n_hidden)
        
        for epoch in range(n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            epoch_error = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch = data[perm[i:i+batch_size]]
                result = train_fn(rbm, batch)
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
    plt.title('Comparison of CD Variants')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nObservations:")
    print("• CD-1: Fast but may have higher bias")
    print("• CD-10: Better approximation but slower")
    print("• PCD-1: Often better than CD-1 with same speed")

compare_cd_variants()
```

## Analysis of CD Bias

### Theoretical Analysis

CD-k provides a biased estimate of the gradient. The bias decreases as $k \to \infty$:

$$\text{Bias}[\nabla_\theta^{\text{CD-}k}] = \mathcal{O}(\rho^k)$$

where $\rho < 1$ is related to the spectral gap of the Markov chain.

### Practical Implications

| CD-k Value | Pros | Cons |
|------------|------|------|
| k=1 | Fast, low variance | Higher bias |
| k=5-10 | Better approximation | Slower |
| k=∞ (exact) | Unbiased | Infeasible |

```python
def analyze_cd_bias(n_steps_list: List[int] = [1, 5, 10, 20, 50]):
    """
    Analyze how CD bias changes with number of steps.
    """
    n_visible, n_hidden = 20, 10
    rbm = RBMWithCDVariants(n_visible, n_hidden)
    
    # Create simple data point
    v_data = torch.bernoulli(torch.ones(1, n_visible) * 0.5)
    
    # Track negative phase samples for different k
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

## Persistent Contrastive Divergence (PCD)

### Motivation

CD-k always restarts chains from data, which may bias samples toward the data distribution. PCD maintains persistent chains that continue between parameter updates.

### Algorithm

```
Initialize: Persistent chains X^(0) ∼ random

For each update step:
  1. Get data batch v_data
  2. Positive phase: sample h_data from P(h|v_data)
  3. Negative phase: 
     - Run k Gibbs steps from X^(t-1) to get X^(t)
     - Use X^(t) as negative samples
  4. Update parameters
  5. Store X^(t) for next iteration
```

### Advantages and Disadvantages

**Pros**:
- Better approximation to equilibrium
- Chains can explore away from data

**Cons**:
- Chains can get stuck in modes
- Requires more memory (storing chains)
- Sensitive to learning rate

## Advanced Training Techniques

### Momentum

Add momentum to parameter updates for faster convergence:

```python
def cd_with_momentum(self, v_data, k=1, lr=0.01, momentum=0.9):
    """CD with momentum."""
    # ... compute gradients ...
    
    # Update with momentum
    self.W_velocity = momentum * self.W_velocity + lr * grad_W
    self.W.data += self.W_velocity
```

### Weight Decay

Add L2 regularization to prevent overfitting:

```python
def cd_with_weight_decay(self, v_data, k=1, lr=0.01, weight_decay=0.0001):
    """CD with weight decay."""
    # ... compute gradients ...
    
    # Add weight decay
    self.W.data -= lr * weight_decay * self.W.data
```

### Sparsity Regularization

Encourage sparse hidden activations:

```python
def cd_with_sparsity(self, v_data, k=1, lr=0.01, sparsity_target=0.05, sparsity_cost=0.1):
    """CD with sparsity penalty on hidden units."""
    # ... compute positive phase ...
    
    # Sparsity penalty
    avg_activation = prob_h_data.mean(dim=0)
    sparsity_penalty = sparsity_cost * (avg_activation - sparsity_target)
    
    self.b.data -= lr * sparsity_penalty
```

## Connection to Other Methods

### Score Matching

Both CD and score matching avoid computing $Z$:
- CD: approximates model expectation via short MCMC
- Score matching: matches gradients of log probability

### Noise Contrastive Estimation

NCE learns to distinguish data from noise:
$$\log \frac{P_\theta(\mathbf{x})}{P_\theta(\mathbf{x}) + P_{\text{noise}}(\mathbf{x})}$$

### Modern EBM Training

Recent approaches use:
- Langevin dynamics instead of Gibbs sampling
- Replay buffers to improve sample diversity
- Short-run MCMC similar to CD

## Key Takeaways

!!! success "Core Concepts"
    1. CD approximates the intractable MLE gradient
    2. Starts MCMC from data and runs k steps
    3. Trade-off: more steps = less bias but slower
    4. PCD maintains persistent chains between updates
    5. Advanced techniques: momentum, weight decay, sparsity

!!! warning "Common Pitfalls"
    - Too few CD steps may cause slow learning
    - Too many steps wastes computation
    - PCD chains can get stuck in modes
    - Learning rate too high can cause divergence

## Exercises

1. **Bias-Variance Analysis**: Empirically measure the bias and variance of CD-k for different values of k.

2. **PCD Diagnostics**: Implement diagnostics to detect when PCD chains get stuck.

3. **Parallel Tempering**: Implement parallel tempering CD where chains run at different temperatures.

## References

- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation.
- Tieleman, T. (2008). Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient. ICML.
- Bengio, Y., & Delalleau, O. (2009). Justifying and Generalizing Contrastive Divergence. Neural Computation.
