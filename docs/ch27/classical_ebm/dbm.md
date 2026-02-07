# Deep Boltzmann Machines

## Learning Objectives

After completing this section, you will be able to:

1. Understand the architecture and energy function of Deep Boltzmann Machines
2. Explain why DBMs require approximate inference and how mean-field variational methods address this
3. Implement layer-wise pretraining followed by joint fine-tuning
4. Analyze the representational advantages of depth in Boltzmann machines

## Introduction

Deep Boltzmann Machines (DBMs), introduced by Salakhutdinov and Hinton (2009), extend Restricted Boltzmann Machines by stacking multiple layers of hidden units with undirected connections between adjacent layers. Unlike Deep Belief Networks (which combine directed and undirected connections), DBMs maintain fully undirected connections throughout, enabling a richer approximate posterior inference through bottom-up and top-down feedback.

The move from shallow RBMs to deep architectures mirrors the broader deep learning principle: hierarchical representations capture increasingly abstract features. In a DBM, the first hidden layer might learn local correlations (e.g., short-term price movements), the second layer captures patterns among those correlations (e.g., sector-level dynamics), and higher layers represent global structure (e.g., market regimes).

## Architecture

### Multi-Layer Structure

A DBM with $L$ hidden layers has the following structure:

```
Visible (v):    ○  ○  ○  ○  ○  ○
                 \\ | // \\ | //
Hidden 1 (h¹):     ○  ○  ○  ○
                     \\ | //
Hidden 2 (h²):       ○  ○  ○
                       \\ |
Hidden 3 (h³):         ○  ○
```

Each layer connects only to its immediate neighbors—no skip connections and no within-layer connections.

### Energy Function

The energy function for a DBM with two hidden layers is:

$$E(\mathbf{v}, \mathbf{h}^1, \mathbf{h}^2) = -\mathbf{v}^T \mathbf{W}^1 \mathbf{h}^1 - (\mathbf{h}^1)^T \mathbf{W}^2 \mathbf{h}^2 - \mathbf{a}^T \mathbf{v} - \mathbf{b}_1^T \mathbf{h}^1 - \mathbf{b}_2^T \mathbf{h}^2$$

For $L$ hidden layers, this generalizes to:

$$E(\mathbf{v}, \mathbf{h}^1, \ldots, \mathbf{h}^L) = -\sum_{\ell=0}^{L-1} (\mathbf{h}^\ell)^T \mathbf{W}^{\ell+1} \mathbf{h}^{\ell+1} - \sum_{\ell=0}^{L} \mathbf{b}_\ell^T \mathbf{h}^\ell$$

where $\mathbf{h}^0 \equiv \mathbf{v}$ denotes the visible layer.

### Key Difference from RBMs

In an RBM, the hidden units are conditionally independent given the visible units, making exact inference tractable. In a DBM, this conditional independence breaks down because hidden units in layer $\ell$ receive input from both layer $\ell-1$ and layer $\ell+1$:

$$P(h_j^1 = 1 | \mathbf{v}, \mathbf{h}^2) = \sigma\left(\sum_i W^1_{ij} v_i + \sum_k W^2_{jk} h_k^2 + b_{1,j}\right)$$

This bidirectional dependency is what makes DBMs more powerful than stacked RBMs but also more challenging to train.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class DeepBoltzmannMachine(nn.Module):
    """
    Deep Boltzmann Machine with multiple hidden layers.
    
    Parameters
    ----------
    layer_sizes : list of int
        Sizes of each layer [n_visible, n_hidden1, n_hidden2, ...]
    """
    
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        # Weight matrices between adjacent layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            for i in range(self.n_layers - 1)
        ])
        
        # Bias for each layer
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(size))
            for size in layer_sizes
        ])
    
    def energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute energy of a complete configuration.
        
        Parameters
        ----------
        layers : list of torch.Tensor
            State of each layer [v, h1, h2, ...]
        
        Returns
        -------
        torch.Tensor
            Energy value(s)
        """
        E = torch.zeros(layers[0].shape[0])
        
        # Interaction terms
        for ell in range(self.n_layers - 1):
            E -= torch.einsum(
                'bi,ij,bj->b', layers[ell], self.weights[ell], layers[ell+1]
            )
        
        # Bias terms
        for ell in range(self.n_layers):
            E -= torch.einsum('i,bi->b', self.biases[ell], layers[ell])
        
        return E
    
    def conditional_prob(self, layer_idx: int, 
                         layers: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute P(h^ℓ = 1 | neighboring layers).
        
        Each unit receives input from the layer above and below.
        """
        activation = self.biases[layer_idx].clone()
        
        # Input from layer below
        if layer_idx > 0:
            activation = activation + F.linear(
                layers[layer_idx - 1], self.weights[layer_idx - 1].t()
            )
        
        # Input from layer above
        if layer_idx < self.n_layers - 1:
            activation = activation + F.linear(
                layers[layer_idx + 1], self.weights[layer_idx]
            )
        
        return torch.sigmoid(activation)
    
    def mean_field_update(self, layers: List[torch.Tensor], 
                          n_iterations: int = 20) -> List[torch.Tensor]:
        """
        Mean-field variational inference.
        
        Iteratively update each hidden layer's mean-field parameters
        until convergence.
        
        Parameters
        ----------
        layers : list of torch.Tensor
            Initial values (visible layer is fixed)
        n_iterations : int
            Number of mean-field iterations
        
        Returns
        -------
        list of torch.Tensor
            Updated mean-field parameters for each layer
        """
        mu = [layer.clone() for layer in layers]
        
        for _ in range(n_iterations):
            # Update hidden layers (keep visible fixed)
            for ell in range(1, self.n_layers):
                mu[ell] = self.conditional_prob(ell, mu)
        
        return mu
    
    def gibbs_sampling(self, layers: List[torch.Tensor],
                       n_steps: int = 1) -> List[torch.Tensor]:
        """
        Gibbs sampling for the negative phase.
        
        Updates all layers in alternating fashion.
        """
        samples = [layer.clone() for layer in layers]
        
        for _ in range(n_steps):
            # Update odd layers given even layers
            for ell in range(1, self.n_layers, 2):
                prob = self.conditional_prob(ell, samples)
                samples[ell] = torch.bernoulli(prob)
            
            # Update even layers given odd layers
            for ell in range(0, self.n_layers, 2):
                prob = self.conditional_prob(ell, samples)
                samples[ell] = torch.bernoulli(prob)
        
        return samples


def pretrain_dbm_layerwise(dbm: DeepBoltzmannMachine,
                           data: torch.Tensor,
                           n_epochs: int = 50,
                           lr: float = 0.01,
                           cd_k: int = 1):
    """
    Greedy layer-wise pretraining for DBM.
    
    Train each pair of adjacent layers as an RBM,
    then use the learned weights to initialize the DBM.
    
    Important: For the DBM, the first and last RBMs use 
    doubled weights to account for missing top-down/bottom-up input.
    """
    current_input = data.clone()
    
    for ell in range(dbm.n_layers - 1):
        print(f"\nPretraining layer {ell} → {ell+1}")
        n_visible = dbm.layer_sizes[ell]
        n_hidden = dbm.layer_sizes[ell + 1]
        
        # Train RBM for this layer pair
        W = torch.randn(n_visible, n_hidden) * 0.01
        a = torch.zeros(n_visible)
        b = torch.zeros(n_hidden)
        
        for epoch in range(n_epochs):
            # Positive phase
            prob_h = torch.sigmoid(current_input @ W + b)
            h_sample = torch.bernoulli(prob_h)
            
            # Negative phase (CD-k)
            v_neg = current_input.clone()
            for _ in range(cd_k):
                h_neg = torch.bernoulli(torch.sigmoid(v_neg @ W + b))
                v_neg = torch.bernoulli(torch.sigmoid(h_neg @ W.t() + a))
            
            h_neg_prob = torch.sigmoid(v_neg @ W + b)
            
            # Update
            batch_size = current_input.shape[0]
            dW = (current_input.t() @ prob_h - v_neg.t() @ h_neg_prob) / batch_size
            da = (current_input - v_neg).mean(0)
            db = (prob_h - h_neg_prob).mean(0)
            
            W += lr * dW
            a += lr * da
            b += lr * db
            
            if (epoch + 1) % 10 == 0:
                recon_err = ((current_input - v_neg)**2).mean().item()
                print(f"  Epoch {epoch+1}: recon error = {recon_err:.4f}")
        
        # Copy weights to DBM (with doubling for intermediate layers)
        with torch.no_grad():
            if ell == 0:
                # First layer: double weights for missing top-down input
                dbm.weights[ell].copy_(W * 2)
            elif ell == dbm.n_layers - 2:
                # Last pair: double weights for missing bottom-up input  
                dbm.weights[ell].copy_(W * 2)
            else:
                dbm.weights[ell].copy_(W)
            
            dbm.biases[ell].copy_(a)
            dbm.biases[ell + 1].copy_(b)
        
        # Propagate data through trained layer for next stage
        with torch.no_grad():
            current_input = torch.sigmoid(current_input @ W + b)
    
    print("\nLayerwise pretraining complete!")


# Example: 3-layer DBM
dbm = DeepBoltzmannMachine(layer_sizes=[784, 500, 200])

# Generate synthetic data for demonstration
data = torch.bernoulli(torch.rand(1000, 784))

# Pretrain
pretrain_dbm_layerwise(dbm, data, n_epochs=30)

# Mean-field inference on test point
test_input = torch.bernoulli(torch.rand(1, 784))
initial_layers = [
    test_input,
    torch.rand(1, 500),
    torch.rand(1, 200)
]
inferred = dbm.mean_field_update(initial_layers, n_iterations=50)
print(f"\nMean-field converged: h1 mean = {inferred[1].mean():.3f}, "
      f"h2 mean = {inferred[2].mean():.3f}")
```

## Mean-Field Variational Inference

Since exact inference in DBMs is intractable (computing $P(\mathbf{h}|\mathbf{v})$ requires marginalizing over exponentially many hidden configurations), DBMs rely on mean-field variational inference.

### The Variational Approximation

We approximate the true posterior $P(\mathbf{h}|\mathbf{v})$ with a factored distribution:

$$Q(\mathbf{h}|\mathbf{v}) = \prod_{\ell=1}^{L} \prod_{j} q(h_j^\ell | \mathbf{v})$$

where each $q(h_j^\ell = 1 | \mathbf{v}) = \mu_j^\ell$ is a Bernoulli parameter. The mean-field parameters $\mu_j^\ell$ are found by minimizing the KL divergence $\text{KL}(Q \| P)$, which leads to the fixed-point equations:

$$\mu_j^\ell = \sigma\left(\sum_i W^{\ell}_{ij} \mu_i^{\ell-1} + \sum_k W^{\ell+1}_{jk} \mu_k^{\ell+1} + b_{\ell,j}\right)$$

These equations are iterated until convergence, typically requiring 10–50 iterations.

### Training with Variational Learning

The DBM training procedure combines mean-field inference (positive phase) with MCMC sampling (negative phase):

1. **Positive phase**: For each data point $\mathbf{v}$, run mean-field to obtain approximate posterior $Q(\mathbf{h}|\mathbf{v})$. Compute data statistics $\langle v_i \mu_j^1 \rangle_{\text{data}}$.

2. **Negative phase**: Run MCMC (typically persistent chains) to sample from the model $P(\mathbf{v}, \mathbf{h})$. Compute model statistics $\langle v_i h_j^1 \rangle_{\text{model}}$.

3. **Update**: $\Delta W_{ij}^1 \propto \langle v_i \mu_j^1 \rangle_{\text{data}} - \langle v_i h_j^1 \rangle_{\text{model}}$

## Advantages of Depth

### Representational Power

DBMs can represent certain distributions exponentially more efficiently than shallow models. A distribution that requires $O(2^n)$ hidden units in a single-layer RBM may be representable with $O(n)$ units spread across $O(\log n)$ layers. This is analogous to the depth-width trade-off in feed-forward networks.

### Robust Feature Learning

The bidirectional connections in DBMs allow top-down feedback during inference, enabling the model to resolve ambiguities using high-level context. For example, when modeling financial time series, knowledge that the market is in a crisis regime (high-level feature) can influence the interpretation of individual price movements (low-level features).

## Key Takeaways

!!! success "Core Concepts"
    1. DBMs stack multiple hidden layers with undirected connections between adjacent layers
    2. Bidirectional connections break the conditional independence that makes RBMs tractable, requiring variational approximations
    3. Mean-field variational inference provides tractable approximate posteriors through iterative fixed-point equations
    4. Greedy layer-wise pretraining initializes DBM weights using RBMs trained on successive representations
    5. Depth enables exponentially more compact representations and top-down contextual feedback

!!! warning "Practical Considerations"
    - DBM training is significantly more complex and less stable than RBM training
    - Mean-field inference introduces approximation error that can degrade learning
    - Modern deep generative models (VAEs, diffusion models) have largely superseded DBMs for practical applications
    - The conceptual insights from DBMs—variational inference, layer-wise pretraining, bidirectional feedback—remain foundational

## Exercises

1. **Depth vs. width**: Compare the reconstruction quality of a DBM with layers [784, 500, 200] against a single RBM with 700 hidden units (similar total parameters). Train both on MNIST and compare.

2. **Mean-field convergence**: Monitor the KL divergence during mean-field iterations. How many iterations are needed for convergence? How does this depend on network size?

3. **Pretraining ablation**: Compare DBM performance with and without layer-wise pretraining. How much does pretraining help?

## References

- Salakhutdinov, R., & Hinton, G. E. (2009). Deep Boltzmann Machines. *AISTATS*.
- Salakhutdinov, R., & Hinton, G. E. (2012). An Efficient Learning Procedure for Deep Boltzmann Machines. *Neural Computation*.
- Goodfellow, I., et al. (2013). Multi-Prediction Deep Boltzmann Machines. *NeurIPS*.
