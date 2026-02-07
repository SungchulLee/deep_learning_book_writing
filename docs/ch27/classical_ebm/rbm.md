# Restricted Boltzmann Machines

## Learning Objectives

After completing this section, you will be able to:

1. Understand the bipartite architecture of RBMs
2. Derive the tractable conditional distributions
3. Implement Contrastive Divergence training
4. Train RBMs on real data (MNIST)
5. Visualize learned features and perform reconstruction

## Introduction

Restricted Boltzmann Machines (RBMs) are the most successful practical application of energy-based learning. The "restricted" refers to the constraint that connections exist only between visible and hidden layers—no within-layer connections. This bipartite structure makes inference tractable through block Gibbs sampling and enables efficient training via Contrastive Divergence.

## Architecture

### Bipartite Graph Structure

RBMs have a specific connection pattern:

```
Visible Layer (v):    ○  ○  ○  ○  ○  ○
                       \\ | // \\ | //
                        \\|//   \\|//
Hidden Layer (h):        ○       ○       ○
```

**Key constraints**:
- No visible-visible connections ($W_{vv} = 0$)
- No hidden-hidden connections ($W_{hh} = 0$)
- Only visible-hidden connections ($W_{vh}$)

### Energy Function

The RBM energy function is:

$$E(\mathbf{v}, \mathbf{h}) = -\mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h} - \mathbf{v}^T \mathbf{W} \mathbf{h}$$

where:
- $\mathbf{v} \in \{0, 1\}^{n_v}$: Visible units (data)
- $\mathbf{h} \in \{0, 1\}^{n_h}$: Hidden units (features)
- $\mathbf{W} \in \mathbb{R}^{n_v \times n_h}$: Weight matrix
- $\mathbf{a} \in \mathbb{R}^{n_v}$: Visible biases
- $\mathbf{b} \in \mathbb{R}^{n_h}$: Hidden biases

## Tractable Conditionals

### The Key Insight

Due to the bipartite structure, the conditional distributions factor:

$$P(\mathbf{h} | \mathbf{v}) = \prod_j P(h_j | \mathbf{v})$$
$$P(\mathbf{v} | \mathbf{h}) = \prod_i P(v_i | \mathbf{h})$$

### Derivation

For hidden unit $j$:
$$P(h_j = 1 | \mathbf{v}) = \sigma(b_j + \sum_i W_{ij} v_i) = \sigma(\mathbf{W}_{:,j}^T \mathbf{v} + b_j)$$

For visible unit $i$:
$$P(v_i = 1 | \mathbf{h}) = \sigma(a_i + \sum_j W_{ij} h_j) = \sigma(\mathbf{W}_{i,:} \mathbf{h} + a_i)$$

where $\sigma(x) = 1/(1 + e^{-x})$ is the sigmoid function.

**This factorization enables parallel sampling of all units in a layer!**

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class RestrictedBoltzmannMachine(nn.Module):
    """
    Restricted Boltzmann Machine with binary visible and hidden units.
    
    Architecture: Bipartite graph with visible-hidden connections only.
    Training: Contrastive Divergence (CD-k)
    
    Parameters
    ----------
    n_visible : int
        Number of visible units
    n_hidden : int
        Number of hidden units
    k : int
        Number of Gibbs steps in CD-k (default: 1)
    learning_rate : float
        Learning rate for parameter updates
    """
    
    def __init__(self, 
                 n_visible: int, 
                 n_hidden: int,
                 k: int = 1,
                 learning_rate: float = 0.01):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = learning_rate
        
        # Parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))   # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))    # hidden bias
    
    def sample_hidden(self, v: torch.Tensor) -> tuple:
        """
        Sample hidden units given visible units.
        
        P(h_j = 1 | v) = σ(W_j · v + b_j)
        """
        activation = F.linear(v, self.W, self.b)
        prob_h = torch.sigmoid(activation)
        sample_h = torch.bernoulli(prob_h)
        return prob_h, sample_h
    
    def sample_visible(self, h: torch.Tensor) -> tuple:
        """
        Sample visible units given hidden units.
        
        P(v_i = 1 | h) = σ(W^T_i · h + a_i)
        """
        activation = F.linear(h, self.W.t(), self.a)
        prob_v = torch.sigmoid(activation)
        sample_v = torch.bernoulli(prob_v)
        return prob_v, sample_v
    
    def energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Compute energy E(v, h) = -a^T v - b^T h - v^T W^T h"""
        visible_term = torch.einsum('bi,i->b', v, self.a)
        hidden_term = torch.einsum('bj,j->b', h, self.b)
        interaction_term = torch.einsum('bi,ji,bj->b', v, self.W, h)
        return -(visible_term + hidden_term + interaction_term)
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy F(v) = -log Σ_h exp(-E(v,h))
        
        F(v) = -a^T v - Σ_j log(1 + exp(b_j + W_j · v))
        """
        visible_term = torch.einsum('bi,i->b', v, self.a)
        wx_b = F.linear(v, self.W, self.b)
        hidden_term = F.softplus(wx_b).sum(dim=1)
        return -(visible_term + hidden_term)
    
    def contrastive_divergence(self, v0: torch.Tensor) -> float:
        """
        Contrastive Divergence CD-k training step.
        """
        batch_size = v0.shape[0]
        
        # Positive phase
        prob_h0, h0 = self.sample_hidden(v0)
        positive_grad = torch.matmul(prob_h0.t(), v0) / batch_size
        
        # Negative phase (k Gibbs steps)
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        negative_grad = torch.matmul(hk.t(), vk) / batch_size
        
        # Update parameters
        self.W.data += self.lr * (positive_grad - negative_grad)
        self.a.data += self.lr * (v0 - vk).mean(dim=0)
        self.b.data += self.lr * (prob_h0 - hk).mean(dim=0)
        
        return ((v0 - vk) ** 2).sum(dim=1).mean().item()
    
    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """Reconstruct: v → h → v'"""
        _, h = self.sample_hidden(v)
        prob_v, _ = self.sample_visible(h)
        return prob_v
```

## Contrastive Divergence

### The CD-k Algorithm

**Exact gradient**:
$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$$

**CD-k approximation**:
$$\frac{\partial \log P(\mathbf{v})}{\partial W_{ij}} \approx \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{k}$$

### Why CD Works

- Starting from data helps the chain mix faster
- $k=1$ is often sufficient for good results
- The approximation is biased but has low variance

## Key Takeaways

!!! success "Core Concepts"
    1. RBMs restrict connections to visible-hidden pairs only
    2. Bipartite structure enables tractable conditional distributions
    3. Contrastive Divergence provides efficient approximate gradients
    4. Free energy is the effective energy over visible units
    5. RBMs learn useful feature representations

## References

- Hinton, G. E. (2002). Training Products of Experts by Minimizing Contrastive Divergence.
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A Fast Learning Algorithm for Deep Belief Nets.
