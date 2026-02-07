# Persistent Contrastive Divergence

## Learning Objectives

After completing this section, you will be able to:

1. Explain why persistent chains improve upon standard CD
2. Implement PCD with proper chain management and diagnostics
3. Diagnose when persistent chains get stuck and apply remedies
4. Compare PCD performance against CD-k empirically

## Motivation

Standard CD-k always restarts the Gibbs chain from the current data point. While this ensures the chain starts in a high-probability region, it also means the negative phase samples are biased toward the data distribution rather than the model distribution. As training progresses and the model distribution diverges from the data, this bias worsens.

Persistent Contrastive Divergence (PCD), introduced by Tieleman (2008), addresses this by maintaining persistent Markov chains that continue between parameter updates rather than restarting from data. These chains gradually explore the model distribution, providing less biased negative phase samples.

## Algorithm

The PCD procedure maintains a set of "fantasy particles"—persistent MCMC chains that run continuously throughout training:

```
Initialize: Persistent chains X^(0) ~ random or data

For each training step:
  1. Get data batch v_data
  2. Positive phase: compute P(h|v_data) from data
  3. Negative phase: 
     - Continue chains from X^(t-1) for k Gibbs steps → X^(t)
     - Use X^(t) as negative samples
  4. Update parameters: ΔW ∝ ⟨vh⟩_data - ⟨vh⟩_chains
  5. Store X^(t) for next iteration
```

The critical difference from CD: the chains in step 3 start from their previous positions $X^{(t-1)}$, not from the current data batch.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

class RBMWithPCD(nn.Module):
    """
    RBM trained with Persistent Contrastive Divergence.
    """
    
    def __init__(self, n_visible: int, n_hidden: int, n_chains: int = 100):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))
        self.b = nn.Parameter(torch.zeros(n_hidden))
        
        # Persistent chains (fantasy particles)
        self.register_buffer(
            'persistent_v',
            torch.bernoulli(torch.ones(n_chains, n_visible) * 0.5)
        )
        
        # Diagnostics
        self.chain_energies_history = []
    
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
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute free energy F(v) = -a^T v - Σ log(1 + exp(Wv + b)).
        """
        wx_b = F.linear(v, self.W, self.b)
        return -torch.mv(v, self.a) - wx_b.clamp(min=-20).exp().log1p().sum(dim=1)
    
    def pcd_step(self, v_data: torch.Tensor, k: int = 1, 
                 lr: float = 0.01) -> dict:
        """
        One PCD training step.
        
        Parameters
        ----------
        v_data : torch.Tensor
            Data batch
        k : int
            Gibbs steps per update
        lr : float
            Learning rate
        
        Returns
        -------
        dict
            Training metrics
        """
        batch_size = v_data.shape[0]
        n_chains = self.persistent_v.shape[0]
        
        # === Positive phase ===
        prob_h_data, _ = self.sample_hidden(v_data)
        
        # === Negative phase (continue persistent chains) ===
        v_chain = self.persistent_v.clone()
        for _ in range(k):
            _, h_chain = self.sample_hidden(v_chain)
            _, v_chain = self.sample_visible(h_chain)
        
        prob_h_chain, _ = self.sample_hidden(v_chain)
        
        # Update persistent chains
        self.persistent_v.copy_(v_chain.detach())
        
        # === Compute gradients ===
        # Use all chains for negative phase statistics
        pos_grad_W = torch.matmul(prob_h_data.t(), v_data) / batch_size
        neg_grad_W = torch.matmul(prob_h_chain.t(), v_chain) / n_chains
        
        # === Parameter updates ===
        self.W.data += lr * (pos_grad_W - neg_grad_W)
        self.a.data += lr * (v_data.mean(0) - v_chain.mean(0))
        self.b.data += lr * (prob_h_data.mean(0) - prob_h_chain.mean(0))
        
        # === Diagnostics ===
        recon_error = self._reconstruction_error(v_data)
        chain_energy = self.free_energy(self.persistent_v).mean().item()
        data_energy = self.free_energy(v_data).mean().item()
        self.chain_energies_history.append(chain_energy)
        
        return {
            'recon_error': recon_error,
            'chain_energy': chain_energy,
            'data_energy': data_energy,
            'energy_gap': chain_energy - data_energy
        }
    
    def _reconstruction_error(self, v: torch.Tensor) -> float:
        """Compute reconstruction error: v → h → v'"""
        _, h = self.sample_hidden(v)
        prob_v, _ = self.sample_visible(h)
        return ((v - prob_v) ** 2).mean().item()
    
    def diagnose_chains(self) -> dict:
        """
        Diagnostic checks for persistent chain health.
        
        Returns warnings if chains appear stuck.
        """
        diagnostics = {}
        
        # Check chain diversity
        pairwise_dist = torch.cdist(
            self.persistent_v.float(), 
            self.persistent_v.float()
        )
        avg_distance = pairwise_dist.sum() / (
            self.persistent_v.shape[0] * (self.persistent_v.shape[0] - 1)
        )
        diagnostics['avg_chain_distance'] = avg_distance.item()
        
        # Check if chains have collapsed
        unique_chains = len(torch.unique(self.persistent_v, dim=0))
        diagnostics['unique_chains'] = unique_chains
        diagnostics['total_chains'] = self.persistent_v.shape[0]
        
        # Energy statistics
        energies = self.free_energy(self.persistent_v)
        diagnostics['energy_mean'] = energies.mean().item()
        diagnostics['energy_std'] = energies.std().item()
        
        # Warnings
        if unique_chains < self.persistent_v.shape[0] * 0.5:
            diagnostics['warning'] = "Many duplicate chains - possible mode collapse"
        elif diagnostics['energy_std'] < 0.1:
            diagnostics['warning'] = "Low energy variance - chains may be stuck"
        else:
            diagnostics['warning'] = None
        
        return diagnostics


def compare_cd_vs_pcd():
    """
    Compare CD-1 and PCD-1 on a synthetic task.
    """
    n_visible = 100
    n_hidden = 50
    n_samples = 1000
    
    # Generate data with clear structure
    data = torch.zeros(n_samples, n_visible)
    for i in range(n_samples):
        pattern_idx = i % 3
        start = pattern_idx * 33
        end = min(start + 33, n_visible)
        data[i, start:end] = 1.0
        # Add noise
        noise_mask = torch.bernoulli(torch.ones(n_visible) * 0.05)
        data[i] = (data[i] + noise_mask) % 2
    
    # Train CD-1
    rbm_cd = RBMWithPCD(n_visible, n_hidden)
    cd_errors = []
    
    # Train PCD-1
    rbm_pcd = RBMWithPCD(n_visible, n_hidden)
    pcd_errors = []
    
    n_epochs = 100
    batch_size = 64
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        cd_err, pcd_err = 0, 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch = data[perm[i:i+batch_size]]
            
            # CD-1 (restart chains from data each time)
            rbm_cd.persistent_v.copy_(batch[:rbm_cd.persistent_v.shape[0]])
            r_cd = rbm_cd.pcd_step(batch, k=1, lr=0.01)
            cd_err += r_cd['recon_error']
            
            # PCD-1 (continue persistent chains)
            r_pcd = rbm_pcd.pcd_step(batch, k=1, lr=0.01)
            pcd_err += r_pcd['recon_error']
            
            n_batches += 1
        
        cd_errors.append(cd_err / n_batches)
        pcd_errors.append(pcd_err / n_batches)
        
        if (epoch + 1) % 20 == 0:
            diag = rbm_pcd.diagnose_chains()
            print(f"Epoch {epoch+1}: CD={cd_errors[-1]:.4f}, "
                  f"PCD={pcd_errors[-1]:.4f}, "
                  f"Unique chains={diag['unique_chains']}/{diag['total_chains']}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(cd_errors, label='CD-1', linewidth=2)
    plt.plot(pcd_errors, label='PCD-1', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('CD-1 vs PCD-1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

compare_cd_vs_pcd()
```

## Advantages and Limitations

### Advantages of PCD

**Better negative phase approximation**: Because chains run continuously, they can explore regions of the model distribution that are far from the data, providing a more accurate estimate of the model expectation.

**Tracks model distribution changes**: As the model parameters evolve, the persistent chains gradually adapt to the new distribution, maintaining a set of approximate model samples.

**Same computational cost as CD**: For the same value of $k$, PCD requires the same number of Gibbs steps per update as CD. The only additional cost is storing the persistent chains.

### Limitations

**Chain stagnation**: If the learning rate is too high, the model distribution can change faster than the chains can track, causing the chains to get stuck in regions of low probability under the updated model.

**Mode collapse**: Persistent chains can collapse into a subset of modes, failing to represent the full model distribution. This is especially problematic for multimodal distributions.

**Memory overhead**: Maintaining $M$ persistent chains of dimension $d$ requires $O(Md)$ additional storage.

### Remedies for Chain Problems

**Occasional chain reinitialization**: Periodically replace a fraction of chains with random samples or data samples to maintain diversity.

**Parallel tempering**: Run chains at multiple temperatures and swap configurations between temperatures to improve mixing.

**Monitoring**: Track chain diversity metrics (pairwise distances, unique configurations, energy statistics) and intervene when chains appear stuck.

## Key Takeaways

!!! success "Core Concepts"
    1. PCD maintains persistent MCMC chains that continue between parameter updates
    2. Persistent chains provide less biased negative phase estimates than CD's data-initialized chains
    3. Chain health must be monitored—stagnation and mode collapse are common failure modes
    4. PCD-1 often outperforms CD-1 with no additional computational cost per step
    5. The PCD framework generalizes to modern EBM training with Langevin dynamics and replay buffers

!!! tip "Connection to Modern Methods"
    The replay buffer strategy used in modern neural EBM training (Section 26.4) is a direct descendant of PCD. Instead of maintaining persistent Gibbs chains, modern methods maintain a buffer of previous MCMC samples that serve as initializations for Langevin dynamics chains—the same principle of reusing previous samples to improve mixing.

## Exercises

1. **Chain diagnostics**: Implement a visualization that shows the evolution of persistent chains over training. Plot the average chain state at regular intervals.

2. **Tempering**: Implement parallel tempering PCD where chains run at different temperatures $T \in \{1, 2, 4, 8\}$ and swap proposals are made between adjacent temperatures.

3. **Adaptive reinitialization**: Design an adaptive scheme that reinitializes chains when their energy exceeds the data energy by more than a threshold. How does this affect training?

## References

- Tieleman, T. (2008). Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient. *ICML*.
- Tieleman, T., & Hinton, G. E. (2009). Using Fast Weights to Improve Persistent Contrastive Divergence. *ICML*.
- Desjardins, G., et al. (2010). Tempered Markov Chain Monte Carlo for training of Restricted Boltzmann Machines. *AISTATS*.
