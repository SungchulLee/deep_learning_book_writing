# Hopfield Networks

## Learning Objectives

After completing this section, you will be able to:

1. Understand Hopfield networks as energy-based associative memories
2. Implement pattern storage using Hebbian learning
3. Perform pattern retrieval through energy minimization
4. Analyze network capacity and spurious states
5. Connect Hopfield networks to modern energy-based models

## Introduction

Hopfield networks, introduced by John Hopfield in 1982, represent one of the earliest and most influential energy-based models. They demonstrate how energy minimization can implement associative memory—the ability to retrieve complete patterns from partial or corrupted inputs. This work helped revive interest in neural networks and established deep connections between neuroscience, physics, and computation.

## Architecture and Dynamics

### Network Structure

A Hopfield network consists of $N$ binary neurons with symmetric, fully-connected weights:

- **Neurons**: $s_i \in \{-1, +1\}$ for $i = 1, \ldots, N$
- **Weights**: $w_{ij} = w_{ji}$ (symmetric)
- **No self-connections**: $w_{ii} = 0$
- **Biases/Thresholds**: $\theta_i$ (often set to 0)

### Energy Function

The energy of a network state $\mathbf{s} = (s_1, \ldots, s_N)$ is:

$$E(\mathbf{s}) = -\frac{1}{2} \sum_{i,j} w_{ij} s_i s_j - \sum_i \theta_i s_i$$

In matrix form:
$$E(\mathbf{s}) = -\frac{1}{2} \mathbf{s}^T \mathbf{W} \mathbf{s} - \boldsymbol{\theta}^T \mathbf{s}$$

### Update Rule

Neurons update asynchronously using a deterministic threshold rule:

$$s_i \leftarrow \text{sign}(h_i)$$

where the local field is:
$$h_i = \sum_j w_{ij} s_j + \theta_i$$

**Critical property**: Asynchronous updates never increase energy.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class HopfieldNetwork(nn.Module):
    """
    Binary Hopfield Network with asynchronous updates.
    
    Implements associative memory through energy minimization.
    Patterns are stored as local minima of the energy function.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons in the network
    """
    
    def __init__(self, n_neurons: int):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Weight matrix (symmetric, no self-connections)
        self.register_buffer('W', torch.zeros(n_neurons, n_neurons))
        
        # Thresholds (biases)
        self.register_buffer('theta', torch.zeros(n_neurons))
        
        # For tracking energy during retrieval
        self.energy_history = []
    
    def train_hebbian(self, patterns: torch.Tensor) -> None:
        """
        Store patterns using Hebbian learning rule.
        
        "Neurons that fire together wire together"
        
        w_ij = (1/P) Σₚ xᵢᵖ xⱼᵖ  for i ≠ j
        
        Parameters
        ----------
        patterns : torch.Tensor
            Shape (n_patterns, n_neurons), values in {-1, +1}
        """
        n_patterns = patterns.shape[0]
        
        # Reset weights
        self.W.zero_()
        
        # Hebbian rule: outer product sum
        for pattern in patterns:
            self.W += torch.outer(pattern, pattern)
        
        # Normalize by number of patterns
        self.W /= n_patterns
        
        # Remove self-connections
        self.W.fill_diagonal_(0)
        
        print(f"Stored {n_patterns} patterns via Hebbian learning")
        print(f"  Network capacity: ~{int(0.15 * self.n_neurons)} patterns")
    
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of network state(s).
        
        E(s) = -½ sᵀWs - θᵀs
        
        Parameters
        ----------
        state : torch.Tensor
            Shape (n_neurons,) or (batch, n_neurons)
        
        Returns
        -------
        torch.Tensor
            Energy value(s)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Quadratic term: -½ sᵀWs
        quadratic = -0.5 * torch.einsum('bi,ij,bj->b', state, self.W, state)
        
        # Linear term: -θᵀs
        linear = -torch.einsum('i,bi->b', self.theta, state)
        
        return quadratic + linear
    
    def local_field(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute local field for all neurons.
        
        h_i = Σⱼ w_ij s_j + θ_i
        """
        return torch.mv(self.W, state) + self.theta
    
    def update_neuron(self, state: torch.Tensor, neuron_idx: int) -> torch.Tensor:
        """
        Asynchronously update a single neuron.
        
        s_i ← sign(h_i)
        
        This update is guaranteed to not increase energy.
        """
        new_state = state.clone()
        h_i = self.local_field(state)[neuron_idx]
        
        # Threshold activation
        if h_i > 0:
            new_state[neuron_idx] = 1.0
        elif h_i < 0:
            new_state[neuron_idx] = -1.0
        # If h_i == 0, keep current state
        
        return new_state
    
    def retrieve(self, 
                 initial_state: torch.Tensor,
                 max_iterations: int = 100,
                 track_energy: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Retrieve pattern from initial (possibly corrupted) state.
        
        Performs asynchronous updates until convergence.
        
        Parameters
        ----------
        initial_state : torch.Tensor
            Starting state (corrupted pattern)
        max_iterations : int
            Maximum update cycles
        track_energy : bool
            Whether to record energy history
        
        Returns
        -------
        final_state : torch.Tensor
            Retrieved pattern (local energy minimum)
        n_iterations : int
            Iterations until convergence
        """
        state = initial_state.clone()
        self.energy_history = []
        
        if track_energy:
            self.energy_history.append(self.energy(state).item())
        
        for iteration in range(max_iterations):
            old_state = state.clone()
            
            # Update neurons in random order
            update_order = torch.randperm(self.n_neurons)
            
            for neuron_idx in update_order:
                state = self.update_neuron(state, neuron_idx.item())
            
            if track_energy:
                self.energy_history.append(self.energy(state).item())
            
            # Check convergence
            if torch.equal(state, old_state):
                return state, iteration + 1
        
        return state, max_iterations
    
    def compute_overlap(self, state: torch.Tensor, pattern: torch.Tensor) -> float:
        """
        Compute overlap (similarity) between state and pattern.
        
        Overlap = (1/N) Σᵢ sᵢ pᵢ ∈ [-1, 1]
        
        +1: identical
        -1: inverted
         0: uncorrelated
        """
        return (state * pattern).mean().item()


def demonstrate_hopfield_retrieval():
    """
    Demonstrate pattern storage and retrieval with Hopfield network.
    """
    # Create simple 5x5 patterns (flattened to 25 neurons)
    pattern_A = torch.tensor([
        [-1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, +1],
        [+1, -1, -1, -1, +1],
        [+1, -1, -1, -1, +1]
    ], dtype=torch.float32).flatten()
    
    pattern_B = torch.tensor([
        [+1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1],
        [+1, -1, -1, -1, +1],
        [+1, +1, +1, +1, -1]
    ], dtype=torch.float32).flatten()
    
    patterns = torch.stack([pattern_A, pattern_B])
    
    # Create and train network
    network = HopfieldNetwork(n_neurons=25)
    network.train_hebbian(patterns)
    
    # Corrupt pattern A (flip 5 random bits)
    corrupted = pattern_A.clone()
    flip_indices = torch.randperm(25)[:5]
    corrupted[flip_indices] *= -1
    
    # Retrieve
    retrieved, n_iter = network.retrieve(corrupted)
    
    # Compute overlaps
    overlap_A = network.compute_overlap(retrieved, pattern_A)
    overlap_B = network.compute_overlap(retrieved, pattern_B)
    
    print(f"\nRetrieval completed in {n_iter} iterations")
    print(f"Overlap with pattern A: {overlap_A:.3f}")
    print(f"Overlap with pattern B: {overlap_B:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    # Row 1: Patterns and retrieval
    axes[0, 0].imshow(pattern_A.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 0].set_title('Original Pattern A')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(corrupted.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 1].set_title('Corrupted Input')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(retrieved.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 2].set_title('Retrieved Pattern')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(pattern_B.reshape(5, 5), cmap='binary', vmin=-1, vmax=1)
    axes[0, 3].set_title('Pattern B (not retrieved)')
    axes[0, 3].axis('off')
    
    # Row 2: Energy and weight matrix
    axes[1, 0].plot(network.energy_history, 'b-o', markersize=4)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Descent During Retrieval')
    axes[1, 0].grid(True, alpha=0.3)
    
    im = axes[1, 1].imshow(network.W, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('Weight Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Energy of stored patterns
    pattern_energies = network.energy(patterns).numpy()
    axes[1, 2].bar(['Pattern A', 'Pattern B'], pattern_energies)
    axes[1, 2].set_ylabel('Energy')
    axes[1, 2].set_title('Energy of Stored Patterns')
    axes[1, 2].grid(True, alpha=0.3)
    
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return network

network = demonstrate_hopfield_retrieval()
```

## Network Capacity

### Theoretical Capacity

The maximum number of patterns a Hopfield network can reliably store is approximately:

$$P_{\max} \approx 0.15 N$$

where $N$ is the number of neurons. This result comes from statistical mechanics analysis.

### Capacity Analysis

```python
def analyze_network_capacity(n_neurons: int = 100, 
                             max_patterns: int = 30,
                             n_trials: int = 10):
    """
    Analyze how retrieval performance degrades with pattern count.
    
    Parameters
    ----------
    n_neurons : int
        Network size
    max_patterns : int
        Maximum patterns to test
    n_trials : int
        Trials per pattern count
    """
    pattern_counts = list(range(1, max_patterns + 1, 2))
    success_rates = []
    avg_overlaps = []
    
    theoretical_capacity = int(0.15 * n_neurons)
    
    for n_patterns in pattern_counts:
        trial_successes = []
        trial_overlaps = []
        
        for trial in range(n_trials):
            # Generate random patterns
            patterns = torch.sign(torch.randn(n_patterns, n_neurons))
            
            # Train network
            network = HopfieldNetwork(n_neurons)
            network.train_hebbian(patterns)
            
            # Test retrieval of each pattern
            for pattern in patterns:
                # Corrupt pattern (flip 10% of bits)
                corrupted = pattern.clone()
                n_flip = int(0.1 * n_neurons)
                flip_idx = torch.randperm(n_neurons)[:n_flip]
                corrupted[flip_idx] *= -1
                
                # Retrieve
                retrieved, _ = network.retrieve(corrupted, track_energy=False)
                
                # Compute overlap
                overlap = network.compute_overlap(retrieved, pattern)
                trial_overlaps.append(overlap)
                trial_successes.append(overlap > 0.9)
        
        success_rates.append(np.mean(trial_successes))
        avg_overlaps.append(np.mean(trial_overlaps))
        
        print(f"Patterns: {n_patterns:2d}, Success rate: {success_rates[-1]:.2f}, "
              f"Avg overlap: {avg_overlaps[-1]:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(pattern_counts, success_rates, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(theoretical_capacity, color='red', linestyle='--', 
                    label=f'Theoretical capacity (0.15N = {theoretical_capacity})')
    axes[0].set_xlabel('Number of Stored Patterns')
    axes[0].set_ylabel('Retrieval Success Rate')
    axes[0].set_title('Network Capacity: Success Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(pattern_counts, avg_overlaps, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(theoretical_capacity, color='red', linestyle='--', 
                    label='Theoretical capacity')
    axes[1].axhline(0.9, color='orange', linestyle=':', label='Good retrieval threshold')
    axes[1].set_xlabel('Number of Stored Patterns')
    axes[1].set_ylabel('Average Overlap')
    axes[1].set_title('Network Capacity: Pattern Overlap')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

analyze_network_capacity(n_neurons=100, max_patterns=30, n_trials=5)
```

## Spurious States

### Definition

Spurious states are stable states (local energy minima) that are not stored patterns. They represent "false memories" of the network.

### Types of Spurious States

1. **Mixture states**: Linear combinations of stored patterns
2. **Reversed patterns**: Negations of stored patterns ($-\mathbf{p}$)
3. **Spin glass states**: Unrelated to any stored pattern

### Analyzing Spurious States

```python
def find_spurious_states(network: HopfieldNetwork, 
                         stored_patterns: torch.Tensor,
                         n_random_starts: int = 100) -> List[torch.Tensor]:
    """
    Find spurious states by random initialization and convergence.
    """
    spurious = []
    
    for _ in range(n_random_starts):
        # Random initial state
        initial = torch.sign(torch.randn(network.n_neurons))
        
        # Converge to fixed point
        final, _ = network.retrieve(initial, track_energy=False)
        
        # Check if it's a stored pattern or its negative
        is_stored = False
        for pattern in stored_patterns:
            overlap = abs(network.compute_overlap(final, pattern))
            if overlap > 0.95:
                is_stored = True
                break
        
        if not is_stored:
            # Check if already found
            already_found = False
            for sp in spurious:
                if abs(network.compute_overlap(final, sp)) > 0.95:
                    already_found = True
                    break
            
            if not already_found:
                spurious.append(final.clone())
    
    return spurious
```

## Energy Landscape Visualization

For small networks, we can visualize the complete energy landscape:

```python
def visualize_energy_landscape():
    """
    Visualize energy landscape for small Hopfield network.
    """
    n_neurons = 6
    
    # Simple patterns
    p1 = torch.tensor([1, 1, 1, -1, -1, -1], dtype=torch.float32)
    p2 = torch.tensor([-1, -1, 1, 1, 1, -1], dtype=torch.float32)
    patterns = torch.stack([p1, p2])
    
    # Train network
    network = HopfieldNetwork(n_neurons)
    network.train_hebbian(patterns)
    
    # Enumerate all 2^6 = 64 states
    all_states = []
    all_energies = []
    
    for i in range(2**n_neurons):
        # Convert integer to binary state
        binary = format(i, f'0{n_neurons}b')
        state = torch.tensor([1.0 if b == '1' else -1.0 for b in binary])
        
        all_states.append(state)
        all_energies.append(network.energy(state).item())
    
    all_energies = np.array(all_energies)
    
    # Find local minima
    local_minima = []
    for i, state in enumerate(all_states):
        is_minimum = True
        current_E = all_energies[i]
        
        # Check all single-bit flips
        for j in range(n_neurons):
            neighbor = state.clone()
            neighbor[j] *= -1
            
            # Find neighbor index
            neighbor_binary = ''.join(['1' if s > 0 else '0' for s in neighbor])
            neighbor_idx = int(neighbor_binary, 2)
            
            if all_energies[neighbor_idx] < current_E:
                is_minimum = False
                break
        
        if is_minimum:
            local_minima.append((i, state, current_E))
    
    print(f"\nFound {len(local_minima)} local minima:")
    for idx, state, E in local_minima:
        # Check overlap with stored patterns
        overlap_1 = network.compute_overlap(state, p1)
        overlap_2 = network.compute_overlap(state, p2)
        
        pattern_type = "Spurious"
        if abs(overlap_1) > 0.9:
            pattern_type = "Pattern 1" if overlap_1 > 0 else "Pattern 1 (inverted)"
        elif abs(overlap_2) > 0.9:
            pattern_type = "Pattern 2" if overlap_2 > 0 else "Pattern 2 (inverted)"
        
        print(f"  State {idx}: E = {E:.3f}, Type: {pattern_type}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_idx = np.argsort(all_energies)
    ax.plot(all_energies[sorted_idx], 'b-', linewidth=1, alpha=0.7)
    
    # Mark stored patterns
    for i, pattern in enumerate(patterns):
        pattern_binary = ''.join(['1' if s > 0 else '0' for s in pattern])
        pattern_idx = int(pattern_binary, 2)
        sorted_pos = np.where(sorted_idx == pattern_idx)[0][0]
        ax.plot(sorted_pos, all_energies[pattern_idx], 'go', markersize=12,
               label=f'Pattern {i+1}' if i == 0 else None)
    
    # Mark local minima
    for idx, state, E in local_minima:
        sorted_pos = np.where(sorted_idx == idx)[0][0]
        ax.plot(sorted_pos, E, 'r^', markersize=10)
    
    ax.set_xlabel('State Index (sorted by energy)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Landscape of 6-Neuron Hopfield Network')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

visualize_energy_landscape()
```

## Connection to Modern Energy-Based Models

### From Hopfield to Modern EBMs

Hopfield networks established key principles used in modern EBMs:

| Hopfield Concept | Modern EBM Analog |
|-----------------|-------------------|
| Binary neurons | Continuous latent variables |
| Hebbian learning | Gradient-based learning |
| Asynchronous updates | Langevin dynamics |
| Energy minima | High-probability regions |

### Modern Hopfield Networks

Recent work has extended Hopfield networks with:
- **Continuous states**: Ramsauer et al. (2021)
- **Exponential capacity**: $\propto 2^{N/2}$ with polynomial patterns
- **Transformer connections**: Attention as associative memory

```python
class ModernHopfield(nn.Module):
    """
    Modern Hopfield Network with continuous states and exponential capacity.
    
    Based on Ramsauer et al., "Hopfield Networks is All You Need" (2021)
    """
    
    def __init__(self, pattern_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.patterns = None  # Stored patterns
    
    def store(self, patterns: torch.Tensor):
        """Store patterns (continuous values allowed)."""
        self.patterns = patterns  # Shape: (n_patterns, pattern_dim)
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve pattern using softmax attention.
        
        x_new = softmax(β * X^T * q)^T * X
        """
        # Attention scores
        scores = self.beta * torch.matmul(self.patterns, query)
        attention = torch.softmax(scores, dim=0)
        
        # Weighted combination of patterns
        return torch.matmul(attention, self.patterns)
```

## Key Takeaways

!!! success "Core Concepts"
    1. Hopfield networks use energy minimization for associative memory
    2. Hebbian learning stores patterns as energy minima
    3. Asynchronous updates guarantee energy descent
    4. Capacity is approximately $0.15N$ patterns
    5. Spurious states are unintended local minima

!!! warning "Limitations"
    - Limited capacity ($O(N)$ patterns for $N$ neurons)
    - Spurious states cause retrieval errors
    - Binary states limit expressiveness
    - Slow convergence for large networks

## Exercises

1. **Capacity Scaling**: Empirically verify the $0.15N$ capacity rule for networks of different sizes.

2. **Correlated Patterns**: What happens when stored patterns are correlated? Implement and analyze.

3. **Modern Hopfield**: Implement the continuous Hopfield network and compare retrieval accuracy with the classical version.

## References

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS.
- Amit, D. J., Gutfreund, H., & Sompolinsky, H. (1985). Storing infinite numbers of patterns in a spin-glass model of neural networks. Physical Review Letters.
- Ramsauer, H., et al. (2021). Hopfield Networks is All You Need. ICLR.
