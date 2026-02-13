"""
Hopfield Networks: Energy-Based Associative Memory
=================================================

This module explores Hopfield Networks, one of the earliest and most elegant
energy-based models. They demonstrate how energy minimization can be used
for pattern storage and retrieval (associative memory).

Learning Objectives:
-------------------
1. Understand Hopfield network architecture and dynamics
2. Learn the energy function for Hopfield networks
3. Implement pattern storage using Hebbian learning
4. Perform pattern retrieval through energy minimization
5. Understand network capacity and spurious states

Key Concepts:
------------
- Recurrent network with symmetric weights
- Energy function: E = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ - Σᵢ θᵢ sᵢ
- Hebbian learning rule: wᵢⱼ = Σᵖ xᵢᵖ xⱼᵖ
- Asynchronous updates decrease energy monotonically
- Network capacity: ~0.15 * N patterns for N neurons

Historical Context:
------------------
- Introduced by John Hopfield (1982)
- Revival of neural network research
- Connection to statistical mechanics (Ising model)
- Inspired modern energy-based models

Duration: 60-75 minutes
Prerequisites: Module 01 (EBM Foundations), Linear algebra
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Tuple

# Set random seeds for reproducibility
np.random.seed(42)

plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# Part 1: Hopfield Network Class Implementation
# ============================================================================

class HopfieldNetwork:
    """
    Binary Hopfield Network with asynchronous updates.
    
    A Hopfield network is a fully connected recurrent network with:
    - Binary neurons: sᵢ ∈ {-1, +1}
    - Symmetric weights: wᵢⱼ = wⱼᵢ
    - No self-connections: wᵢᵢ = 0
    
    Energy Function:
    E(s) = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ - Σᵢ θᵢ sᵢ
    
    where:
    - s = (s₁, ..., sₙ) is the state vector
    - W = [wᵢⱼ] is the weight matrix
    - θ = (θ₁, ..., θₙ) are the thresholds (biases)
    
    Update Rule:
    sᵢ ← sign(Σⱼ wᵢⱼ sⱼ + θᵢ)
    
    Key Property:
    Asynchronous updates never increase energy (energy always decreases or stays same)
    """
    
    def __init__(self, n_neurons: int):
        """
        Initialize Hopfield network.
        
        Parameters:
        -----------
        n_neurons : int
            Number of neurons in the network
        """
        self.n_neurons = n_neurons
        
        # Initialize weight matrix (symmetric, no self-connections)
        self.W = np.zeros((n_neurons, n_neurons))
        
        # Initialize thresholds (biases) to zero
        self.theta = np.zeros(n_neurons)
        
        # Storage for energy history during retrieval
        self.energy_history = []
    
    def train(self, patterns: np.ndarray):
        """
        Store patterns using Hebbian learning rule.
        
        The Hebbian rule states: "Neurons that fire together, wire together"
        
        For binary patterns xᵖ ∈ {-1, +1}ⁿ, the weight matrix is:
        
        wᵢⱼ = (1/P) Σₚ xᵢᵖ xⱼᵖ    for i ≠ j
        wᵢᵢ = 0                   (no self-connections)
        
        where P is the number of patterns.
        
        Parameters:
        -----------
        patterns : np.ndarray, shape (n_patterns, n_neurons)
            Binary patterns to store (-1 or +1)
        """
        n_patterns = patterns.shape[0]
        
        print(f"Training Hopfield network with {n_patterns} patterns...")
        print(f"Network size: {self.n_neurons} neurons")
        
        # Reset weights
        self.W = np.zeros((self.n_neurons, self.n_neurons))
        
        # Apply Hebbian learning rule
        # W = (1/P) Σₚ xᵖ (xᵖ)ᵀ
        for pattern in patterns:
            # Outer product: pattern ⊗ pattern
            self.W += np.outer(pattern, pattern)
        
        # Normalize by number of patterns
        self.W /= n_patterns
        
        # Ensure no self-connections
        np.fill_diagonal(self.W, 0)
        
        # Verify symmetry
        assert np.allclose(self.W, self.W.T), "Weight matrix must be symmetric!"
        
        print(f"✓ Stored {n_patterns} patterns using Hebbian learning")
        print(f"  Weight matrix range: [{self.W.min():.3f}, {self.W.max():.3f}]")
        
        # Estimate network capacity
        capacity = 0.15 * self.n_neurons
        if n_patterns <= capacity:
            print(f"  Network capacity: ~{capacity:.0f} patterns (currently using {n_patterns})")
        else:
            print(f"  ⚠ Warning: Exceeding network capacity (~{capacity:.0f} patterns)")
            print(f"    This may lead to poor retrieval performance")
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of a given state.
        
        E(s) = -½ sᵀ W s - θᵀ s
        
        Lower energy indicates more stable configurations (stored patterns).
        
        Parameters:
        -----------
        state : np.ndarray, shape (n_neurons,)
            Network state (-1 or +1)
        
        Returns:
        --------
        energy : float
            Energy value (more negative = more stable)
        """
        # Energy = -½ sᵀ W s - θᵀ s
        quadratic_term = -0.5 * state @ self.W @ state
        linear_term = -self.theta @ state
        
        return quadratic_term + linear_term
    
    def update_neuron(self, state: np.ndarray, neuron_idx: int) -> np.ndarray:
        """
        Update a single neuron asynchronously.
        
        Update rule:
        sᵢ ← sign(hᵢ)    where hᵢ = Σⱼ wᵢⱼ sⱼ + θᵢ
        
        This update is guaranteed to not increase energy.
        
        Parameters:
        -----------
        state : np.ndarray
            Current network state
        neuron_idx : int
            Index of neuron to update
        
        Returns:
        --------
        new_state : np.ndarray
            Updated state
        """
        new_state = state.copy()
        
        # Compute local field (input to neuron i)
        h_i = self.W[neuron_idx] @ state + self.theta[neuron_idx]
        
        # Apply threshold (sign function)
        new_state[neuron_idx] = np.sign(h_i) if h_i != 0 else state[neuron_idx]
        
        return new_state
    
    def retrieve(self, initial_state: np.ndarray, max_iterations: int = 100,
                track_energy: bool = True) -> Tuple[np.ndarray, int]:
        """
        Retrieve a pattern from an initial state via energy minimization.
        
        The network performs asynchronous updates until:
        - Convergence to a fixed point (stable state)
        - Maximum iterations reached
        
        Parameters:
        -----------
        initial_state : np.ndarray
            Starting state (possibly corrupted pattern)
        max_iterations : int
            Maximum number of update cycles
        track_energy : bool
            Whether to track energy during retrieval
        
        Returns:
        --------
        final_state : np.ndarray
            Retrieved pattern (local energy minimum)
        n_iterations : int
            Number of iterations until convergence
        """
        state = initial_state.copy()
        
        if track_energy:
            self.energy_history = [self.energy(state)]
        
        for iteration in range(max_iterations):
            # Store previous state to check for convergence
            prev_state = state.copy()
            
            # Asynchronous updates (update neurons in random order)
            # This ensures energy never increases
            update_order = np.random.permutation(self.n_neurons)
            
            for neuron_idx in update_order:
                state = self.update_neuron(state, neuron_idx)
            
            # Track energy
            if track_energy:
                self.energy_history.append(self.energy(state))
            
            # Check for convergence (stable fixed point)
            if np.array_equal(state, prev_state):
                print(f"  ✓ Converged after {iteration + 1} iterations")
                return state, iteration + 1
        
        print(f"  ⚠ Reached maximum iterations ({max_iterations})")
        return state, max_iterations
    
    def pattern_overlap(self, state: np.ndarray, pattern: np.ndarray) -> float:
        """
        Compute overlap between state and pattern.
        
        Overlap = (1/N) Σᵢ sᵢ pᵢ
        
        Returns:
        - +1: Perfect match
        - -1: Perfect anti-match
        -  0: Orthogonal/uncorrelated
        
        Parameters:
        -----------
        state : np.ndarray
            Network state
        pattern : np.ndarray
            Reference pattern
        
        Returns:
        --------
        overlap : float
            Normalized overlap in [-1, 1]
        """
        return (state @ pattern) / self.n_neurons


# ============================================================================
# Part 2: Pattern Generation and Visualization
# ============================================================================

def generate_patterns(n_patterns: int = 3, pattern_size: int = 10) -> np.ndarray:
    """
    Generate random binary patterns.
    
    Parameters:
    -----------
    n_patterns : int
        Number of patterns to generate
    pattern_size : int
        Size of each pattern (will be reshaped to square if possible)
    
    Returns:
    --------
    patterns : np.ndarray, shape (n_patterns, pattern_size)
        Binary patterns in {-1, +1}
    """
    # Generate random binary patterns
    patterns = 2 * np.random.randint(0, 2, (n_patterns, pattern_size)) - 1
    
    return patterns.astype(float)


def create_letter_patterns() -> np.ndarray:
    """
    Create simple letter patterns (5x5 pixels).
    
    These are classic patterns used to demonstrate Hopfield networks.
    
    Returns:
    --------
    patterns : np.ndarray, shape (n_letters, 25)
        Binary letter patterns
    """
    # Define 5x5 patterns for letters T, L, X
    # +1 = black pixel, -1 = white pixel
    
    letter_T = np.array([
        [+1, +1, +1, +1, +1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1]
    ]).flatten()
    
    letter_L = np.array([
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, +1, +1, +1, +1]
    ]).flatten()
    
    letter_X = np.array([
        [+1, -1, -1, -1, +1],
        [-1, +1, -1, +1, -1],
        [-1, -1, +1, -1, -1],
        [-1, +1, -1, +1, -1],
        [+1, -1, -1, -1, +1]
    ]).flatten()
    
    return np.array([letter_T, letter_L, letter_X])


def visualize_patterns(patterns: np.ndarray, titles: List[str] = None,
                       pattern_shape: Tuple[int, int] = (5, 5)):
    """
    Visualize binary patterns as images.
    
    Parameters:
    -----------
    patterns : np.ndarray, shape (n_patterns, n_pixels)
        Binary patterns to visualize
    titles : list of str
        Titles for each pattern
    pattern_shape : tuple
        Shape to reshape patterns into (height, width)
    """
    n_patterns = patterns.shape[0]
    
    fig, axes = plt.subplots(1, n_patterns, figsize=(3*n_patterns, 3))
    if n_patterns == 1:
        axes = [axes]
    
    for idx, (pattern, ax) in enumerate(zip(patterns, axes)):
        # Reshape pattern to 2D image
        img = pattern.reshape(pattern_shape)
        
        # Display (use binary colormap)
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        ax.axis('off')
        
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def add_noise(pattern: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
    """
    Add noise to a pattern by flipping random bits.
    
    Parameters:
    -----------
    pattern : np.ndarray
        Original pattern
    noise_level : float
        Fraction of bits to flip (0 to 1)
    
    Returns:
    --------
    noisy_pattern : np.ndarray
        Pattern with noise
    """
    noisy = pattern.copy()
    n_flips = int(noise_level * len(pattern))
    
    # Randomly select indices to flip
    flip_indices = np.random.choice(len(pattern), n_flips, replace=False)
    
    # Flip selected bits
    noisy[flip_indices] *= -1
    
    return noisy


# ============================================================================
# Part 3: Pattern Retrieval Demonstration
# ============================================================================

def demonstrate_pattern_retrieval():
    """
    Demonstrate pattern storage and retrieval with letter patterns.
    """
    print("\n" + "="*70)
    print("HOPFIELD NETWORK: PATTERN RETRIEVAL DEMONSTRATION")
    print("="*70)
    
    # Create letter patterns
    patterns = create_letter_patterns()
    letter_names = ['T', 'L', 'X']
    
    print(f"\nCreated {len(patterns)} letter patterns (5×5 pixels each)")
    print(f"Total neurons: {patterns.shape[1]}")
    
    # Visualize original patterns
    fig1 = visualize_patterns(patterns, [f'Pattern {name}' for name in letter_names])
    plt.savefig('/home/claude/50_energy_based_models/02_original_patterns.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create and train Hopfield network
    network = HopfieldNetwork(n_neurons=25)
    network.train(patterns)
    
    # Test retrieval with noisy patterns
    print("\n" + "-"*70)
    print("RETRIEVAL TEST: Noisy Pattern → Stored Pattern")
    print("-"*70)
    
    # Test each pattern with different noise levels
    noise_levels = [0.2, 0.3, 0.4]
    
    for pattern_idx, (pattern, name) in enumerate(zip(patterns, letter_names)):
        print(f"\nTesting pattern '{name}':")
        
        # Create figure for this pattern's retrieval
        fig, axes = plt.subplots(len(noise_levels), 4, figsize=(12, 3*len(noise_levels)))
        
        for noise_idx, noise_level in enumerate(noise_levels):
            # Add noise
            noisy_pattern = add_noise(pattern, noise_level)
            
            # Compute initial overlap
            initial_overlap = network.pattern_overlap(noisy_pattern, pattern)
            initial_energy = network.energy(noisy_pattern)
            
            print(f"  Noise {noise_level*100:.0f}%: ", end="")
            print(f"Overlap = {initial_overlap:.3f}, Energy = {initial_energy:.1f}")
            
            # Retrieve pattern
            retrieved, n_iter = network.retrieve(noisy_pattern, max_iterations=100)
            
            # Compute final overlap and energy
            final_overlap = network.pattern_overlap(retrieved, pattern)
            final_energy = network.energy(retrieved)
            
            print(f"             After retrieval: Overlap = {final_overlap:.3f}, "
                  f"Energy = {final_energy:.1f}")
            
            # Visualize: Original → Noisy → Retrieved → Energy curve
            row_axes = axes[noise_idx]
            
            # Original pattern
            row_axes[0].imshow(pattern.reshape(5, 5), cmap='gray', vmin=-1, vmax=1)
            row_axes[0].set_title('Original', fontsize=10)
            row_axes[0].axis('off')
            
            # Noisy pattern
            row_axes[1].imshow(noisy_pattern.reshape(5, 5), cmap='gray', vmin=-1, vmax=1)
            row_axes[1].set_title(f'Noisy ({noise_level*100:.0f}%)', fontsize=10)
            row_axes[1].axis('off')
            
            # Retrieved pattern
            row_axes[2].imshow(retrieved.reshape(5, 5), cmap='gray', vmin=-1, vmax=1)
            row_axes[2].set_title('Retrieved', fontsize=10)
            row_axes[2].axis('off')
            
            # Energy during retrieval
            row_axes[3].plot(network.energy_history, 'b-', linewidth=2)
            row_axes[3].set_xlabel('Iteration', fontsize=9)
            row_axes[3].set_ylabel('Energy', fontsize=9)
            row_axes[3].set_title('Energy Minimization', fontsize=10)
            row_axes[3].grid(True, alpha=0.3)
            
            # Add overlap annotation
            row_axes[3].text(0.95, 0.95, f'Overlap: {final_overlap:.2f}',
                           transform=row_axes[3].transAxes,
                           ha='right', va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.suptitle(f"Pattern '{name}' Retrieval with Different Noise Levels",
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'/home/claude/50_energy_based_models/02_retrieval_{name}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n✓ Pattern retrieval demonstration complete")


# ============================================================================
# Part 4: Network Capacity and Spurious States
# ============================================================================

def analyze_network_capacity():
    """
    Analyze Hopfield network capacity by varying number of stored patterns.
    
    Theoretical capacity: ~0.15 * N patterns for N neurons
    Beyond this, retrieval performance degrades significantly.
    """
    print("\n" + "="*70)
    print("NETWORK CAPACITY ANALYSIS")
    print("="*70)
    
    n_neurons = 100
    pattern_counts = [5, 10, 15, 20, 25, 30]  # Up to 30% of neuron count
    n_trials = 10
    
    print(f"\nNetwork size: {n_neurons} neurons")
    print(f"Theoretical capacity: ~{0.15 * n_neurons:.0f} patterns")
    print(f"Testing with {pattern_counts} patterns\n")
    
    results = {
        'n_patterns': [],
        'retrieval_success': [],
        'avg_overlap': []
    }
    
    for n_patterns in pattern_counts:
        print(f"Testing with {n_patterns} patterns...")
        
        successes = []
        overlaps = []
        
        for trial in range(n_trials):
            # Generate random patterns
            patterns = generate_patterns(n_patterns, n_neurons)
            
            # Train network
            network = HopfieldNetwork(n_neurons)
            network.train(patterns)
            
            # Test retrieval with noisy versions of all patterns
            trial_overlaps = []
            
            for pattern in patterns:
                # Add 20% noise
                noisy = add_noise(pattern, noise_level=0.2)
                
                # Retrieve
                retrieved, _ = network.retrieve(noisy, max_iterations=50,
                                               track_energy=False)
                
                # Compute overlap with original
                overlap = network.pattern_overlap(retrieved, pattern)
                trial_overlaps.append(overlap)
            
            # Average overlap for this trial
            avg_overlap = np.mean(trial_overlaps)
            overlaps.append(avg_overlap)
            
            # Count successful retrievals (overlap > 0.9)
            success_rate = np.mean([o > 0.9 for o in trial_overlaps])
            successes.append(success_rate)
        
        # Store results
        results['n_patterns'].append(n_patterns)
        results['retrieval_success'].append(np.mean(successes))
        results['avg_overlap'].append(np.mean(overlaps))
        
        print(f"  Success rate: {np.mean(successes)*100:.1f}%")
        print(f"  Average overlap: {np.mean(overlaps):.3f}")
    
    # Plot capacity analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Success rate vs number of patterns
    ax1 = axes[0]
    ax1.plot(results['n_patterns'], np.array(results['retrieval_success']) * 100,
            'bo-', linewidth=2, markersize=8)
    ax1.axvline(0.15 * n_neurons, color='r', linestyle='--', linewidth=2,
               label='Theoretical capacity (0.15N)')
    ax1.set_xlabel('Number of Stored Patterns', fontsize=12)
    ax1.set_ylabel('Retrieval Success Rate (%)', fontsize=12)
    ax1.set_title('Network Capacity: Success Rate vs Pattern Count',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average overlap vs number of patterns
    ax2 = axes[1]
    ax2.plot(results['n_patterns'], results['avg_overlap'],
            'go-', linewidth=2, markersize=8)
    ax2.axvline(0.15 * n_neurons, color='r', linestyle='--', linewidth=2,
               label='Theoretical capacity (0.15N)')
    ax2.axhline(0.9, color='orange', linestyle=':', linewidth=2,
               label='Good retrieval threshold')
    ax2.set_xlabel('Number of Stored Patterns', fontsize=12)
    ax2.set_ylabel('Average Overlap', fontsize=12)
    ax2.set_title('Network Capacity: Overlap vs Pattern Count',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/02_capacity_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Capacity analysis complete")
    print("\nKey observations:")
    print("  - Performance degrades beyond ~0.15N patterns")
    print("  - Network can still partially retrieve beyond capacity")
    print("  - Spurious states (false memories) become more common")


# ============================================================================
# Part 5: Energy Landscape Visualization
# ============================================================================

def visualize_energy_landscape():
    """
    Visualize the energy landscape of a small Hopfield network.
    
    For a network with N neurons, there are 2^N possible states.
    We can visualize the energy of each state for small N.
    """
    print("\n" + "="*70)
    print("ENERGY LANDSCAPE VISUALIZATION")
    print("="*70)
    
    # Use small network for visualization (N=6)
    n_neurons = 6
    print(f"\nNetwork size: {n_neurons} neurons")
    print(f"Total possible states: {2**n_neurons} = {2**n_neurons}")
    
    # Create two simple patterns
    pattern1 = np.array([+1, +1, +1, -1, -1, -1])
    pattern2 = np.array([-1, -1, +1, +1, +1, -1])
    patterns = np.array([pattern1, pattern2])
    
    # Train network
    network = HopfieldNetwork(n_neurons)
    network.train(patterns)
    
    # Enumerate all possible states and compute energies
    all_states = []
    all_energies = []
    
    for i in range(2**n_neurons):
        # Convert integer to binary state
        binary = format(i, f'0{n_neurons}b')
        state = np.array([1 if bit == '1' else -1 for bit in binary])
        
        all_states.append(state)
        all_energies.append(network.energy(state))
    
    all_states = np.array(all_states)
    all_energies = np.array(all_energies)
    
    # Find stored patterns and their energies
    stored_energies = [network.energy(p) for p in patterns]
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energy histogram
    ax1 = axes[0]
    ax1.hist(all_energies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    for i, (pattern, energy) in enumerate(zip(patterns, stored_energies)):
        ax1.axvline(energy, color=f'C{i+1}', linestyle='--', linewidth=2,
                   label=f'Pattern {i+1}')
    ax1.set_xlabel('Energy', fontsize=12)
    ax1.set_ylabel('Number of States', fontsize=12)
    ax1.set_title(f'Energy Distribution ({2**n_neurons} states)',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy vs state index (sorted)
    ax2 = axes[1]
    sorted_indices = np.argsort(all_energies)
    ax2.plot(all_energies[sorted_indices], 'b-', linewidth=1, alpha=0.6)
    
    # Highlight stored patterns
    for i, (pattern, energy) in enumerate(zip(patterns, stored_energies)):
        # Find index of this pattern in sorted array
        pattern_idx = np.where(sorted_indices == np.where((all_states == pattern).all(axis=1))[0][0])[0][0]
        ax2.plot(pattern_idx, energy, 'o', markersize=12, color=f'C{i+1}',
                label=f'Pattern {i+1}')
    
    ax2.set_xlabel('State Index (sorted by energy)', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Landscape (sorted)',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/02_energy_landscape.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nEnergy statistics:")
    print(f"  Minimum energy: {all_energies.min():.2f} (global minimum)")
    print(f"  Maximum energy: {all_energies.max():.2f}")
    print(f"  Pattern 1 energy: {stored_energies[0]:.2f}")
    print(f"  Pattern 2 energy: {stored_energies[1]:.2f}")
    
    # Count local minima (stable states)
    # A state is a local minimum if updating any neuron increases energy
    local_minima = []
    for state in all_states:
        is_minimum = True
        current_energy = network.energy(state)
        
        for i in range(n_neurons):
            # Flip neuron i
            flipped = state.copy()
            flipped[i] *= -1
            
            # Check if energy increased
            if network.energy(flipped) <= current_energy:
                is_minimum = False
                break
        
        if is_minimum:
            local_minima.append(state)
    
    print(f"\nNumber of local minima (stable states): {len(local_minima)}")
    print(f"  Stored patterns: {len(patterns)}")
    print(f"  Spurious states: {len(local_minima) - len(patterns)}")
    print("\n✓ Energy landscape visualization complete")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Execute all demonstrations for Hopfield Networks.
    """
    print("="*70)
    print("HOPFIELD NETWORKS: ENERGY-BASED ASSOCIATIVE MEMORY")
    print("="*70)
    print("\nThis module demonstrates:")
    print("  1. Hopfield network architecture and dynamics")
    print("  2. Pattern storage using Hebbian learning")
    print("  3. Pattern retrieval through energy minimization")
    print("  4. Network capacity and limitations")
    print("  5. Energy landscape and spurious states")
    print("="*70)
    
    # Part 1: Pattern retrieval with letters
    print("\n[Part 1] Pattern Retrieval Demonstration")
    demonstrate_pattern_retrieval()
    
    # Part 2: Network capacity analysis
    print("\n[Part 2] Network Capacity Analysis")
    analyze_network_capacity()
    
    # Part 3: Energy landscape
    print("\n[Part 3] Energy Landscape Visualization")
    visualize_energy_landscape()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Hopfield networks store patterns in connection weights")
    print("  ✓ Energy decreases monotonically during asynchronous updates")
    print("  ✓ Stored patterns are local energy minima")
    print("  ✓ Network capacity is ~0.15N patterns")
    print("  ✓ Spurious states (false memories) can emerge")
    print("\nNext: 03_boltzmann_machines.py - Stochastic energy-based models")


if __name__ == "__main__":
    main()
