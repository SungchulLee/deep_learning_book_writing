"""
Boltzmann Machines: Stochastic Energy-Based Models
=================================================

This module introduces Boltzmann Machines (BMs), which extend Hopfield networks
to stochastic dynamics using thermal noise. BMs are generative models that can
learn probability distributions over binary data.

Learning Objectives:
-------------------
1. Understand stochastic vs deterministic neuron dynamics
2. Learn Boltzmann distribution and thermal equilibrium
3. Implement Gibbs sampling for inference
4. Understand the difference from Hopfield networks
5. Learn the challenges of training general BMs

Key Concepts:
------------
- Stochastic binary units: P(sᵢ=1) = σ(hᵢ/T)
- Boltzmann distribution at equilibrium
- Gibbs sampling for generating samples
- Visible and hidden units
- Intractable partition function

Historical Context:
------------------
- Hinton & Sejnowski (1985)
- Connection to statistical mechanics
- Inspired by Boltzmann-Gibbs distribution
- Foundation for Restricted Boltzmann Machines

Duration: 60-75 minutes
Prerequisites: Modules 01 (EBM Foundations), 02 (Hopfield Networks)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from tqdm import tqdm

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# Part 1: Boltzmann Machine Implementation
# ============================================================================

class BoltzmannMachine:
    """
    General Boltzmann Machine with stochastic binary units.
    
    Unlike Hopfield networks with deterministic updates, BMs use
    stochastic sampling based on the Boltzmann distribution:
    
    P(sᵢ = 1 | s₋ᵢ) = σ(hᵢ / T)
    
    where:
    - hᵢ = Σⱼ wᵢⱼ sⱼ + θᵢ (local field)
    - σ(x) = 1 / (1 + exp(-x)) (sigmoid)
    - T is the temperature
    
    At thermal equilibrium, the joint distribution is:
    P(s) = exp(-E(s)/T) / Z
    """
    
    def __init__(self, n_visible: int, n_hidden: int = 0, temperature: float = 1.0):
        """
        Initialize Boltzmann Machine.
        
        Parameters:
        -----------
        n_visible : int
            Number of visible units
        n_hidden : int
            Number of hidden units (0 for no hidden layer)
        temperature : float
            Temperature parameter for stochastic sampling
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden
        self.temperature = temperature
        
        # Initialize weights (symmetric, no self-connections)
        self.W = np.random.randn(self.n_total, self.n_total) * 0.01
        self.W = (self.W + self.W.T) / 2  # Ensure symmetry
        np.fill_diagonal(self.W, 0)
        
        # Initialize biases
        self.theta = np.zeros(self.n_total)
        
        print(f"Initialized Boltzmann Machine:")
        print(f"  Visible units: {n_visible}")
        print(f"  Hidden units: {n_hidden}")
        print(f"  Temperature: {temperature}")
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of a configuration.
        
        E(s) = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ - Σᵢ θᵢ sᵢ
        
        Parameters:
        -----------
        state : np.ndarray, shape (n_total,)
            Binary state (0 or 1)
        
        Returns:
        --------
        energy : float
            Energy value
        """
        # Convert {0,1} to {-1,+1} for energy computation
        s = 2 * state - 1
        
        # E = -½ sᵀ W s - θᵀ s
        return -0.5 * s @ self.W @ s - self.theta @ s
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_unit(self, unit_idx: int, state: np.ndarray) -> int:
        """
        Sample a single unit given the current state.
        
        Stochastic update rule:
        P(sᵢ = 1) = σ((Σⱼ wᵢⱼ sⱼ + θᵢ) / T)
        
        Parameters:
        -----------
        unit_idx : int
            Index of unit to sample
        state : np.ndarray
            Current state (0 or 1)
        
        Returns:
        --------
        new_value : int
            Sampled value (0 or 1)
        """
        # Compute local field
        # Convert state to {-1, +1} for computation
        s = 2 * state - 1
        h_i = self.W[unit_idx] @ s + self.theta[unit_idx]
        
        # Compute activation probability
        prob_on = self.sigmoid(h_i / self.temperature)
        
        # Sample from Bernoulli distribution
        return 1 if np.random.random() < prob_on else 0
    
    def gibbs_step(self, state: np.ndarray) -> np.ndarray:
        """
        Perform one step of Gibbs sampling.
        
        Updates all units in random order using stochastic sampling.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        
        Returns:
        --------
        new_state : np.ndarray
            Updated state
        """
        new_state = state.copy()
        
        # Update all units in random order
        update_order = np.random.permutation(self.n_total)
        
        for unit_idx in update_order:
            new_state[unit_idx] = self.sample_unit(unit_idx, new_state)
        
        return new_state
    
    def sample(self, n_steps: int = 1000, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Generate a sample by running Gibbs sampling.
        
        Parameters:
        -----------
        n_steps : int
            Number of Gibbs steps to run
        initial_state : np.ndarray, optional
            Initial state (random if None)
        
        Returns:
        --------
        state : np.ndarray
            Final state after n_steps
        """
        # Initialize state
        if initial_state is None:
            state = np.random.randint(0, 2, self.n_total)
        else:
            state = initial_state.copy()
        
        # Run Gibbs sampling
        for _ in range(n_steps):
            state = self.gibbs_step(state)
        
        return state


# ============================================================================
# Part 2: Comparing Deterministic vs Stochastic Dynamics
# ============================================================================

def compare_hopfield_vs_boltzmann():
    """
    Compare deterministic (Hopfield) vs stochastic (Boltzmann) dynamics.
    """
    print("\n" + "="*70)
    print("DETERMINISTIC VS STOCHASTIC DYNAMICS")
    print("="*70)
    
    # Create a simple pattern
    pattern = np.array([1, 1, 1, 0, 0, 0])
    n_units = len(pattern)
    
    print(f"\nPattern to store: {pattern}")
    
    # Train both networks with same weights (Hebbian)
    # Convert pattern to {-1, +1} for Hebbian rule
    pattern_pm = 2 * pattern - 1
    W = np.outer(pattern_pm, pattern_pm)
    np.fill_diagonal(W, 0)
    
    # Test with noisy pattern
    noisy_pattern = pattern.copy()
    noisy_pattern[2] = 0  # Flip one bit
    noisy_pattern[4] = 1  # Flip another bit
    
    print(f"Noisy pattern:   {noisy_pattern}")
    
    # Deterministic dynamics (Hopfield-style)
    print("\n" + "-"*70)
    print("DETERMINISTIC DYNAMICS (Hopfield-style)")
    print("-"*70)
    
    state_det = noisy_pattern.copy()
    energies_det = []
    
    for step in range(10):
        # Convert to {-1, +1}
        s = 2 * state_det - 1
        
        # Compute energy
        energy = -0.5 * s @ W @ s
        energies_det.append(energy)
        
        # Deterministic update
        h = W @ s
        state_det = (np.sign(h) + 1) // 2  # Convert back to {0, 1}
        
        print(f"Step {step}: State = {state_det}, Energy = {energy:.2f}")
    
    # Stochastic dynamics (Boltzmann)
    print("\n" + "-"*70)
    print("STOCHASTIC DYNAMICS (Boltzmann)")
    print("-"*70)
    
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, temp in enumerate(temperatures):
        print(f"\nTemperature T = {temp}:")
        
        # Track states over time
        states_over_time = []
        energies_stoch = []
        
        state_stoch = noisy_pattern.copy()
        
        for step in range(100):
            # Convert to {-1, +1} for energy
            s = 2 * state_stoch - 1
            energy = -0.5 * s @ W @ s
            
            states_over_time.append(state_stoch.copy())
            energies_stoch.append(energy)
            
            # Stochastic update
            for i in range(n_units):
                h_i = W[i] @ s
                prob = 1.0 / (1.0 + np.exp(-h_i / temp))
                state_stoch[i] = 1 if np.random.random() < prob else 0
        
        # Print some example states
        for step in [0, 10, 50, 99]:
            print(f"  Step {step}: {states_over_time[step]}, " + 
                  f"Energy = {energies_stoch[step]:.2f}")
        
        # Plot energy trajectory
        ax = axes[idx]
        ax.plot(energies_stoch, 'b-', alpha=0.6, linewidth=1)
        ax.plot(energies_det, 'r--', linewidth=2, label='Deterministic')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title(f'Temperature T = {temp}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotation about behavior
        if temp < 0.5:
            behavior = "Nearly deterministic"
        elif temp < 1.5:
            behavior = "Balanced exploration"
        else:
            behavior = "High randomness"
        
        ax.text(0.98, 0.97, behavior,
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/03_deterministic_vs_stochastic.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Comparison complete")
    print("\nKey differences:")
    print("  • Deterministic: Always follows energy gradient downhill")
    print("  • Stochastic: Can escape local minima via thermal fluctuations")
    print("  • Low T → Nearly deterministic behavior")
    print("  • High T → More random exploration")


# ============================================================================
# Part 3: Gibbs Sampling and Equilibrium Distribution
# ============================================================================

def demonstrate_gibbs_sampling():
    """
    Demonstrate Gibbs sampling converging to equilibrium distribution.
    """
    print("\n" + "="*70)
    print("GIBBS SAMPLING AND EQUILIBRIUM DISTRIBUTION")
    print("="*70)
    
    # Create simple 4-unit Boltzmann machine
    n_units = 4
    bm = BoltzmannMachine(n_visible=n_units, n_hidden=0, temperature=1.0)
    
    # Set up specific weights to create known distribution
    # This will favor certain patterns
    bm.W = np.array([
        [0.0,  1.5, -1.0,  0.5],
        [1.5,  0.0,  0.5, -1.0],
        [-1.0, 0.5,  0.0,  1.5],
        [0.5, -1.0,  1.5,  0.0]
    ])
    bm.theta = np.array([0.5, -0.5, 0.0, 0.5])
    
    print(f"\nNetwork: {n_units} units")
    print("Running Gibbs sampling to reach equilibrium...")
    
    # Run long Gibbs chain
    n_samples = 10000
    burn_in = 1000
    
    state = np.random.randint(0, 2, n_units)
    samples = []
    energies = []
    
    for step in tqdm(range(burn_in + n_samples), desc="Sampling"):
        state = bm.gibbs_step(state)
        
        if step >= burn_in:
            samples.append(state.copy())
            energies.append(bm.energy(state))
    
    samples = np.array(samples)
    energies = np.array(energies)
    
    print(f"✓ Collected {n_samples} samples after {burn_in} burn-in steps")
    
    # Analyze samples
    print("\n" + "-"*70)
    print("EMPIRICAL DISTRIBUTION FROM SAMPLES")
    print("-"*70)
    
    # Count frequency of each state
    state_counts = {}
    for sample in samples:
        state_tuple = tuple(sample)
        state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1
    
    # Compute empirical probabilities
    empirical_probs = {state: count / n_samples 
                      for state, count in state_counts.items()}
    
    # Compute theoretical probabilities (for small network)
    print("\nComparing empirical vs theoretical probabilities:")
    print("-"*70)
    
    all_states = []
    all_energies = []
    for i in range(2**n_units):
        binary = format(i, f'0{n_units}b')
        state = np.array([int(bit) for bit in binary])
        all_states.append(state)
        all_energies.append(bm.energy(state))
    
    # Compute partition function
    Z = np.sum(np.exp(-np.array(all_energies) / bm.temperature))
    
    # Compute theoretical probabilities
    theoretical_probs = {}
    for state, energy in zip(all_states, all_energies):
        prob = np.exp(-energy / bm.temperature) / Z
        theoretical_probs[tuple(state)] = prob
    
    # Compare top 10 most probable states
    print("\nTop 10 states by empirical probability:")
    sorted_empirical = sorted(empirical_probs.items(), 
                             key=lambda x: x[1], reverse=True)[:10]
    
    for rank, (state, emp_prob) in enumerate(sorted_empirical, 1):
        theo_prob = theoretical_probs.get(state, 0)
        energy = bm.energy(np.array(state))
        print(f"{rank:2d}. State {state}: " + 
              f"Empirical={emp_prob:.4f}, Theoretical={theo_prob:.4f}, " + 
              f"Energy={energy:.2f}")
    
    # Visualize distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energy distribution
    ax1 = axes[0]
    ax1.hist(energies, bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax1.set_xlabel('Energy', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Energy Distribution from Gibbs Sampling',
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Empirical vs Theoretical probabilities
    ax2 = axes[1]
    
    # Get common states
    common_states = set(empirical_probs.keys()) | set(theoretical_probs.keys())
    emp_vals = [empirical_probs.get(s, 0) for s in common_states]
    theo_vals = [theoretical_probs.get(s, 0) for s in common_states]
    
    ax2.scatter(theo_vals, emp_vals, alpha=0.6, s=50)
    
    # Plot y=x line
    max_prob = max(max(theo_vals), max(emp_vals))
    ax2.plot([0, max_prob], [0, max_prob], 'r--', linewidth=2, label='Perfect match')
    
    ax2.set_xlabel('Theoretical Probability', fontsize=12)
    ax2.set_ylabel('Empirical Probability', fontsize=12)
    ax2.set_title('Gibbs Sampling Accuracy',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Compute correlation
    correlation = np.corrcoef(theo_vals, emp_vals)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.4f}',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/03_gibbs_sampling.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Correlation between empirical and theoretical: {correlation:.4f}")
    print("\nKey insight: Gibbs sampling converges to the Boltzmann distribution")


# ============================================================================
# Part 4: Temperature Effects on Sampling
# ============================================================================

def analyze_temperature_effects():
    """
    Analyze how temperature affects the sampling distribution.
    """
    print("\n" + "="*70)
    print("TEMPERATURE EFFECTS ON SAMPLING")
    print("="*70)
    
    # Create simple BM
    n_units = 6
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, temp in enumerate(temperatures):
        print(f"\nTesting temperature T = {temp}")
        
        bm = BoltzmannMachine(n_visible=n_units, temperature=temp)
        
        # Use same weights for all temperatures
        if idx == 0:
            base_W = bm.W.copy()
            base_theta = bm.theta.copy()
        else:
            bm.W = base_W.copy()
            bm.theta = base_theta.copy()
        
        # Sample
        n_samples = 2000
        samples = []
        
        state = np.random.randint(0, 2, n_units)
        for _ in range(n_samples):
            state = bm.gibbs_step(state)
            samples.append(bm.energy(state))
        
        # Plot energy distribution
        ax = axes[idx]
        ax.hist(samples, bins=30, alpha=0.7, color='blue', 
               edgecolor='black', density=True)
        ax.set_xlabel('Energy', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'Temperature T = {temp}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Compute statistics
        mean_energy = np.mean(samples)
        std_energy = np.std(samples)
        
        ax.text(0.98, 0.97, f'Mean: {mean_energy:.2f}\nStd: {std_energy:.2f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        print(f"  Mean energy: {mean_energy:.2f}")
        print(f"  Std energy: {std_energy:.2f}")
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/03_temperature_effects.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Temperature analysis complete")
    print("\nObservations:")
    print("  • Low T → Concentrated at low energies")
    print("  • High T → More uniform across energies")
    print("  • Temperature controls exploration vs exploitation")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Execute all demonstrations for Boltzmann Machines.
    """
    print("="*70)
    print("BOLTZMANN MACHINES: STOCHASTIC ENERGY-BASED MODELS")
    print("="*70)
    print("\nThis module demonstrates:")
    print("  1. Stochastic vs deterministic dynamics")
    print("  2. Gibbs sampling for inference")
    print("  3. Equilibrium distributions")
    print("  4. Temperature effects on sampling")
    print("="*70)
    
    # Part 1: Deterministic vs Stochastic
    print("\n[Part 1] Deterministic vs Stochastic Dynamics")
    compare_hopfield_vs_boltzmann()
    
    # Part 2: Gibbs sampling
    print("\n[Part 2] Gibbs Sampling and Equilibrium")
    demonstrate_gibbs_sampling()
    
    # Part 3: Temperature effects
    print("\n[Part 3] Temperature Effects")
    analyze_temperature_effects()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Stochastic dynamics allow escape from local minima")
    print("  ✓ Gibbs sampling converges to Boltzmann distribution")
    print("  ✓ Temperature controls exploration vs exploitation")
    print("  ✓ Foundation for Restricted Boltzmann Machines")
    print("\nNext: 04_restricted_boltzmann_machines.py - Practical EBMs")


if __name__ == "__main__":
    main()
