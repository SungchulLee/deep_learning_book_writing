"""
Energy-Based Models: Foundations and Basic Concepts
===================================================

This module introduces the fundamental concepts of Energy-Based Models (EBMs),
including energy functions, probability distributions, and partition functions.

Learning Objectives:
-------------------
1. Understand the relationship between energy and probability
2. Learn about partition functions and normalization
3. Visualize energy landscapes in 1D and 2D
4. Compute probabilities from energy functions
5. Understand the connection to statistical physics

Key Concepts:
------------
- Energy Function: E(x) - assigns energy to configurations
- Boltzmann Distribution: p(x) = exp(-E(x)) / Z
- Partition Function: Z = ∫ exp(-E(x)) dx
- Temperature parameter and its effects

Duration: 45-60 minutes
Prerequisites: Basic probability, calculus
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.special import logsumexp
import torch
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

plt.style.use('seaborn-v0_8-darkgrid')


# ============================================================================
# Part 1: Energy Functions and Boltzmann Distribution
# ============================================================================

def simple_energy_1d(x):
    """
    Simple 1D energy function with two wells (modes).
    
    This creates a double-well potential with minima at x=-2 and x=2.
    Low energy regions correspond to high probability regions.
    
    Mathematical form:
    E(x) = 0.5 * (x^2 - 4)^2
    
    Parameters:
    -----------
    x : float or array
        Input value(s)
    
    Returns:
    --------
    energy : float or array
        Energy value(s)
    """
    return 0.5 * (x**2 - 4)**2


def boltzmann_distribution(x, energy_func, temperature=1.0):
    """
    Compute probability using Boltzmann distribution.
    
    The Boltzmann distribution relates energy to probability:
    p(x) = exp(-E(x) / T) / Z
    
    where:
    - E(x) is the energy function
    - T is the temperature parameter
    - Z is the partition function (normalization constant)
    
    Lower energy → Higher probability
    Higher temperature → More uniform distribution
    
    Parameters:
    -----------
    x : array
        Input values
    energy_func : function
        Energy function E(x)
    temperature : float
        Temperature parameter (default: 1.0)
    
    Returns:
    --------
    prob : array
        Probability density (normalized)
    energy : array
        Energy values
    """
    # Compute energy for all x values
    energy = energy_func(x)
    
    # Apply Boltzmann formula: exp(-E/T)
    # Use log-sum-exp trick for numerical stability
    log_unnormalized = -energy / temperature
    
    # Compute partition function Z (normalization)
    # Z = ∫ exp(-E(x)/T) dx ≈ Σ exp(-E(x)/T) Δx
    dx = x[1] - x[0]  # assuming uniform spacing
    log_Z = logsumexp(log_unnormalized) + np.log(dx)
    
    # Compute normalized probability
    log_prob = log_unnormalized - log_Z
    prob = np.exp(log_prob)
    
    return prob, energy


def plot_energy_and_probability_1d():
    """
    Visualize the relationship between energy and probability in 1D.
    
    This demonstrates the key insight of EBMs:
    - Low energy regions have high probability
    - High energy regions have low probability
    - Temperature controls the sharpness of the distribution
    """
    # Create x values
    x = np.linspace(-4, 4, 1000)
    
    # Compute for different temperatures
    temperatures = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Energy function
    ax1 = axes[0]
    energy = simple_energy_1d(x)
    ax1.plot(x, energy, 'b-', linewidth=2, label='Energy E(x)')
    ax1.fill_between(x, energy, alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Energy E(x)', fontsize=12)
    ax1.set_title('Energy Function (Double-Well Potential)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Mark the minima (low energy = high probability)
    minima_x = [-2, 2]
    minima_energy = [simple_energy_1d(m) for m in minima_x]
    ax1.plot(minima_x, minima_energy, 'ro', markersize=10, 
             label='Energy minima (high probability)')
    ax1.legend(fontsize=11)
    
    # Plot 2: Boltzmann distributions for different temperatures
    ax2 = axes[1]
    
    for temp in temperatures:
        prob, _ = boltzmann_distribution(x, simple_energy_1d, temp)
        ax2.plot(x, prob, linewidth=2, label=f'T = {temp}')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probability Density p(x)', fontsize=12)
    ax2.set_title('Boltzmann Distribution p(x) = exp(-E(x)/T) / Z', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.text(0, ax2.get_ylim()[1] * 0.9, 
             'Lower T → Sharper peaks (more concentrated)\n'
             'Higher T → Flatter (more uniform)',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                                                  facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/01_energy_probability_1d.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Generated visualization: Energy function and Boltzmann distribution in 1D")
    print(f"  - Energy minima at x = {minima_x}")
    print(f"  - Corresponding to probability maxima")
    print(f"  - Temperature controls distribution sharpness")


# ============================================================================
# Part 2: 2D Energy Landscapes
# ============================================================================

def energy_2d_gaussian_mixture(x, y):
    """
    2D energy function representing a mixture of Gaussians.
    
    This creates an energy landscape with multiple wells (modes),
    corresponding to a mixture of Gaussian distributions.
    
    E(x,y) = -log(Σ w_k * N(μ_k, Σ_k))
    
    Parameters:
    -----------
    x, y : arrays
        2D coordinates (can be meshgrid outputs)
    
    Returns:
    --------
    energy : array
        Energy values at each point
    """
    # Define mixture components (centers, weights, covariances)
    means = [
        np.array([-2, -2]),
        np.array([2, 2]),
        np.array([-2, 2])
    ]
    
    weights = [0.4, 0.4, 0.2]
    covariances = [
        np.array([[0.5, 0.1], [0.1, 0.5]]),
        np.array([[0.3, -0.1], [-0.1, 0.4]]),
        np.array([[0.6, 0.0], [0.0, 0.3]])
    ]
    
    # Reshape for vectorized computation
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    
    # Compute mixture density
    density = np.zeros(len(points))
    
    for mean, weight, cov in zip(means, weights, covariances):
        # Compute Gaussian density for this component
        diff = points - mean
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        
        # Mahalanobis distance
        mahal = np.sum(diff @ cov_inv * diff, axis=1)
        
        # Gaussian density (unnormalized by 2π factor for simplicity)
        component_density = weight * np.exp(-0.5 * mahal) / np.sqrt(cov_det)
        density += component_density
    
    # Energy is negative log probability
    # E(x) = -log(p(x))
    energy = -np.log(density + 1e-10)  # add small constant for numerical stability
    
    return energy.reshape(x.shape)


def plot_energy_landscape_2d():
    """
    Visualize 2D energy landscape as both surface plot and contour plot.
    
    This demonstrates how energy functions define probability distributions
    in higher dimensions.
    """
    # Create 2D grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute energy
    E = energy_2d_gaussian_mixture(X, Y)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, E, cmap='viridis', alpha=0.9,
                           linewidth=0, antialiased=True)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('Energy E(x,y)', fontsize=11)
    ax1.set_title('3D Energy Landscape', fontsize=13, fontweight='bold')
    ax1.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax1, shrink=0.5)
    
    # Plot 2: Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, E, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    contourf = ax2.contourf(X, Y, E, levels=50, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Energy Contours (Low energy = High probability)', 
                  fontsize=13, fontweight='bold')
    fig.colorbar(contourf, ax=ax2)
    
    # Mark the energy minima (probability maxima)
    minima = [(-2, -2), (2, 2), (-2, 2)]
    for mx, my in minima:
        ax2.plot(mx, my, 'r*', markersize=15, markeredgecolor='white', 
                markeredgewidth=1.5)
    ax2.legend(['Energy minima'], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/01_energy_landscape_2d.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Generated visualization: 2D energy landscape")
    print("  - Valleys (low energy) = high probability regions")
    print("  - Peaks (high energy) = low probability regions")


# ============================================================================
# Part 3: Partition Function and Normalization
# ============================================================================

def compute_partition_function_1d():
    """
    Demonstrate the computation and role of the partition function.
    
    The partition function Z ensures that probabilities sum to 1:
    Z = ∫ exp(-E(x)) dx
    
    This integral is often intractable in high dimensions, which is
    a fundamental challenge in energy-based modeling.
    """
    print("\n" + "="*70)
    print("PARTITION FUNCTION COMPUTATION")
    print("="*70)
    
    # Define a simple 1D energy function
    def energy(x):
        return 0.25 * x**4 - x**2
    
    # Method 1: Numerical integration (only works in low dimensions)
    print("\nMethod 1: Numerical Integration")
    print("-" * 70)
    
    def integrand(x):
        return np.exp(-energy(x))
    
    # Integrate from -5 to 5 (assuming negligible probability beyond)
    Z_numerical, error = quad(integrand, -5, 5)
    print(f"Partition function Z = {Z_numerical:.6f}")
    print(f"Integration error: {error:.2e}")
    
    # Method 2: Monte Carlo approximation (scales to high dimensions)
    print("\nMethod 2: Monte Carlo Approximation")
    print("-" * 70)
    
    # Sample uniformly from the domain
    n_samples = 100000
    x_samples = np.random.uniform(-5, 5, n_samples)
    
    # Compute energy for all samples
    energy_samples = energy(x_samples)
    
    # Monte Carlo estimate: Z ≈ (b-a) * mean(exp(-E(x)))
    # where (b-a) is the domain width
    domain_width = 10  # from -5 to 5
    Z_mc = domain_width * np.mean(np.exp(-energy_samples))
    
    print(f"Partition function Z ≈ {Z_mc:.6f}")
    print(f"Relative error: {abs(Z_mc - Z_numerical) / Z_numerical * 100:.2f}%")
    print(f"Number of samples: {n_samples:,}")
    
    # Demonstrate normalization
    print("\nVerifying Normalization")
    print("-" * 70)
    
    x = np.linspace(-5, 5, 10000)
    prob, _ = boltzmann_distribution(x, energy, temperature=1.0)
    
    # Check if probabilities sum to 1 (numerical integration)
    dx = x[1] - x[0]
    total_prob = np.sum(prob) * dx
    
    print(f"∫ p(x) dx = {total_prob:.6f} (should be 1.0)")
    print(f"Error: {abs(total_prob - 1.0):.2e}")
    
    return Z_numerical, Z_mc


# ============================================================================
# Part 4: Temperature Parameter Effects
# ============================================================================

def demonstrate_temperature_effects():
    """
    Show how temperature affects the energy-based distribution.
    
    Temperature interpretation:
    - T → 0: Distribution concentrates at energy minima (sharp)
    - T → ∞: Distribution becomes uniform (flat)
    - T = 1: Standard Boltzmann distribution
    
    Physical interpretation: Temperature represents the "noise" or
    "randomness" in the system.
    """
    print("\n" + "="*70)
    print("TEMPERATURE PARAMETER EFFECTS")
    print("="*70)
    
    # Create energy function
    x = np.linspace(-4, 4, 1000)
    energy = simple_energy_1d(x)
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, temp in enumerate(temperatures):
        ax = axes[idx]
        
        # Compute Boltzmann distribution
        prob, _ = boltzmann_distribution(x, simple_energy_1d, temp)
        
        # Plot
        ax.plot(x, prob, 'b-', linewidth=2)
        ax.fill_between(x, prob, alpha=0.3)
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title(f'Temperature T = {temp}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Compute and display statistics
        # Expected value: E[x] = ∫ x p(x) dx
        dx = x[1] - x[0]
        mean = np.sum(x * prob) * dx
        
        # Variance: Var[x] = E[x²] - E[x]²
        second_moment = np.sum(x**2 * prob) * dx
        variance = second_moment - mean**2
        
        # Entropy: H = -∫ p(x) log(p(x)) dx
        entropy = -np.sum(prob * np.log(prob + 1e-10)) * dx
        
        # Add text box with statistics
        stats_text = f'Mean: {mean:.3f}\nVar: {variance:.3f}\nEntropy: {entropy:.3f}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        print(f"\nTemperature T = {temp}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Variance: {variance:.4f}")
        print(f"  Entropy: {entropy:.4f}")
    
    # Hide the extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/01_temperature_effects.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Key observations:")
    print("  - Lower T → Lower variance (more concentrated)")
    print("  - Higher T → Higher variance (more spread out)")
    print("  - Higher T → Higher entropy (more uniform)")


# ============================================================================
# Part 5: Connection to Maximum Likelihood Estimation
# ============================================================================

def energy_based_mle_example():
    """
    Demonstrate how learning in EBMs relates to maximum likelihood estimation.
    
    Given data x₁, ..., xₙ, we want to find parameters θ that maximize:
    L(θ) = Π p(xᵢ; θ) = Π exp(-E(xᵢ; θ)) / Z(θ)
    
    Log-likelihood:
    log L(θ) = Σ [-E(xᵢ; θ)] - n log Z(θ)
    
    Gradient:
    ∂ log L / ∂θ = -Σ ∂E(xᵢ; θ)/∂θ + n * E_model[∂E(x; θ)/∂θ]
                  = E_data[-∂E/∂θ] - E_model[-∂E/∂θ]
    
    This is the "positive phase" minus the "negative phase".
    """
    print("\n" + "="*70)
    print("ENERGY-BASED MAXIMUM LIKELIHOOD ESTIMATION")
    print("="*70)
    
    # Generate synthetic data from a known distribution
    np.random.seed(42)
    
    # True distribution: mixture of two Gaussians
    true_means = [-2, 2]
    true_weights = [0.6, 0.4]
    
    n_samples = 500
    component_samples = np.random.choice([0, 1], size=n_samples, p=true_weights)
    data = np.array([np.random.normal(true_means[c], 0.5) for c in component_samples])
    
    print(f"\nGenerated {n_samples} samples from true distribution")
    print(f"True means: {true_means}")
    print(f"True weights: {true_weights}")
    
    # Define parametric energy function
    # E(x; μ₁, μ₂) = -log(0.5 * N(x; μ₁, 1) + 0.5 * N(x; μ₂, 1))
    class SimpleEBM:
        def __init__(self, mu1=-1.5, mu2=1.5):
            self.mu1 = mu1
            self.mu2 = mu2
            self.sigma = 0.5
        
        def energy(self, x):
            """Compute energy for given x values."""
            # Mixture of Gaussians
            gaussian1 = np.exp(-0.5 * ((x - self.mu1) / self.sigma)**2)
            gaussian2 = np.exp(-0.5 * ((x - self.mu2) / self.sigma)**2)
            
            density = 0.5 * (gaussian1 + gaussian2) / (self.sigma * np.sqrt(2 * np.pi))
            energy = -np.log(density + 1e-10)
            
            return energy
        
        def probability(self, x):
            """Compute probability density."""
            energy = self.energy(x)
            # For this simple case, we can compute Z analytically
            # but in general, this is intractable
            
            # Numerical approximation of Z
            x_range = np.linspace(-6, 6, 1000)
            dx = x_range[1] - x_range[0]
            Z = np.sum(np.exp(-self.energy(x_range))) * dx
            
            return np.exp(-energy) / Z
    
    # Initialize model with suboptimal parameters
    model = SimpleEBM(mu1=-1.0, mu2=1.0)
    
    print(f"\nInitial model parameters:")
    print(f"  μ₁ = {model.mu1:.2f}, μ₂ = {model.mu2:.2f}")
    
    # Visualize initial fit
    x_plot = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data histogram vs initial model
    ax1 = axes[0]
    ax1.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='True data')
    ax1.plot(x_plot, model.probability(x_plot), 'r-', linewidth=2, 
            label='Initial model')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Data vs Initial Model', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy landscape
    ax2 = axes[1]
    ax2.plot(x_plot, model.energy(x_plot), 'g-', linewidth=2, label='Energy E(x)')
    ax2.scatter(data, model.energy(data), alpha=0.3, s=10, color='blue',
               label='Data points')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Energy Landscape and Data', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/50_energy_based_models/01_mle_initial.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualization saved: Initial model vs true data")
    print("\nKey concepts:")
    print("  1. Model assigns low energy to observed data")
    print("  2. Model assigns high energy to unlikely regions")
    print("  3. MLE adjusts parameters to increase data likelihood")
    print("  4. Challenge: Computing partition function Z(θ)")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Execute all demonstrations for EBM foundations.
    """
    print("="*70)
    print("ENERGY-BASED MODELS: FOUNDATIONS")
    print("="*70)
    print("\nThis module demonstrates:")
    print("  1. Energy functions and Boltzmann distributions")
    print("  2. Partition functions and normalization")
    print("  3. Temperature parameter effects")
    print("  4. Energy landscapes in 1D and 2D")
    print("  5. Connection to maximum likelihood estimation")
    print("="*70)
    
    # Part 1: Basic 1D demonstrations
    print("\n[Part 1] Energy and Probability in 1D")
    plot_energy_and_probability_1d()
    
    # Part 2: 2D energy landscapes
    print("\n[Part 2] 2D Energy Landscapes")
    plot_energy_landscape_2d()
    
    # Part 3: Partition function
    print("\n[Part 3] Partition Function Computation")
    Z_numerical, Z_mc = compute_partition_function_1d()
    
    # Part 4: Temperature effects
    print("\n[Part 4] Temperature Parameter Effects")
    demonstrate_temperature_effects()
    
    # Part 5: Connection to MLE
    print("\n[Part 5] Maximum Likelihood Estimation")
    energy_based_mle_example()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Energy functions define probability distributions")
    print("  ✓ Low energy = High probability")
    print("  ✓ Partition function ensures normalization (but is intractable)")
    print("  ✓ Temperature controls distribution sharpness")
    print("  ✓ Learning involves balancing data and model expectations")
    print("\nNext: 02_hopfield_networks.py - Classical energy-based memory")


if __name__ == "__main__":
    main()
