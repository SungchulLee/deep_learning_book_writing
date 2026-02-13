#!/usr/bin/env python3
"""
================================================================================
NORMAL DISTRIBUTION MLE - Estimating Mean and Variance
================================================================================

DIFFICULTY: ⭐⭐ Medium (Level 2)

Learn to estimate both mean (μ) and variance (σ²) of a Gaussian distribution
simultaneously using MLE.

MLE Solutions:
μ̂ = (1/N) Σ xᵢ  (sample mean)
σ̂² = (1/N) Σ (xᵢ - μ̂)²  (sample variance)

This is a foundational example for understanding multivariate MLE!
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_normal_data(n_samples: int, true_mu: float, true_sigma: float, seed: int = 42):
    """Generate data from normal distribution"""
    torch.manual_seed(seed)
    data = torch.randn(n_samples) * true_sigma + true_mu
    return data


def compute_log_likelihood(data: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    """
    Compute log-likelihood for normal distribution.
    
    ℓ(μ, σ²) = -N/2 * log(2πσ²) - (1/2σ²) * Σ(xᵢ - μ)²
    """
    n = len(data)
    sigma = torch.clamp(sigma, min=1e-6)  # Avoid division by zero
    
    # Log-likelihood formula
    log_lik = (-n/2) * torch.log(2 * np.pi * sigma**2) - torch.sum((data - mu)**2) / (2 * sigma**2)
    return log_lik


def analytical_mle(data: torch.Tensor):
    """Compute MLE analytically"""
    mu_mle = torch.mean(data)
    sigma_mle = torch.std(data, unbiased=False)  # MLE uses biased estimator
    return mu_mle.item(), sigma_mle.item()


def gradient_based_mle(data: torch.Tensor, n_iterations: int = 1000):
    """
    Estimate parameters using gradient ascent on log-likelihood.
    
    We use log-parameterization for sigma to ensure positivity:
    σ = exp(log_sigma)
    """
    # Initialize parameters
    mu = torch.tensor(0.0, requires_grad=True)
    log_sigma = torch.tensor(0.0, requires_grad=True)  # σ = exp(log_sigma) > 0
    
    optimizer = torch.optim.Adam([mu, log_sigma], lr=0.01)
    history = []
    
    for i in range(n_iterations):
        sigma = torch.exp(log_sigma)  # Transform to ensure σ > 0
        
        # Compute log-likelihood
        log_lik = compute_log_likelihood(data, mu, sigma)
        
        # Loss is negative log-likelihood
        loss = -log_lik
        
        history.append((mu.item(), sigma.item(), loss.item()))
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print(f"   Iter {i+1}: μ={mu.item():.4f}, σ={sigma.item():.4f}, LL={log_lik.item():.2f}")
    
    return mu.item(), torch.exp(log_sigma).item(), history


def visualize_results(data, true_mu, true_sigma, analytical_mu, analytical_sigma,
                     gradient_mu, gradient_sigma, history):
    """Create visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Data histogram with fitted distributions
    ax = axes[0, 0]
    ax.hist(data.numpy(), bins=30, density=True, alpha=0.6, edgecolor='black', label='Data')
    
    x_range = np.linspace(data.min().item(), data.max().item(), 100)
    
    # True distribution
    from scipy.stats import norm
    ax.plot(x_range, norm.pdf(x_range, true_mu, true_sigma), 
           'g-', linewidth=2, label=f'True N({true_mu:.1f}, {true_sigma:.1f}²)')
    
    # MLE distribution
    ax.plot(x_range, norm.pdf(x_range, analytical_mu, analytical_sigma),
           'r--', linewidth=2, label=f'MLE N({analytical_mu:.2f}, {analytical_sigma:.2f}²)')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Data Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameter convergence (μ)
    ax = axes[0, 1]
    mus = [h[0] for h in history]
    ax.plot(mus, 'b-', linewidth=2, label='μ estimate')
    ax.axhline(true_mu, color='g', linestyle='--', label=f'True μ={true_mu}')
    ax.axhline(analytical_mu, color='r', linestyle='--', label=f'Analytical MLE')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('μ')
    ax.set_title('Mean Parameter Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Parameter convergence (σ)
    ax = axes[0, 2]
    sigmas = [h[1] for h in history]
    ax.plot(sigmas, 'b-', linewidth=2, label='σ estimate')
    ax.axhline(true_sigma, color='g', linestyle='--', label=f'True σ={true_sigma}')
    ax.axhline(analytical_sigma, color='r', linestyle='--', label=f'Analytical MLE')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('σ')
    ax.set_title('Std Dev Parameter Convergence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Log-likelihood surface
    ax = axes[1, 0]
    mu_range = np.linspace(true_mu - 2, true_mu + 2, 50)
    sigma_range = np.linspace(max(0.1, true_sigma - 1), true_sigma + 1, 50)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    
    LL = np.zeros_like(MU)
    for i in range(len(mu_range)):
        for j in range(len(sigma_range)):
            LL[j, i] = compute_log_likelihood(
                data, torch.tensor(MU[j, i]), torch.tensor(SIGMA[j, i])
            ).item()
    
    contour = ax.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.plot(analytical_mu, analytical_sigma, 'r*', markersize=20, label='MLE')
    ax.plot(true_mu, true_sigma, 'go', markersize=12, label='True')
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    ax.set_title('Log-Likelihood Surface', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot((data.numpy() - analytical_mu) / analytical_sigma, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Comparison table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Method', 'μ', 'σ'],
        ['True', f'{true_mu:.4f}', f'{true_sigma:.4f}'],
        ['Analytical', f'{analytical_mu:.4f}', f'{analytical_sigma:.4f}'],
        ['Gradient', f'{gradient_mu:.4f}', f'{gradient_sigma:.4f}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Results Comparison', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('normal_distribution_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'normal_distribution_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("NORMAL DISTRIBUTION MLE - Parameter Estimation")
    print("=" * 80)
    
    # Setup
    N_SAMPLES = 500
    TRUE_MU = 5.0
    TRUE_SIGMA = 2.0
    
    print(f"\n📋 Setup: N={N_SAMPLES}, True μ={TRUE_MU}, True σ={TRUE_SIGMA}")
    
    # Generate data
    print("\n🎲 Generating data...")
    data = generate_normal_data(N_SAMPLES, TRUE_MU, TRUE_SIGMA)
    print(f"   Sample mean: {data.mean():.4f}")
    print(f"   Sample std:  {data.std():.4f}")
    
    # Analytical MLE
    print("\n📐 Analytical MLE:")
    analytical_mu, analytical_sigma = analytical_mle(data)
    print(f"   μ̂ = {analytical_mu:.4f}")
    print(f"   σ̂ = {analytical_sigma:.4f}")
    
    # Gradient-based MLE
    print("\n🔄 Gradient-Based MLE:")
    gradient_mu, gradient_sigma, history = gradient_based_mle(data, n_iterations=1000)
    print(f"   Final μ̂ = {gradient_mu:.4f}")
    print(f"   Final σ̂ = {gradient_sigma:.4f}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    visualize_results(data, TRUE_MU, TRUE_SIGMA, analytical_mu, analytical_sigma,
                     gradient_mu, gradient_sigma, history)
    
    print("\n✅ Complete!")
    print("💡 Key takeaway: MLE for normal distribution gives sample mean and variance!")


if __name__ == "__main__":
    main()
