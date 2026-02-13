#!/usr/bin/env python3
"""
================================================================================
COIN FLIP MLE - Maximum Likelihood Estimation for a Biased Coin
================================================================================

DIFFICULTY: ⭐ Easy (Level 1)

LEARNING OBJECTIVES:
- Understand the basic concept of Maximum Likelihood Estimation (MLE)
- Learn about the Bernoulli distribution
- Visualize the likelihood function
- Implement MLE using PyTorch

PROBLEM STATEMENT:
You have a coin that may be biased. You flip it N times and observe the results.
Your goal is to estimate the probability p of getting heads.

MATHEMATICAL BACKGROUND:
- Each coin flip follows a Bernoulli distribution: X ~ Bernoulli(p)
- Probability of heads: P(X=1) = p
- Probability of tails: P(X=0) = 1-p

For N independent flips with k heads:
- Likelihood: L(p) = p^k * (1-p)^(N-k)
- Log-likelihood: ℓ(p) = k*log(p) + (N-k)*log(1-p)

The MLE solution is: p* = k/N (the sample proportion)

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# ============================================================================
# PART 1: DATA GENERATION
# ============================================================================

def generate_coin_flips(n_flips: int, true_p: float, seed: int = 42) -> torch.Tensor:
    """
    Generate synthetic coin flip data.
    
    This function simulates flipping a biased coin multiple times.
    
    Parameters:
    -----------
    n_flips : int
        Number of coin flips to simulate
    true_p : float
        True probability of getting heads (0.0 to 1.0)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    flips : torch.Tensor
        Binary tensor where 1 = heads, 0 = tails
        Shape: (n_flips,)
    
    Example:
    --------
    >>> flips = generate_coin_flips(100, 0.7)
    >>> print(f"Got {flips.sum()} heads out of {len(flips)} flips")
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Generate random numbers between 0 and 1
    # If random number < true_p, it's heads (1), otherwise tails (0)
    random_values = torch.rand(n_flips)
    flips = (random_values < true_p).float()
    
    return flips


# ============================================================================
# PART 2: LIKELIHOOD COMPUTATION
# ============================================================================

def compute_log_likelihood(flips: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-likelihood for a given probability p.
    
    This is the key function in MLE! The log-likelihood tells us how probable
    our observed data is under a specific parameter value.
    
    Mathematical Formula:
    ℓ(p) = Σ[x_i * log(p) + (1-x_i) * log(1-p)]
    
    where x_i is 1 for heads and 0 for tails.
    
    Parameters:
    -----------
    flips : torch.Tensor
        Observed coin flips (1 for heads, 0 for tails)
    p : torch.Tensor
        Probability parameter to evaluate
        
    Returns:
    --------
    log_likelihood : torch.Tensor
        The log-likelihood value
        
    Note:
    -----
    We add a small epsilon (1e-8) to avoid log(0) which would be -infinity
    """
    epsilon = 1e-8  # Small constant to avoid numerical issues
    
    # Clamp p to avoid log(0)
    p = torch.clamp(p, epsilon, 1 - epsilon)
    
    # Compute log-likelihood using the Bernoulli formula
    # For each flip: if heads (1), contribute log(p); if tails (0), contribute log(1-p)
    log_likelihood = torch.sum(
        flips * torch.log(p) + (1 - flips) * torch.log(1 - p)
    )
    
    return log_likelihood


def compute_likelihood_curve(flips: torch.Tensor, 
                            p_values: np.ndarray) -> np.ndarray:
    """
    Compute likelihood for multiple p values (for visualization).
    
    This function evaluates the likelihood function across a range of
    possible parameter values so we can visualize how likelihood changes.
    
    Parameters:
    -----------
    flips : torch.Tensor
        Observed coin flips
    p_values : np.ndarray
        Array of p values to evaluate
        
    Returns:
    --------
    log_likelihoods : np.ndarray
        Log-likelihood values for each p
    """
    log_likelihoods = []
    
    # Evaluate log-likelihood for each candidate p value
    for p in p_values:
        p_tensor = torch.tensor(p, dtype=torch.float32)
        log_lik = compute_log_likelihood(flips, p_tensor)
        log_likelihoods.append(log_lik.item())
    
    return np.array(log_likelihoods)


# ============================================================================
# PART 3: MLE ESTIMATION (ANALYTICAL)
# ============================================================================

def analytical_mle(flips: torch.Tensor) -> float:
    """
    Compute the MLE analytically (closed-form solution).
    
    For the Bernoulli/Binomial distribution, we can derive the MLE solution
    mathematically:
    
    p* = (number of heads) / (total number of flips)
    
    This is the sample proportion, which is intuitive!
    
    Parameters:
    -----------
    flips : torch.Tensor
        Observed coin flips
        
    Returns:
    --------
    p_mle : float
        The maximum likelihood estimate of p
    """
    # Count the number of heads (1s)
    n_heads = torch.sum(flips).item()
    
    # Total number of flips
    n_total = len(flips)
    
    # MLE is simply the proportion
    p_mle = n_heads / n_total
    
    return p_mle


# ============================================================================
# PART 4: MLE ESTIMATION (NUMERICAL/GRADIENT-BASED)
# ============================================================================

def gradient_based_mle(flips: torch.Tensor, 
                       learning_rate: float = 0.01,
                       n_iterations: int = 1000,
                       initial_p: float = 0.5) -> Tuple[float, list]:
    """
    Compute MLE using gradient descent (optimization approach).
    
    This demonstrates how PyTorch can be used to find the MLE through
    optimization, similar to training neural networks!
    
    Why use gradient descent for MLE?
    - It's the same method used in deep learning
    - Works for complex models where analytical solutions don't exist
    - Educational: shows the connection between MLE and loss minimization
    
    Parameters:
    -----------
    flips : torch.Tensor
        Observed coin flips
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of optimization steps
    initial_p : float
        Starting guess for p
        
    Returns:
    --------
    p_mle : float
        The estimated probability
    history : list
        History of p values during optimization
    """
    # Initialize p as a parameter that requires gradients
    # We use a transformed parameterization to ensure 0 < p < 1
    # logit(p) = log(p / (1-p)), then p = sigmoid(logit)
    logit_p = torch.tensor(
        np.log(initial_p / (1 - initial_p)),  # Inverse sigmoid
        requires_grad=True
    )
    
    # Store the history for visualization
    history = []
    
    # Gradient descent loop
    for iteration in range(n_iterations):
        # Convert logit to probability using sigmoid
        p = torch.sigmoid(logit_p)
        
        # Compute log-likelihood
        log_lik = compute_log_likelihood(flips, p)
        
        # We want to MAXIMIZE log-likelihood, which is equivalent to
        # MINIMIZING negative log-likelihood (this is our "loss")
        loss = -log_lik
        
        # Store current p value
        history.append(p.item())
        
        # Compute gradients
        if logit_p.grad is not None:
            logit_p.grad.zero_()  # Reset gradients
        loss.backward()  # Compute gradients via backpropagation
        
        # Update parameters using gradient descent
        with torch.no_grad():
            logit_p -= learning_rate * logit_p.grad
        
    # Final estimate
    p_mle = torch.sigmoid(logit_p).item()
    
    return p_mle, history


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def visualize_results(flips: torch.Tensor, 
                     true_p: float,
                     p_analytical: float,
                     p_gradient: float,
                     gradient_history: list):
    """
    Create comprehensive visualizations of the MLE results.
    
    This function creates three plots:
    1. Likelihood curve showing where the maximum is
    2. Coin flip data visualization
    3. Gradient descent convergence
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 4))
    
    # ========================================================================
    # Plot 1: Likelihood Curve
    # ========================================================================
    ax1 = fig.add_subplot(1, 3, 1)
    
    # Compute likelihood for range of p values
    p_values = np.linspace(0.01, 0.99, 200)
    log_likelihoods = compute_likelihood_curve(flips, p_values)
    
    # Plot the likelihood curve
    ax1.plot(p_values, log_likelihoods, 'b-', linewidth=2, label='Log-Likelihood')
    
    # Mark the true value
    ax1.axvline(true_p, color='g', linestyle='--', linewidth=2, 
                label=f'True p = {true_p:.3f}')
    
    # Mark the MLE
    ax1.axvline(p_analytical, color='r', linestyle='-', linewidth=2,
                label=f'MLE p = {p_analytical:.3f}')
    
    # Find and mark the maximum
    max_idx = np.argmax(log_likelihoods)
    ax1.plot(p_values[max_idx], log_likelihoods[max_idx], 'ro', 
             markersize=10, label='Maximum')
    
    ax1.set_xlabel('Probability p', fontsize=12)
    ax1.set_ylabel('Log-Likelihood', fontsize=12)
    ax1.set_title('Likelihood Function', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Coin Flip Visualization
    # ========================================================================
    ax2 = fig.add_subplot(1, 3, 2)
    
    # Create a simple bar chart of outcomes
    n_heads = int(torch.sum(flips).item())
    n_tails = len(flips) - n_heads
    
    bars = ax2.bar(['Heads', 'Tails'], [n_heads, n_tails], 
                   color=['gold', 'silver'], edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Coin Flip Results (n={len(flips)})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add text with proportions
    ax2.text(0.5, 0.95, 
             f'Observed proportion: {p_analytical:.3f}\nTrue probability: {true_p:.3f}',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top',
             horizontalalignment='center',
             fontsize=10)
    
    # ========================================================================
    # Plot 3: Gradient Descent Convergence
    # ========================================================================
    ax3 = fig.add_subplot(1, 3, 3)
    
    # Plot convergence history
    ax3.plot(gradient_history, 'b-', linewidth=2, label='Gradient Descent')
    ax3.axhline(p_analytical, color='r', linestyle='--', linewidth=2,
                label=f'Analytical MLE = {p_analytical:.3f}')
    ax3.axhline(true_p, color='g', linestyle='--', linewidth=2,
                label=f'True p = {true_p:.3f}')
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Estimated p', fontsize=12)
    ax3.set_title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coin_flip_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'coin_flip_mle_results.png'")
    plt.show()


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the coin flip MLE example.
    """
    print("=" * 80)
    print("COIN FLIP MLE - Maximum Likelihood Estimation Tutorial")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Set up the problem
    # ========================================================================
    print("\n📋 STEP 1: Problem Setup")
    print("-" * 80)
    
    # Parameters (you can change these!)
    N_FLIPS = 100          # Number of coin flips
    TRUE_P = 0.7           # True probability of heads
    SEED = 42              # Random seed for reproducibility
    
    print(f"   • Number of flips: {N_FLIPS}")
    print(f"   • True probability of heads: {TRUE_P}")
    print(f"   • Random seed: {SEED}")
    
    # ========================================================================
    # Step 2: Generate data
    # ========================================================================
    print("\n🎲 STEP 2: Generating Coin Flip Data")
    print("-" * 80)
    
    flips = generate_coin_flips(N_FLIPS, TRUE_P, SEED)
    n_heads = int(torch.sum(flips).item())
    n_tails = N_FLIPS - n_heads
    
    print(f"   • Generated {N_FLIPS} coin flips")
    print(f"   • Observed {n_heads} heads ({n_heads/N_FLIPS:.2%})")
    print(f"   • Observed {n_tails} tails ({n_tails/N_FLIPS:.2%})")
    
    # ========================================================================
    # Step 3: Analytical MLE
    # ========================================================================
    print("\n📐 STEP 3: Analytical MLE (Closed-Form Solution)")
    print("-" * 80)
    
    p_analytical = analytical_mle(flips)
    print(f"   • MLE estimate: p = {p_analytical:.4f}")
    print(f"   • True value:   p = {TRUE_P:.4f}")
    print(f"   • Error:        {abs(p_analytical - TRUE_P):.4f}")
    
    # ========================================================================
    # Step 4: Gradient-Based MLE
    # ========================================================================
    print("\n🔄 STEP 4: Gradient-Based MLE (Optimization)")
    print("-" * 80)
    
    p_gradient, history = gradient_based_mle(
        flips, 
        learning_rate=0.1,
        n_iterations=500,
        initial_p=0.5
    )
    
    print(f"   • Initial guess:     p = 0.5000")
    print(f"   • After optimization: p = {p_gradient:.4f}")
    print(f"   • Analytical MLE:     p = {p_analytical:.4f}")
    print(f"   • Difference:         {abs(p_gradient - p_analytical):.6f}")
    
    # ========================================================================
    # Step 5: Visualization
    # ========================================================================
    print("\n📊 STEP 5: Creating Visualizations")
    print("-" * 80)
    
    visualize_results(flips, TRUE_P, p_analytical, p_gradient, history)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print(f"   True probability:        {TRUE_P:.4f}")
    print(f"   Observed proportion:     {p_analytical:.4f}")
    print(f"   Analytical MLE:          {p_analytical:.4f}")
    print(f"   Gradient-based MLE:      {p_gradient:.4f}")
    print(f"   Estimation error:        {abs(p_analytical - TRUE_P):.4f}")
    print("=" * 80)
    
    # ========================================================================
    # Learning Points
    # ========================================================================
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. MLE finds the parameter that makes observed data most likely")
    print("   2. For Bernoulli/Binomial: MLE = sample proportion (intuitive!)")
    print("   3. Analytical and gradient-based methods give the same answer")
    print("   4. Gradient descent is the same method used in deep learning")
    print("   5. More data → MLE gets closer to true value")
    print("\n" + "=" * 80)


# ============================================================================
# EXERCISES FOR STUDENTS
# ============================================================================
"""
🎓 EXERCISES TO TRY:

1. EASY: Change N_FLIPS and TRUE_P and observe how the estimates change
   - What happens with very few flips (N=10)?
   - What happens with many flips (N=10000)?

2. MEDIUM: Modify the code to track confidence intervals
   - The standard error is: SE = sqrt(p*(1-p)/N)
   - Add confidence bands to the visualization

3. MEDIUM: What if the coin is fair? Set TRUE_P = 0.5
   - Is the likelihood function symmetric?
   - How does this affect the estimation?

4. CHALLENGING: Implement MLE for multiple coins with different probabilities
   - Generate data from 3 different coins
   - Estimate all three probabilities simultaneously
   - How does the likelihood surface look in 3D?

5. CHALLENGING: Add a Bayesian prior to create Maximum A Posteriori (MAP) estimation
   - Assume a Beta(α, β) prior on p
   - How does this change the estimate?
   - What happens as N increases?
"""


if __name__ == "__main__":
    main()
