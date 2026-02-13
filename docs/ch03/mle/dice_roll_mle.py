#!/usr/bin/env python3
"""
================================================================================
DICE ROLL MLE - Maximum Likelihood Estimation for a Loaded Die
================================================================================

DIFFICULTY: ⭐ Easy (Level 1)

LEARNING OBJECTIVES:
- Extend MLE to categorical distributions (more than 2 outcomes)
- Learn about multinomial/categorical distributions
- Understand constrained optimization (probabilities must sum to 1)
- Visualize likelihood in higher dimensions

PROBLEM STATEMENT:
You have a 6-sided die that might be loaded (unfair). You roll it N times
and observe the frequencies of each face. Your goal is to estimate the
probability of each face appearing.

MATHEMATICAL BACKGROUND:
- Each die roll follows a Categorical distribution with 6 outcomes
- Parameters: θ = (p₁, p₂, p₃, p₄, p₅, p₆) where Σpᵢ = 1
- For N rolls with counts (n₁, n₂, ..., n₆):
  
  Likelihood: L(θ) = ∏ pᵢ^nᵢ
  Log-likelihood: ℓ(θ) = Σ nᵢ * log(pᵢ)

The MLE solution is: pᵢ* = nᵢ/N (observed frequencies)

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


# ============================================================================
# PART 1: DATA GENERATION
# ============================================================================

def generate_dice_rolls(n_rolls: int, 
                       true_probs: torch.Tensor,
                       seed: int = 42) -> torch.Tensor:
    """
    Generate synthetic dice roll data.
    
    This simulates rolling a potentially loaded die multiple times.
    
    Parameters:
    -----------
    n_rolls : int
        Number of dice rolls to simulate
    true_probs : torch.Tensor
        True probability of each face (must sum to 1)
        Shape: (6,) for a 6-sided die
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    rolls : torch.Tensor
        Die outcomes (values from 0 to 5, representing faces 1-6)
        Shape: (n_rolls,)
    
    Example:
    --------
    >>> probs = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    >>> rolls = generate_dice_rolls(1000, probs)
    >>> print(f"Face 1 appeared {(rolls == 0).sum()} times")
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Validate that probabilities sum to 1
    assert torch.abs(true_probs.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"
    
    # Use torch.multinomial to sample from categorical distribution
    # We sample with replacement n_rolls times
    rolls = torch.multinomial(true_probs, n_rolls, replacement=True)
    
    return rolls


def count_faces(rolls: torch.Tensor, n_faces: int = 6) -> torch.Tensor:
    """
    Count the frequency of each face in the dice rolls.
    
    Parameters:
    -----------
    rolls : torch.Tensor
        Die roll outcomes (values 0 to n_faces-1)
    n_faces : int
        Number of faces on the die
        
    Returns:
    --------
    counts : torch.Tensor
        Count of each face
        Shape: (n_faces,)
    """
    counts = torch.zeros(n_faces, dtype=torch.float32)
    
    # Count each face
    for face in range(n_faces):
        counts[face] = (rolls == face).sum().float()
    
    return counts


# ============================================================================
# PART 2: LIKELIHOOD COMPUTATION
# ============================================================================

def compute_log_likelihood(counts: torch.Tensor, 
                          probs: torch.Tensor) -> torch.Tensor:
    """
    Compute log-likelihood for categorical distribution.
    
    Mathematical Formula:
    ℓ(θ) = Σ nᵢ * log(pᵢ)
    
    where nᵢ is the count of face i, and pᵢ is the probability of face i.
    
    Parameters:
    -----------
    counts : torch.Tensor
        Observed counts for each face
        Shape: (n_faces,)
    probs : torch.Tensor
        Probability parameters to evaluate
        Shape: (n_faces,)
        
    Returns:
    --------
    log_likelihood : torch.Tensor
        The log-likelihood value
    """
    epsilon = 1e-8  # Avoid log(0)
    
    # Ensure probabilities are valid
    probs = torch.clamp(probs, epsilon, 1.0)
    
    # Normalize to ensure they sum to 1
    probs = probs / probs.sum()
    
    # Compute log-likelihood: Σ counts[i] * log(probs[i])
    log_likelihood = torch.sum(counts * torch.log(probs))
    
    return log_likelihood


# ============================================================================
# PART 3: MLE ESTIMATION (ANALYTICAL)
# ============================================================================

def analytical_mle(counts: torch.Tensor) -> torch.Tensor:
    """
    Compute MLE analytically for categorical distribution.
    
    The solution is beautifully simple:
    p̂ᵢ = nᵢ / N
    
    where nᵢ is the count of face i and N is the total number of rolls.
    
    This is just the observed frequency of each face!
    
    Parameters:
    -----------
    counts : torch.Tensor
        Observed counts for each face
        
    Returns:
    --------
    probs_mle : torch.Tensor
        Maximum likelihood estimates of probabilities
    """
    # Total number of rolls
    total = counts.sum()
    
    # MLE is simply the observed proportion for each face
    probs_mle = counts / total
    
    return probs_mle


# ============================================================================
# PART 4: MLE ESTIMATION (GRADIENT-BASED WITH SOFTMAX)
# ============================================================================

def gradient_based_mle(counts: torch.Tensor,
                       learning_rate: float = 0.1,
                       n_iterations: int = 1000) -> Tuple[torch.Tensor, List]:
    """
    Compute MLE using gradient descent with softmax parameterization.
    
    To ensure probabilities sum to 1 and are positive, we use softmax:
    pᵢ = exp(θᵢ) / Σ exp(θⱼ)
    
    This automatically satisfies the constraints!
    
    Parameters:
    -----------
    counts : torch.Tensor
        Observed counts for each face
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of optimization iterations
        
    Returns:
    --------
    probs_mle : torch.Tensor
        Estimated probabilities
    history : List[torch.Tensor]
        History of probability estimates during optimization
    """
    n_faces = len(counts)
    
    # Initialize logits (unconstrained parameters)
    # We'll use softmax to convert these to probabilities
    logits = torch.zeros(n_faces, requires_grad=True)
    
    # Optimizer (using Adam for faster convergence)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    
    # Store history for visualization
    history = []
    
    # Optimization loop
    for iteration in range(n_iterations):
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=0)
        
        # Compute log-likelihood
        log_lik = compute_log_likelihood(counts, probs)
        
        # Loss is negative log-likelihood
        loss = -log_lik
        
        # Store current probabilities
        history.append(probs.detach().clone())
        
        # Optimization step
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Optional: Print progress
        if (iteration + 1) % 200 == 0:
            print(f"   Iteration {iteration + 1}/{n_iterations}, "
                  f"Log-Likelihood: {log_lik.item():.4f}")
    
    # Final probabilities
    with torch.no_grad():
        probs_mle = torch.softmax(logits, dim=0)
    
    return probs_mle, history


# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

def visualize_results(rolls: torch.Tensor,
                     counts: torch.Tensor,
                     true_probs: torch.Tensor,
                     analytical_probs: torch.Tensor,
                     gradient_probs: torch.Tensor,
                     history: List[torch.Tensor]):
    """
    Create comprehensive visualizations of the dice MLE results.
    """
    n_faces = len(counts)
    face_labels = [str(i+1) for i in range(n_faces)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # ========================================================================
    # Plot 1: Probability Comparison (Bar Chart)
    # ========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    x = np.arange(n_faces)
    width = 0.25
    
    # Plot bars for true, analytical, and gradient-based estimates
    bars1 = ax1.bar(x - width, true_probs.numpy(), width, 
                    label='True Probabilities', color='green', alpha=0.7)
    bars2 = ax1.bar(x, analytical_probs.numpy(), width,
                    label='Analytical MLE', color='red', alpha=0.7)
    bars3 = ax1.bar(x + width, gradient_probs.numpy(), width,
                    label='Gradient MLE', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Die Face', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Probability Estimates Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(face_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 2: Observed Frequencies (Histogram)
    # ========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    bars = ax2.bar(face_labels, counts.numpy(), 
                   color='skyblue', edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Die Face', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Observed Frequencies (N={len(rolls)})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 3: Error Comparison
    # ========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    analytical_error = torch.abs(analytical_probs - true_probs).numpy()
    gradient_error = torch.abs(gradient_probs - true_probs).numpy()
    
    x = np.arange(n_faces)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, analytical_error, width,
                    label='Analytical MLE Error', color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, gradient_error, width,
                    label='Gradient MLE Error', color='blue', alpha=0.7)
    
    ax3.set_xlabel('Die Face', fontsize=12)
    ax3.set_ylabel('Absolute Error', fontsize=12)
    ax3.set_title('Estimation Errors', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(face_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 4: Convergence History (All Faces)
    # ========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Convert history to numpy array
    history_array = torch.stack(history).numpy()
    
    # Plot convergence for each face
    for face in range(n_faces):
        ax4.plot(history_array[:, face], label=f'Face {face+1}', linewidth=2)
        # Mark true probability
        ax4.axhline(true_probs[face].item(), 
                   color=f'C{face}', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Estimated Probability', fontsize=12)
    ax4.set_title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 5: Chi-Square Goodness of Fit
    # ========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Expected counts under true distribution
    expected = true_probs * len(rolls)
    
    # Plot observed vs expected
    x = np.arange(n_faces)
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, counts.numpy(), width,
                    label='Observed', color='skyblue', edgecolor='black')
    bars2 = ax5.bar(x + width/2, expected.numpy(), width,
                    label='Expected (True)', color='orange', alpha=0.7)
    
    ax5.set_xlabel('Die Face', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Observed vs Expected Counts', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(face_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Compute chi-square statistic
    chi_square = torch.sum((counts - expected)**2 / expected).item()
    ax5.text(0.5, 0.95, f'χ² = {chi_square:.2f}',
             transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='top',
             horizontalalignment='center',
             fontsize=11)
    
    # ========================================================================
    # Plot 6: Log-Likelihood Surface (2D slice)
    # ========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Create a 2D slice: vary p1 and p2, keep others proportional
    p1_range = np.linspace(0.05, 0.5, 50)
    p2_range = np.linspace(0.05, 0.5, 50)
    P1, P2 = np.meshgrid(p1_range, p2_range)
    
    log_likelihoods = np.zeros_like(P1)
    
    for i in range(len(p1_range)):
        for j in range(len(p2_range)):
            p1, p2 = P1[j, i], P2[j, i]
            if p1 + p2 < 1:  # Valid probability range
                # Distribute remaining probability equally
                remaining = 1 - p1 - p2
                p_others = remaining / (n_faces - 2)
                probs = torch.tensor([p1, p2] + [p_others] * (n_faces - 2))
                log_likelihoods[j, i] = compute_log_likelihood(counts, probs).item()
            else:
                log_likelihoods[j, i] = -1e10  # Invalid region
    
    # Plot contours
    levels = np.linspace(log_likelihoods.max() - 50, log_likelihoods.max(), 15)
    contour = ax6.contour(P1, P2, log_likelihoods, levels=levels, cmap='viridis')
    ax6.clabel(contour, inline=True, fontsize=8)
    
    # Mark the MLE
    ax6.plot(analytical_probs[0].item(), analytical_probs[1].item(), 
            'r*', markersize=20, label='MLE')
    
    # Mark the true value
    ax6.plot(true_probs[0].item(), true_probs[1].item(),
            'go', markersize=12, label='True')
    
    ax6.set_xlabel('P(Face 1)', fontsize=12)
    ax6.set_ylabel('P(Face 2)', fontsize=12)
    ax6.set_title('Log-Likelihood Surface (2D Slice)', 
                  fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dice_roll_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'dice_roll_mle_results.png'")
    plt.show()


# ============================================================================
# PART 6: STATISTICAL TESTS
# ============================================================================

def chi_square_test(observed: torch.Tensor, 
                   expected: torch.Tensor) -> Tuple[float, float]:
    """
    Perform chi-square goodness of fit test.
    
    Tests whether observed frequencies significantly differ from expected.
    
    H₀: The die is fair (or follows the specified distribution)
    H₁: The die does not follow the specified distribution
    
    Parameters:
    -----------
    observed : torch.Tensor
        Observed counts
    expected : torch.Tensor
        Expected counts under null hypothesis
        
    Returns:
    --------
    chi_square_statistic : float
        The χ² test statistic
    p_value : float
        P-value (requires scipy, approximated here)
    """
    # Chi-square statistic: Σ (O - E)² / E
    chi_square_stat = torch.sum((observed - expected)**2 / expected).item()
    
    # Degrees of freedom
    df = len(observed) - 1
    
    print(f"\n📊 Chi-Square Goodness of Fit Test")
    print(f"   χ² statistic: {chi_square_stat:.4f}")
    print(f"   Degrees of freedom: {df}")
    print(f"   Critical value (α=0.05): ~{11.07:.2f}")  # For df=5
    
    if chi_square_stat < 11.07:
        print(f"   ✅ Fail to reject H₀: Data is consistent with expected distribution")
    else:
        print(f"   ❌ Reject H₀: Data suggests die may be loaded")
    
    return chi_square_stat, df


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the dice roll MLE example.
    """
    print("=" * 80)
    print("DICE ROLL MLE - Maximum Likelihood Estimation Tutorial")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Set up the problem
    # ========================================================================
    print("\n📋 STEP 1: Problem Setup")
    print("-" * 80)
    
    # Parameters (you can modify these!)
    N_ROLLS = 600           # Number of dice rolls
    SEED = 42              # Random seed
    
    # True probabilities for a loaded die
    # This die favors higher numbers!
    TRUE_PROBS = torch.tensor([0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    
    print(f"   • Number of rolls: {N_ROLLS}")
    print(f"   • True probabilities:")
    for i, p in enumerate(TRUE_PROBS):
        print(f"      Face {i+1}: {p:.2f}")
    
    # ========================================================================
    # Step 2: Generate data
    # ========================================================================
    print("\n🎲 STEP 2: Generating Dice Roll Data")
    print("-" * 80)
    
    rolls = generate_dice_rolls(N_ROLLS, TRUE_PROBS, SEED)
    counts = count_faces(rolls)
    
    print(f"   • Generated {N_ROLLS} dice rolls")
    print(f"   • Observed frequencies:")
    for i, count in enumerate(counts):
        proportion = count.item() / N_ROLLS
        print(f"      Face {i+1}: {int(count.item())} times ({proportion:.2%})")
    
    # ========================================================================
    # Step 3: Analytical MLE
    # ========================================================================
    print("\n📐 STEP 3: Analytical MLE")
    print("-" * 80)
    
    analytical_probs = analytical_mle(counts)
    
    print(f"   • MLE estimates:")
    for i, (true_p, est_p) in enumerate(zip(TRUE_PROBS, analytical_probs)):
        error = abs(est_p.item() - true_p.item())
        print(f"      Face {i+1}: {est_p:.4f} (true: {true_p:.4f}, error: {error:.4f})")
    
    # ========================================================================
    # Step 4: Gradient-Based MLE
    # ========================================================================
    print("\n🔄 STEP 4: Gradient-Based MLE (with Softmax)")
    print("-" * 80)
    
    gradient_probs, history = gradient_based_mle(
        counts,
        learning_rate=0.1,
        n_iterations=1000
    )
    
    print(f"\n   • Final estimates:")
    for i, (ana_p, grad_p) in enumerate(zip(analytical_probs, gradient_probs)):
        diff = abs(grad_p.item() - ana_p.item())
        print(f"      Face {i+1}: {grad_p:.4f} "
              f"(analytical: {ana_p:.4f}, diff: {diff:.6f})")
    
    # ========================================================================
    # Step 5: Statistical Testing
    # ========================================================================
    print("\n📊 STEP 5: Statistical Testing")
    print("-" * 80)
    
    expected = TRUE_PROBS * N_ROLLS
    chi_square_test(counts, expected)
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    print("\n📊 STEP 6: Creating Visualizations")
    print("-" * 80)
    
    visualize_results(rolls, counts, TRUE_PROBS, 
                     analytical_probs, gradient_probs, history)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    
    # Calculate total errors
    analytical_total_error = torch.sum(torch.abs(analytical_probs - TRUE_PROBS)).item()
    gradient_total_error = torch.sum(torch.abs(gradient_probs - TRUE_PROBS)).item()
    
    print(f"   Total absolute error (Analytical): {analytical_total_error:.4f}")
    print(f"   Total absolute error (Gradient):   {gradient_total_error:.4f}")
    print(f"   Methods agree within:              {abs(analytical_total_error - gradient_total_error):.6f}")
    print("=" * 80)
    
    # ========================================================================
    # Learning Points
    # ========================================================================
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. MLE for categorical: p̂ᵢ = observed frequency")
    print("   2. Softmax ensures probabilities sum to 1 (constraint satisfaction)")
    print("   3. Chi-square test can assess goodness of fit")
    print("   4. More categories → more parameters to estimate")
    print("   5. Gradient descent naturally handles constraints via parameterization")
    print("\n" + "=" * 80)


# ============================================================================
# EXERCISES FOR STUDENTS
# ============================================================================
"""
🎓 EXERCISES TO TRY:

1. EASY: Create a fair die (all probabilities = 1/6)
   - How do the estimates look with different numbers of rolls?
   - What's the minimum number of rolls for good estimates?

2. EASY: Modify the code to work with a 4-sided die (d4) or 20-sided die (d20)
   - Change n_faces and adjust the probability vectors
   - How does the number of faces affect estimation accuracy?

3. MEDIUM: Implement a likelihood ratio test
   - H₀: Die is fair (all p = 1/6)
   - H₁: Die is loaded (p ≠ 1/6)
   - Compare -2*log(LR) to chi-square distribution

4. MEDIUM: Add confidence intervals for each probability
   - Use the asymptotic normality of MLE
   - SE(p̂ᵢ) ≈ sqrt(pᵢ(1-pᵢ)/N)
   - Visualize with error bars

5. CHALLENGING: Implement Bayesian estimation with Dirichlet prior
   - Prior: Dir(α₁, ..., αₖ)
   - Posterior: Dir(α₁+n₁, ..., αₖ+nₖ)
   - Compare MAP estimate to MLE

6. CHALLENGING: Two dice problem
   - You have two dice with unknown probabilities
   - Estimate all 12 probabilities simultaneously
   - Can you identify which rolls came from which die?
"""


if __name__ == "__main__":
    main()
