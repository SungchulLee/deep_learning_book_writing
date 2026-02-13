#!/usr/bin/env python3
"""
================================================================================
LINEAR REGRESSION MLE - Connection Between MLE and Least Squares
================================================================================

DIFFICULTY: ⭐⭐ Medium (Level 2)

LEARNING OBJECTIVES:
- Understand how linear regression relates to MLE
- See the connection between MLE and least squares
- Learn about Gaussian (Normal) noise assumptions
- Implement regression using PyTorch

PROBLEM STATEMENT:
Given data points (x, y), find the best-fit line y = α + βx that explains
the relationship. We'll see that minimizing squared error is equivalent to
maximum likelihood under Gaussian noise assumptions!

MATHEMATICAL BACKGROUND:
Linear model: y = α + βx + ε, where ε ~ N(0, σ²)

This means: y | x ~ N(α + βx, σ²)

Likelihood for N observations:
L(α, β, σ | data) = ∏ (1/√(2πσ²)) * exp(-(yᵢ - α - βxᵢ)² / (2σ²))

Log-likelihood:
ℓ(α, β, σ) = -N/2 * log(2πσ²) - (1/2σ²) * Σ(yᵢ - α - βxᵢ)²

KEY INSIGHT: Maximizing log-likelihood is equivalent to minimizing Σ(yᵢ - α - βxᵢ)²
This is the mean squared error (MSE)!

MLE Solutions (when σ is known or estimated separately):
- β̂ = Cov(X,Y) / Var(X)
- α̂ = ȳ - β̂x̄

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from mpl_toolkits.mplot3d import Axes3D


# ============================================================================
# PART 1: DATA GENERATION
# ============================================================================

def generate_linear_data(n_samples: int = 100,
                        true_alpha: float = 2.0,
                        true_beta: float = 3.0,
                        noise_std: float = 1.0,
                        seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic linear regression data with Gaussian noise.
    
    Model: y = α + βx + ε, where ε ~ N(0, σ²)
    
    Parameters:
    -----------
    n_samples : int
        Number of data points
    true_alpha : float
        True intercept
    true_beta : float
        True slope
    noise_std : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed
        
    Returns:
    --------
    x : torch.Tensor
        Input features, shape (n_samples, 1)
    y : torch.Tensor
        Output values, shape (n_samples, 1)
    """
    torch.manual_seed(seed)
    
    # Generate x values uniformly in [0, 10]
    x = torch.rand(n_samples, 1) * 10
    
    # Generate y values: y = α + βx + noise
    noise = torch.randn(n_samples, 1) * noise_std
    y = true_alpha + true_beta * x + noise
    
    return x, y


# ============================================================================
# PART 2: LIKELIHOOD COMPUTATION
# ============================================================================

def compute_log_likelihood(x: torch.Tensor,
                          y: torch.Tensor,
                          alpha: torch.Tensor,
                          beta: torch.Tensor,
                          sigma: float) -> torch.Tensor:
    """
    Compute log-likelihood for linear regression with Gaussian noise.
    
    Log-likelihood:
    ℓ(α, β, σ) = -N/2 * log(2πσ²) - (1/2σ²) * Σ(yᵢ - α - βxᵢ)²
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Data points
    alpha, beta : torch.Tensor
        Model parameters (intercept and slope)
    sigma : float
        Standard deviation of noise
        
    Returns:
    --------
    log_likelihood : torch.Tensor
        The log-likelihood value
    """
    n = len(x)
    
    # Predicted values
    y_pred = alpha + beta * x
    
    # Residuals (errors)
    residuals = y - y_pred
    
    # Sum of squared errors
    sse = torch.sum(residuals ** 2)
    
    # Log-likelihood formula
    # ℓ = -N/2 * log(2πσ²) - SSE/(2σ²)
    log_likelihood = (-n/2) * torch.log(torch.tensor(2 * np.pi * sigma**2)) - sse / (2 * sigma**2)
    
    return log_likelihood


def compute_mse(x: torch.Tensor,
               y: torch.Tensor,
               alpha: torch.Tensor,
               beta: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).
    
    MSE = (1/N) * Σ(yᵢ - α - βxᵢ)²
    
    This is equivalent to negative log-likelihood (up to constants)!
    """
    y_pred = alpha + beta * x
    mse = torch.mean((y - y_pred) ** 2)
    return mse


# ============================================================================
# PART 3: ANALYTICAL MLE SOLUTION
# ============================================================================

def analytical_mle(x: torch.Tensor, 
                  y: torch.Tensor) -> Tuple[float, float]:
    """
    Compute MLE analytically using closed-form formulas.
    
    For linear regression:
    β̂ = Cov(X,Y) / Var(X)
    α̂ = ȳ - β̂x̄
    
    These formulas come from setting derivatives of log-likelihood to zero.
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Data points
        
    Returns:
    --------
    alpha_mle, beta_mle : float
        MLE estimates of parameters
    """
    # Flatten tensors for computation
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Compute means
    x_mean = torch.mean(x_flat)
    y_mean = torch.mean(y_flat)
    
    # Compute covariance and variance
    # Cov(X,Y) = E[(X - E[X])(Y - E[Y])]
    cov_xy = torch.mean((x_flat - x_mean) * (y_flat - y_mean))
    
    # Var(X) = E[(X - E[X])²]
    var_x = torch.mean((x_flat - x_mean) ** 2)
    
    # MLE estimates
    beta_mle = (cov_xy / var_x).item()
    alpha_mle = (y_mean - beta_mle * x_mean).item()
    
    return alpha_mle, beta_mle


def estimate_sigma(x: torch.Tensor,
                  y: torch.Tensor,
                  alpha: float,
                  beta: float) -> float:
    """
    Estimate the noise standard deviation σ.
    
    MLE for σ²:
    σ̂² = (1/N) * Σ(yᵢ - α̂ - β̂xᵢ)²
    
    This is the mean squared error of the residuals.
    """
    y_pred = alpha + beta * x
    residuals = y - y_pred
    sigma_squared = torch.mean(residuals ** 2).item()
    sigma = np.sqrt(sigma_squared)
    return sigma


# ============================================================================
# PART 4: GRADIENT-BASED MLE
# ============================================================================

def gradient_based_mle(x: torch.Tensor,
                      y: torch.Tensor,
                      learning_rate: float = 0.01,
                      n_iterations: int = 1000) -> Tuple[float, float, List]:
    """
    Compute MLE using gradient descent.
    
    This is the approach used in modern deep learning!
    We minimize the loss (negative log-likelihood / MSE).
    
    Parameters:
    -----------
    x, y : torch.Tensor
        Data points
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Number of iterations
        
    Returns:
    --------
    alpha_mle, beta_mle : float
        Estimated parameters
    history : List
        History of (alpha, beta, loss) during training
    """
    # Initialize parameters (with requires_grad=True for autograd)
    alpha = torch.tensor(0.0, requires_grad=True)
    beta = torch.tensor(0.0, requires_grad=True)
    
    # Optimizer
    optimizer = torch.optim.SGD([alpha, beta], lr=learning_rate)
    
    # Training history
    history = []
    
    # Training loop
    for iteration in range(n_iterations):
        # Forward pass: compute predictions
        y_pred = alpha + beta * x
        
        # Compute loss (MSE, which is equivalent to negative log-likelihood)
        loss = torch.mean((y - y_pred) ** 2)
        
        # Store history
        history.append((alpha.item(), beta.item(), loss.item()))
        
        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if (iteration + 1) % 200 == 0:
            print(f"   Iteration {iteration + 1}/{n_iterations}, "
                  f"Loss: {loss.item():.4f}, "
                  f"α: {alpha.item():.4f}, β: {beta.item():.4f}")
    
    return alpha.item(), beta.item(), history


# ============================================================================
# PART 5: PYTORCH NN MODULE APPROACH
# ============================================================================

class LinearRegressionModel(nn.Module):
    """
    Linear regression implemented as a PyTorch nn.Module.
    
    This demonstrates how linear regression is just a simple neural network!
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # A linear layer with 1 input and 1 output
        # This automatically creates weight (β) and bias (α) parameters
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        """Forward pass: compute predictions"""
        return self.linear(x)


def pytorch_nn_mle(x: torch.Tensor,
                   y: torch.Tensor,
                   learning_rate: float = 0.01,
                   n_iterations: int = 1000) -> Tuple[float, float, List]:
    """
    Compute MLE using PyTorch's nn.Module (the "official" way).
    
    This is how you'd typically implement regression in PyTorch.
    """
    # Create model
    model = LinearRegressionModel()
    
    # Loss function (MSE)
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training history
    history = []
    
    # Training loop
    for iteration in range(n_iterations):
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        # Get current parameters
        alpha = model.linear.bias.item()
        beta = model.linear.weight.item()
        
        # Store history
        history.append((alpha, beta, loss.item()))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress
        if (iteration + 1) % 200 == 0:
            print(f"   Iteration {iteration + 1}/{n_iterations}, "
                  f"Loss: {loss.item():.4f}, "
                  f"α: {alpha:.4f}, β: {beta:.4f}")
    
    # Extract final parameters
    alpha_final = model.linear.bias.item()
    beta_final = model.linear.weight.item()
    
    return alpha_final, beta_final, history


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def visualize_results(x: torch.Tensor,
                     y: torch.Tensor,
                     true_alpha: float,
                     true_beta: float,
                     analytical_alpha: float,
                     analytical_beta: float,
                     gradient_alpha: float,
                     gradient_beta: float,
                     pytorch_alpha: float,
                     pytorch_beta: float,
                     gradient_history: List,
                     pytorch_history: List):
    """
    Create comprehensive visualizations of linear regression MLE.
    """
    # Convert to numpy for plotting
    x_np = x.numpy().flatten()
    y_np = y.numpy().flatten()
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # ========================================================================
    # Plot 1: Data and Fitted Lines
    # ========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Plot data points
    ax1.scatter(x_np, y_np, alpha=0.5, s=30, label='Data', color='blue')
    
    # Plot true line
    x_line = np.linspace(0, 10, 100)
    y_true = true_alpha + true_beta * x_line
    ax1.plot(x_line, y_true, 'g--', linewidth=2, label=f'True: y={true_alpha:.1f}+{true_beta:.1f}x')
    
    # Plot fitted lines
    y_analytical = analytical_alpha + analytical_beta * x_line
    ax1.plot(x_line, y_analytical, 'r-', linewidth=2, 
            label=f'Analytical: y={analytical_alpha:.2f}+{analytical_beta:.2f}x')
    
    y_gradient = gradient_alpha + gradient_beta * x_line
    ax1.plot(x_line, y_gradient, 'b:', linewidth=2,
            label=f'Gradient: y={gradient_alpha:.2f}+{gradient_beta:.2f}x')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Data and Fitted Lines', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 2: Residuals
    # ========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Compute residuals for analytical MLE
    y_pred_analytical = analytical_alpha + analytical_beta * x
    residuals = (y - y_pred_analytical).numpy().flatten()
    
    # Residual plot
    ax2.scatter(x_np, residuals, alpha=0.5, s=30)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Residuals (y - ŷ)', fontsize=12)
    ax2.set_title('Residual Plot (Analytical MLE)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add histogram of residuals
    ax2_hist = ax2.inset_axes([0.6, 0.6, 0.35, 0.35])
    ax2_hist.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax2_hist.set_title('Residuals', fontsize=8)
    ax2_hist.tick_params(labelsize=7)
    
    # ========================================================================
    # Plot 3: Parameter Convergence (Gradient Descent)
    # ========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    # Extract history
    alphas = [h[0] for h in gradient_history]
    betas = [h[1] for h in gradient_history]
    iterations = list(range(len(alphas)))
    
    ax3_alpha = ax3
    ax3_beta = ax3.twinx()
    
    # Plot alpha convergence
    line1 = ax3_alpha.plot(iterations, alphas, 'b-', linewidth=2, label='α (intercept)')
    ax3_alpha.axhline(true_alpha, color='b', linestyle='--', alpha=0.5, label='True α')
    
    # Plot beta convergence
    line2 = ax3_beta.plot(iterations, betas, 'r-', linewidth=2, label='β (slope)')
    ax3_beta.axhline(true_beta, color='r', linestyle='--', alpha=0.5, label='True β')
    
    ax3_alpha.set_xlabel('Iteration', fontsize=12)
    ax3_alpha.set_ylabel('α (Intercept)', fontsize=12, color='b')
    ax3_beta.set_ylabel('β (Slope)', fontsize=12, color='r')
    ax3_alpha.set_title('Parameter Convergence', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3_alpha.legend(lines, labels, loc='best', fontsize=9)
    
    ax3_alpha.grid(True, alpha=0.3)
    ax3_alpha.tick_params(axis='y', labelcolor='b')
    ax3_beta.tick_params(axis='y', labelcolor='r')
    
    # ========================================================================
    # Plot 4: Loss Convergence
    # ========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Loss from gradient descent
    losses_grad = [h[2] for h in gradient_history]
    losses_pytorch = [h[2] for h in pytorch_history]
    
    ax4.plot(losses_grad, 'b-', linewidth=2, label='Manual Gradient Descent')
    ax4.plot(losses_pytorch, 'r:', linewidth=2, label='PyTorch nn.Module')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Loss (MSE)', fontsize=12)
    ax4.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Plot 5: Likelihood Surface (3D)
    # ========================================================================
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Create grid of parameter values
    alpha_range = np.linspace(true_alpha - 2, true_alpha + 2, 30)
    beta_range = np.linspace(true_beta - 1, true_beta + 1, 30)
    ALPHA, BETA = np.meshgrid(alpha_range, beta_range)
    
    # Compute log-likelihood for each point
    LOG_LIK = np.zeros_like(ALPHA)
    for i in range(len(alpha_range)):
        for j in range(len(beta_range)):
            alpha_val = torch.tensor(ALPHA[j, i])
            beta_val = torch.tensor(BETA[j, i])
            LOG_LIK[j, i] = compute_log_likelihood(x, y, alpha_val, beta_val, sigma=1.0).item()
    
    # Plot surface
    surf = ax5.plot_surface(ALPHA, BETA, LOG_LIK, cmap='viridis', alpha=0.8)
    
    # Mark the MLE
    ax5.scatter([analytical_alpha], [analytical_beta], 
               [compute_log_likelihood(x, y, torch.tensor(analytical_alpha), 
                                      torch.tensor(analytical_beta), sigma=1.0).item()],
               color='red', s=100, marker='*', label='MLE')
    
    ax5.set_xlabel('α (Intercept)', fontsize=10)
    ax5.set_ylabel('β (Slope)', fontsize=10)
    ax5.set_zlabel('Log-Likelihood', fontsize=10)
    ax5.set_title('Log-Likelihood Surface', fontsize=14, fontweight='bold')
    
    # ========================================================================
    # Plot 6: Comparison Table
    # ========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create comparison table
    table_data = [
        ['Method', 'α (Intercept)', 'β (Slope)', 'MSE'],
        ['True Values', f'{true_alpha:.4f}', f'{true_beta:.4f}', '-'],
        ['Analytical MLE', f'{analytical_alpha:.4f}', f'{analytical_beta:.4f}', 
         f'{gradient_history[-1][2]:.4f}'],
        ['Gradient Descent', f'{gradient_alpha:.4f}', f'{gradient_beta:.4f}', 
         f'{gradient_history[-1][2]:.4f}'],
        ['PyTorch nn.Module', f'{pytorch_alpha:.4f}', f'{pytorch_beta:.4f}', 
         f'{pytorch_history[-1][2]:.4f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style true values row
    for i in range(4):
        table[(1, i)].set_facecolor('#E8F5E9')
    
    ax6.set_title('Method Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('linear_regression_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'linear_regression_mle_results.png'")
    plt.show()


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the linear regression MLE example.
    """
    print("=" * 80)
    print("LINEAR REGRESSION MLE - Tutorial")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Setup
    # ========================================================================
    print("\n📋 STEP 1: Problem Setup")
    print("-" * 80)
    
    N_SAMPLES = 100
    TRUE_ALPHA = 2.0
    TRUE_BETA = 3.0
    NOISE_STD = 1.0
    SEED = 42
    
    print(f"   • Number of samples: {N_SAMPLES}")
    print(f"   • True model: y = {TRUE_ALPHA} + {TRUE_BETA}x + ε")
    print(f"   • Noise: ε ~ N(0, {NOISE_STD}²)")
    
    # ========================================================================
    # Step 2: Generate Data
    # ========================================================================
    print("\n🎲 STEP 2: Generating Data")
    print("-" * 80)
    
    x, y = generate_linear_data(N_SAMPLES, TRUE_ALPHA, TRUE_BETA, NOISE_STD, SEED)
    print(f"   • Generated {N_SAMPLES} data points")
    print(f"   • x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"   • y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # ========================================================================
    # Step 3: Analytical MLE
    # ========================================================================
    print("\n📐 STEP 3: Analytical MLE (Closed-Form)")
    print("-" * 80)
    
    analytical_alpha, analytical_beta = analytical_mle(x, y)
    analytical_sigma = estimate_sigma(x, y, analytical_alpha, analytical_beta)
    
    print(f"   • Intercept (α): {analytical_alpha:.4f} (true: {TRUE_ALPHA:.4f})")
    print(f"   • Slope (β):     {analytical_beta:.4f} (true: {TRUE_BETA:.4f})")
    print(f"   • Noise std (σ): {analytical_sigma:.4f} (true: {NOISE_STD:.4f})")
    
    # ========================================================================
    # Step 4: Gradient-Based MLE
    # ========================================================================
    print("\n🔄 STEP 4: Gradient-Based MLE")
    print("-" * 80)
    
    gradient_alpha, gradient_beta, gradient_history = gradient_based_mle(
        x, y, learning_rate=0.01, n_iterations=1000
    )
    
    print(f"\n   • Final Intercept (α): {gradient_alpha:.4f}")
    print(f"   • Final Slope (β):     {gradient_beta:.4f}")
    print(f"   • Difference from analytical:")
    print(f"      Δα = {abs(gradient_alpha - analytical_alpha):.6f}")
    print(f"      Δβ = {abs(gradient_beta - analytical_beta):.6f}")
    
    # ========================================================================
    # Step 5: PyTorch nn.Module Approach
    # ========================================================================
    print("\n🔥 STEP 5: PyTorch nn.Module Approach")
    print("-" * 80)
    
    pytorch_alpha, pytorch_beta, pytorch_history = pytorch_nn_mle(
        x, y, learning_rate=0.01, n_iterations=1000
    )
    
    print(f"\n   • Final Intercept (α): {pytorch_alpha:.4f}")
    print(f"   • Final Slope (β):     {pytorch_beta:.4f}")
    
    # ========================================================================
    # Step 6: Visualization
    # ========================================================================
    print("\n📊 STEP 6: Creating Visualizations")
    print("-" * 80)
    
    visualize_results(x, y, TRUE_ALPHA, TRUE_BETA,
                     analytical_alpha, analytical_beta,
                     gradient_alpha, gradient_beta,
                     pytorch_alpha, pytorch_beta,
                     gradient_history, pytorch_history)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ SUMMARY")
    print("=" * 80)
    print(f"   MLE connects regression to probability!")
    print(f"   • MSE minimization = MLE with Gaussian noise")
    print(f"   • All three methods converge to same solution")
    print(f"   • Gradient descent is the foundation of deep learning")
    print("=" * 80)
    
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Linear regression IS maximum likelihood estimation")
    print("   2. Minimizing MSE = Maximizing likelihood (with Gaussian noise)")
    print("   3. Analytical solution exists, but gradient descent also works")
    print("   4. PyTorch makes implementation easy with automatic differentiation")
    print("   5. This generalizes to neural networks!")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. EASY: Try different noise levels (NOISE_STD)
   - How does noise affect parameter estimation?
   - Plot estimation error vs noise level

2. MEDIUM: Implement polynomial regression (y = α + βx + γx²)
   - Extend the model to include quadratic term
   - Compare MLE with analytical solution

3. MEDIUM: Add regularization (Ridge regression)
   - Modify loss: MSE + λ||θ||²
   - How does this change the MLE?
   - This is equivalent to MAP with Gaussian prior!

4. CHALLENGING: Heteroscedastic noise (non-constant variance)
   - Let σ depend on x: σ(x) = σ₀ + σ₁x
   - Estimate both model and noise parameters

5. CHALLENGING: Multiple regression (multiple input features)
   - Extend to y = α + β₁x₁ + β₂x₂ + ... + βₚxₚ
   - Visualize likelihood surface in higher dimensions
"""


if __name__ == "__main__":
    main()
