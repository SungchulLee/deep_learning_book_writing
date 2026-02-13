#!/usr/bin/env python3
"""
================================================================================
NEURAL NETWORK MLE - Deep Learning with Maximum Likelihood
================================================================================

DIFFICULTY: ⭐⭐⭐ Advanced (Level 3)

LEARNING OBJECTIVES:
- Understand how neural networks use MLE
- See the connection between loss functions and likelihood
- Implement custom MLE-based losses
- Learn about heteroscedastic regression (predicting uncertainty)

KEY INSIGHT: Neural network training IS maximum likelihood estimation!

STANDARD REGRESSION:
- Network predicts: ŷ = f(x; θ)
- Assume: y ~ N(ŷ, σ²) with fixed σ
- MLE objective: minimize Σ(y - ŷ)² (MSE loss)

HETEROSCEDASTIC REGRESSION:
- Network predicts BOTH mean AND variance: (μ̂, σ̂²) = f(x; θ)
- Model: y ~ N(μ̂, σ̂²) with varying σ
- MLE objective: maximize Σ log N(y | μ̂, σ̂²)
             = minimize Σ [log(σ̂²) + (y - μ̂)²/σ̂²]

This allows the network to express UNCERTAINTY in its predictions!

APPLICATIONS:
- Regression with uncertainty quantification
- Robust regression (outlier handling)
- Active learning (query high-uncertainty points)
- Risk-sensitive decision making

AUTHOR: PyTorch MLE Tutorial
DATE: 2025
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_heteroscedastic_data(n_samples: int = 300, seed: int = 42):
    """
    Generate data where noise varies with x (heteroscedastic).
    
    y = sin(x) + ε, where ε ~ N(0, σ(x)²) and σ(x) increases with |x|
    """
    torch.manual_seed(seed)
    
    # Generate x values
    x = torch.rand(n_samples, 1) * 10 - 5  # Range: [-5, 5]
    
    # True function: sine wave
    y_true = torch.sin(x)
    
    # Heteroscedastic noise: σ(x) = 0.1 + 0.1 * |x|
    sigma_x = 0.1 + 0.1 * torch.abs(x)
    noise = torch.randn_like(x) * sigma_x
    
    y = y_true + noise
    
    return x, y, sigma_x


class StandardNN(nn.Module):
    """Standard neural network predicting only mean"""
    
    def __init__(self, hidden_size=50):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class HeteroscedasticNN(nn.Module):
    """
    Neural network predicting BOTH mean and variance.
    
    This is the MLE approach for heteroscedastic regression!
    """
    
    def __init__(self, hidden_size=50):
        super().__init__()
        
        # Shared hidden layers
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Mean head
        self.mean_head = nn.Linear(hidden_size, 1)
        
        # Log-variance head (we predict log(σ²) to ensure σ² > 0)
        self.logvar_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Returns:
            mean: Predicted mean
            logvar: Predicted log-variance (log(σ²))
        """
        features = self.shared(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar


def gaussian_nll_loss(y_true, y_pred_mean, y_pred_logvar):
    """
    Gaussian Negative Log-Likelihood loss.
    
    This is the MLE objective for heteroscedastic regression!
    
    NLL = -log N(y | μ, σ²)
        = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
    
    Ignoring constants:
    NLL = 0.5 * [log(σ²) + (y - μ)² / σ²]
        = 0.5 * [log_var + (y - μ)² / exp(log_var)]
    """
    # Compute negative log-likelihood
    variance = torch.exp(y_pred_logvar)
    loss = 0.5 * (y_pred_logvar + (y_true - y_pred_mean) ** 2 / variance)
    
    return loss.mean()


def train_standard_nn(x_train, y_train, epochs=1000, lr=0.01):
    """Train standard NN with MSE loss"""
    
    model = StandardNN(hidden_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model, history


def train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01):
    """Train heteroscedastic NN with custom MLE loss"""
    
    model = HeteroscedasticNN(hidden_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred_mean, y_pred_logvar = model(x_train)
        
        # Compute negative log-likelihood (our MLE objective!)
        loss = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{epochs}, NLL Loss: {loss.item():.4f}")
    
    return model, history


def visualize_results(x, y, x_test, true_sigma, model_standard, model_hetero):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # Get predictions
    with torch.no_grad():
        # Standard model
        y_pred_standard = model_standard(x_test)
        
        # Heteroscedastic model
        y_pred_mean, y_pred_logvar = model_hetero(x_test)
        y_pred_std = torch.sqrt(torch.exp(y_pred_logvar))
    
    x_np = x.numpy().flatten()
    y_np = y.numpy().flatten()
    x_test_np = x_test.numpy().flatten()
    
    # ================================================================
    # Plot 1: Standard NN Predictions
    # ================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Sort for plotting
    sort_idx = torch.argsort(x_test.flatten())
    x_sorted = x_test_np[sort_idx]
    y_pred_sorted = y_pred_standard.numpy().flatten()[sort_idx]
    
    ax1.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
    ax1.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Standard NN')
    ax1.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Standard NN (MSE Loss)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 2: Heteroscedastic NN with Uncertainty
    # ================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    y_mean_sorted = y_pred_mean.numpy().flatten()[sort_idx]
    y_std_sorted = y_pred_std.numpy().flatten()[sort_idx]
    
    ax2.scatter(x_np, y_np, alpha=0.5, s=20, label='Data', color='blue')
    ax2.plot(x_sorted, y_mean_sorted, 'r-', linewidth=2, label='Predicted mean')
    ax2.plot(x_sorted, np.sin(x_sorted), 'g--', linewidth=2, label='True function')
    
    # Plot uncertainty bands (±1σ, ±2σ)
    ax2.fill_between(x_sorted, 
                     y_mean_sorted - 2*y_std_sorted,
                     y_mean_sorted + 2*y_std_sorted,
                     alpha=0.2, color='red', label='±2σ (95% CI)')
    ax2.fill_between(x_sorted,
                     y_mean_sorted - y_std_sorted,
                     y_mean_sorted + y_std_sorted,
                     alpha=0.3, color='red', label='±1σ (68% CI)')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Heteroscedastic NN (MLE Loss)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 3: Predicted vs True Uncertainty
    # ================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    true_sigma_sorted = true_sigma.numpy().flatten()[sort_idx]
    
    ax3.plot(x_sorted, true_sigma_sorted, 'g-', linewidth=3, label='True σ(x)')
    ax3.plot(x_sorted, y_std_sorted, 'r-', linewidth=3, label='Predicted σ(x)')
    ax3.fill_between(x_sorted, 0, true_sigma_sorted, alpha=0.2, color='green')
    
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('σ (Standard Deviation)', fontsize=12)
    ax3.set_title('Uncertainty Estimation', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 4: Residuals Comparison
    # ================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    residuals_standard = (y - model_standard(x)).numpy().flatten()
    residuals_hetero = (y - y_pred_mean).numpy().flatten()
    
    ax4.scatter(x_np, residuals_standard, alpha=0.5, s=20, label='Standard NN', color='blue')
    ax4.scatter(x_np, residuals_hetero, alpha=0.5, s=20, label='Heteroscedastic NN', color='red')
    ax4.axhline(0, color='black', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('Residuals', fontsize=12)
    ax4.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 5: Calibration Plot
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # Compute normalized residuals for heteroscedastic model
    residuals = (y - y_pred_mean).numpy().flatten()
    predicted_stds = y_pred_std.numpy().flatten()
    normalized_residuals = residuals / predicted_stds
    
    # Histogram of normalized residuals (should be N(0,1) if well-calibrated)
    ax5.hist(normalized_residuals, bins=30, density=True, alpha=0.7, 
            edgecolor='black', label='Normalized Residuals')
    
    # Overlay N(0,1) distribution
    x_range = np.linspace(-4, 4, 100)
    from scipy.stats import norm
    ax5.plot(x_range, norm.pdf(x_range), 'r-', linewidth=2, label='N(0,1)')
    
    ax5.set_xlabel('Normalized Residuals', fontsize=12)
    ax5.set_ylabel('Density', fontsize=12)
    ax5.set_title('Calibration Check', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 6: Log-Likelihood Comparison
    # ================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute log-likelihoods
    with torch.no_grad():
        # Standard model (assume fixed σ = 1)
        residuals_std = y - model_standard(x)
        nll_standard = 0.5 * (np.log(2 * np.pi) + torch.mean(residuals_std ** 2)).item()
        
        # Heteroscedastic model
        mean_het, logvar_het = model_hetero(x)
        nll_hetero = gaussian_nll_loss(y, mean_het, logvar_het).item()
    
    # Comparison table
    table_data = [
        ['Model', 'Negative Log-Likelihood'],
        ['Standard NN', f'{nll_standard:.4f}'],
        ['Heteroscedastic NN', f'{nll_hetero:.4f}'],
        ['Improvement', f'{nll_standard - nll_hetero:.4f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight improvement
    table[(3, 0)].set_facecolor('#FFF9C4')
    table[(3, 1)].set_facecolor('#FFF9C4')
    
    ax6.set_title('Model Comparison (Lower is Better)', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('neural_network_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'neural_network_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("NEURAL NETWORK MLE - Deep Learning with Uncertainty")
    print("=" * 80)
    
    # Generate data
    print("\n🎲 Generating heteroscedastic data...")
    x_train, y_train, true_sigma = generate_heteroscedastic_data(n_samples=300)
    
    # Test data for smooth predictions
    x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
    
    print(f"   • Training samples: {len(x_train)}")
    print(f"   • Noise varies with x (heteroscedastic)")
    
    # Train standard NN
    print("\n🔵 Training Standard NN (MSE Loss)...")
    print("-" * 80)
    model_standard, history_standard = train_standard_nn(x_train, y_train, epochs=1000, lr=0.01)
    
    # Train heteroscedastic NN
    print("\n🔴 Training Heteroscedastic NN (MLE Loss)...")
    print("-" * 80)
    model_hetero, history_hetero = train_heteroscedastic_nn(x_train, y_train, epochs=1000, lr=0.01)
    
    # Evaluation
    print("\n📊 Evaluation:")
    print("-" * 80)
    
    with torch.no_grad():
        # Standard model
        y_pred_std = model_standard(x_train)
        mse_std = torch.mean((y_train - y_pred_std) ** 2).item()
        
        # Heteroscedastic model
        y_pred_mean, y_pred_logvar = model_hetero(x_train)
        mse_het = torch.mean((y_train - y_pred_mean) ** 2).item()
        nll_het = gaussian_nll_loss(y_train, y_pred_mean, y_pred_logvar).item()
    
    print(f"   Standard NN:")
    print(f"      MSE: {mse_std:.4f}")
    
    print(f"\n   Heteroscedastic NN:")
    print(f"      MSE: {mse_het:.4f}")
    print(f"      NLL: {nll_het:.4f}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    visualize_results(x_train, y_train, x_test, true_sigma, model_standard, model_hetero)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Neural networks ARE MLE when trained with appropriate losses")
    print("   2. MSE = MLE with Gaussian assumption and fixed variance")
    print("   3. Heteroscedastic networks predict uncertainty!")
    print("   4. Custom loss functions = Custom probabilistic assumptions")
    print("   5. This enables uncertainty-aware deep learning")
    print("\n   🎯 Applications:")
    print("      • Medical diagnosis (quantify confidence)")
    print("      • Autonomous vehicles (safety-critical decisions)")
    print("      • Financial modeling (risk assessment)")
    print("      • Active learning (query uncertain points)")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. MEDIUM: Classification with uncertainty
   - Extend to classification task
   - Predict class probabilities (softmax)
   - Use negative log-likelihood (cross-entropy)
   - Visualize prediction confidence

2. MEDIUM: Different noise models
   - Laplace noise: use absolute error instead of squared
   - Student-t noise: robust to outliers
   - Compare likelihood functions

3. CHALLENGING: Bayesian Neural Networks
   - Add dropout for uncertainty estimation
   - Monte Carlo dropout: multiple forward passes
   - Compare epistemic vs aleatoric uncertainty

4. CHALLENGING: Multi-output regression
   - Predict vector outputs with covariance
   - Full covariance matrix vs diagonal
   - Multivariate Gaussian likelihood

5. CHALLENGING: Active learning
   - Use uncertainty to select informative samples
   - Train on small dataset
   - Iteratively query high-uncertainty points
   - Show learning curve improves faster
"""


if __name__ == "__main__":
    main()
