"""
Bayesian Inference - Module 9: Bayesian Linear Regression
Level: Advanced
Topics: Bayesian regression, predictive distributions, uncertainty quantification

Bayesian linear regression provides full posterior distributions over parameters
and predictions, naturally quantifying uncertainty.

Author: Professor Sungchul, Yonsei University
Email: sungchulyonsei@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

"""
BAYESIAN LINEAR REGRESSION:

Model: y = Xβ + ε, where ε ~ N(0, σ²I)

Prior: β ~ N(m₀, V₀)

Posterior: β|y ~ N(mₙ, Vₙ)
where:
  Vₙ = (V₀⁻¹ + (1/σ²)X'X)⁻¹
  mₙ = Vₙ(V₀⁻¹m₀ + (1/σ²)X'y)

Predictive Distribution:
  y*|y ~ N(x*'mₙ, σ² + x*'Vₙx*)
  
The extra x*'Vₙx* term captures parameter uncertainty.
"""

def bayesian_linear_regression_demo():
    """
    Demonstrate Bayesian linear regression with uncertainty quantification.
    """
    print("="*70)
    print("BAYESIAN LINEAR REGRESSION")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 30
    X = np.linspace(0, 10, n)
    true_intercept = 2.0
    true_slope = 1.5
    noise_std = 2.0
    
    y = true_intercept + true_slope * X + np.random.normal(0, noise_std, n)
    
    # Design matrix
    X_design = np.column_stack([np.ones(n), X])
    
    # Prior: weakly informative
    m0 = np.array([0, 0])
    V0 = np.eye(2) * 100
    
    # Posterior (assuming known noise variance)
    sigma_sq = noise_std ** 2
    V_inv = np.linalg.inv(V0) + (1/sigma_sq) * X_design.T @ X_design
    Vn = np.linalg.inv(V_inv)
    mn = Vn @ (np.linalg.inv(V0) @ m0 + (1/sigma_sq) * X_design.T @ y)
    
    print(f"\nTrue parameters: β₀={true_intercept}, β₁={true_slope}")
    print(f"Posterior mean: β₀={mn[0]:.3f}, β₁={mn[1]:.3f}")
    print(f"Posterior std:  β₀={np.sqrt(Vn[0,0]):.3f}, β₁={np.sqrt(Vn[1,1]):.3f}")
    
    # Predictions
    X_test = np.linspace(-1, 11, 200)
    X_test_design = np.column_stack([np.ones(len(X_test)), X_test])
    
    # Mean prediction
    y_pred_mean = X_test_design @ mn
    
    # Prediction uncertainty
    pred_var = sigma_sq + np.sum((X_test_design @ Vn) * X_test_design, axis=1)
    pred_std = np.sqrt(pred_var)
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, s=50, label='Data')
    plt.plot(X_test, y_pred_mean, 'r-', linewidth=2, label='Posterior mean')
    plt.fill_between(X_test, y_pred_mean - 2*pred_std, y_pred_mean + 2*pred_std,
                     alpha=0.3, color='red', label='95% Predictive interval')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Bayesian Linear Regression', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sample from posterior
    plt.subplot(1, 2, 2)
    plt.scatter(X, y, alpha=0.6, s=50)
    for _ in range(20):
        beta_sample = np.random.multivariate_normal(mn, Vn)
        y_sample = X_test_design @ beta_sample
        plt.plot(X_test, y_sample, 'r-', alpha=0.2, linewidth=1)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Posterior Samples of Regression Lines', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYESIAN INFERENCE - MODULE 9: BAYESIAN LINEAR REGRESSION")
    print("="*70)
    
    bayesian_linear_regression_demo()
    
    print("\n" + "="*70)
    print("MODULE 9 COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("1. Bayesian regression gives full posterior over parameters")
    print("2. Predictive distribution includes parameter uncertainty")
    print("3. Naturally regularized through prior")
    print("4. Uncertainty quantification is automatic")
    print("\nNext: Module 10 - Advanced Applications")
    print("="*70)
