#!/usr/bin/env python3
"""
================================================================================
LOGISTIC REGRESSION MLE - Binary Classification
================================================================================

DIFFICULTY: ⭐⭐⭐ Advanced (Level 3)

LEARNING OBJECTIVES:
- Understand logistic regression as MLE
- See connection to binary cross-entropy loss
- Implement classification in PyTorch
- Learn about sigmoid function and log-odds

PROBLEM: Binary classification - predict y ∈ {0, 1} from features x

MODEL: P(y=1 | x) = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))

where σ is the sigmoid/logistic function

MLE FORMULATION:
Likelihood: L(w, b) = ∏ P(y_i | x_i, w, b)
           = ∏ σ(w^T x_i + b)^y_i * (1 - σ(w^T x_i + b))^(1-y_i)

Log-likelihood: ℓ(w, b) = Σ [y_i log(σ(w^T x_i + b)) + (1-y_i) log(1 - σ(w^T x_i + b))]

This is equivalent to MINIMIZING binary cross-entropy loss!

KEY INSIGHT: Cross-entropy = Negative log-likelihood

AUTHOR: PyTorch MLE Tutorial  
DATE: 2025
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_classification_data(n_samples: int = 200, seed: int = 42):
    """Generate synthetic binary classification data"""
    torch.manual_seed(seed)
    
    # Generate two clusters
    n_per_class = n_samples // 2
    
    # Class 0: centered at (-2, -2)
    X0 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([-2.0, -2.0])
    y0 = torch.zeros(n_per_class, 1)
    
    # Class 1: centered at (2, 2)
    X1 = torch.randn(n_per_class, 2) * 0.8 + torch.tensor([2.0, 2.0])
    y1 = torch.ones(n_per_class, 1)
    
    # Combine
    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    X, y = X[perm], y[perm]
    
    return X, y


def compute_log_likelihood(X, y, w, b):
    """
    Compute log-likelihood for logistic regression.
    
    ℓ(w,b) = Σ [y_i log(σ(z_i)) + (1-y_i) log(1 - σ(z_i))]
    where z_i = w^T x_i + b
    """
    # Compute logits
    logits = X @ w + b
    
    # Apply sigmoid (numerically stable version)
    probs = torch.sigmoid(logits)
    
    # Log-likelihood
    epsilon = 1e-8  # For numerical stability
    log_lik = torch.sum(
        y * torch.log(probs + epsilon) + (1 - y) * torch.log(1 - probs + epsilon)
    )
    
    return log_lik


class LogisticRegressionModel(nn.Module):
    """Logistic Regression implemented as PyTorch module"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass"""
        logits = self.linear(x)
        probs = self.sigmoid(logits)
        return probs


def train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000):
    """Train logistic regression using MLE (cross-entropy loss)"""
    
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim)
    
    # Loss function: Binary Cross-Entropy (equivalent to negative log-likelihood)
    criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(n_epochs):
        # Forward pass
        probs = model(X)
        loss = criterion(probs, y)
        
        # Compute accuracy
        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        
        # Store history
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.2%}")
    
    return model, history


def visualize_results(X, y, model, history):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # ================================================================
    # Plot 1: Decision Boundary
    # ================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    # Plot decision boundary and regions
    ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot data points
    X_np, y_np = X.numpy(), y.numpy().flatten()
    scatter = ax1.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], 
                         c='blue', marker='o', s=50, edgecolors='black', label='Class 0')
    scatter = ax1.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], 
                         c='red', marker='^', s=50, edgecolors='black', label='Class 1')
    
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Decision Boundary', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 2: Loss Convergence (Log-Likelihood)
    # ================================================================
    ax2 = plt.subplot(2, 3, 2)
    
    # Convert loss to log-likelihood (loss = -log_lik / n)
    log_likelihood = [-loss * len(X) for loss in history['loss']]
    
    ax2.plot(log_likelihood, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Log-Likelihood', fontsize=12)
    ax2.set_title('Log-Likelihood Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 3: Accuracy Convergence
    # ================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.plot(history['accuracy'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy Convergence', fontsize=14, fontweight='bold')
    ax3.axhline(0.5, color='gray', linestyle='--', label='Random chance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # ================================================================
    # Plot 4: Probability Predictions
    # ================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    with torch.no_grad():
        probs = model(X).numpy().flatten()
    
    # Separate by true class
    probs_class0 = probs[y.numpy().flatten() == 0]
    probs_class1 = probs[y.numpy().flatten() == 1]
    
    ax4.hist(probs_class0, bins=20, alpha=0.6, color='blue', label='True Class 0', edgecolor='black')
    ax4.hist(probs_class1, bins=20, alpha=0.6, color='red', label='True Class 1', edgecolor='black')
    ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ================================================================
    # Plot 5: Sigmoid Function
    # ================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    z = np.linspace(-6, 6, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    
    ax5.plot(z, sigmoid, 'b-', linewidth=3)
    ax5.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Logit (w^T x + b)', fontsize=12)
    ax5.set_ylabel('P(y=1)', fontsize=12)
    ax5.set_title('Sigmoid/Logistic Function', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.text(3, 0.2, 'σ(z) = 1/(1 + e^(-z))', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # ================================================================
    # Plot 6: Confusion Matrix
    # ================================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Compute confusion matrix
    with torch.no_grad():
        predictions = (model(X) > 0.5).float()
        
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y.numpy(), predictions.numpy())
    
    # Display confusion matrix
    im = ax6.imshow(cm, cmap='Blues', alpha=0.7)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, str(cm[i, j]), ha="center", va="center",
                          color="black", fontsize=20, fontweight='bold')
    
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Pred 0', 'Pred 1'])
    ax6.set_yticklabels(['True 0', 'True 1'])
    ax6.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""
Accuracy:  {accuracy:.3f}
Precision: {precision:.3f}
Recall:    {recall:.3f}
F1-Score:  {f1:.3f}
"""
    ax6.text(0.5, -0.25, metrics_text, ha='center', va='top',
            fontsize=10, family='monospace', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('logistic_regression_mle_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Figure saved as 'logistic_regression_mle_results.png'")
    plt.show()


def main():
    print("=" * 80)
    print("LOGISTIC REGRESSION MLE - Binary Classification")
    print("=" * 80)
    
    # Generate data
    print("\n🎲 Generating classification data...")
    X, y = generate_classification_data(n_samples=200)
    
    print(f"   • Dataset size: {len(X)} samples")
    print(f"   • Features: {X.shape[1]} dimensions")
    print(f"   • Class 0: {(y == 0).sum().item()} samples")
    print(f"   • Class 1: {(y == 1).sum().item()} samples")
    
    # Train model
    print("\n🔥 Training Logistic Regression via MLE...")
    print("-" * 80)
    model, history = train_logistic_regression(X, y, learning_rate=0.1, n_epochs=1000)
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    print("-" * 80)
    with torch.no_grad():
        probs = model(X)
        predictions = (probs > 0.5).float()
        accuracy = (predictions == y).float().mean()
        
        # Compute log-likelihood
        w = model.linear.weight.data
        b = model.linear.bias.data
        log_lik = compute_log_likelihood(X, y, w.T, b)
        
    print(f"   • Final Accuracy: {accuracy.item():.2%}")
    print(f"   • Final Log-Likelihood: {log_lik.item():.2f}")
    print(f"   • Final Loss (BCE): {history['loss'][-1]:.4f}")
    
    # Show model parameters
    print(f"\n   Model Parameters:")
    print(f"   • Weight w: {model.linear.weight.data.numpy().flatten()}")
    print(f"   • Bias b: {model.linear.bias.data.item():.4f}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    visualize_results(X, y, model, history)
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\n💡 KEY TAKEAWAYS:")
    print("   1. Logistic regression IS maximum likelihood estimation")
    print("   2. Binary cross-entropy = Negative log-likelihood")
    print("   3. Sigmoid maps real numbers to probabilities [0, 1]")
    print("   4. Gradient descent optimizes the MLE")
    print("   5. This is the foundation of neural networks!")
    print("\n" + "=" * 80)


"""
🎓 EXERCISES:

1. MEDIUM: Multi-class logistic regression (softmax)
   - Extend to 3+ classes
   - Use categorical cross-entropy
   - Visualize decision boundaries

2. MEDIUM: Regularized logistic regression
   - Add L2 regularization: loss + λ||w||²
   - This is equivalent to MAP with Gaussian prior!
   - Compare regularized vs unregularized

3. CHALLENGING: Feature engineering
   - Add polynomial features (x₁², x₁x₂, x₂²)
   - Create non-linear decision boundaries
   - Compare with linear features

4. CHALLENGING: Imbalanced classes
   - Generate data with 10:1 class ratio
   - Try weighted loss functions
   - Evaluate with precision, recall, F1

5. CHALLENGING: Probabilistic interpretation
   - Plot calibration curves (predicted vs actual probability)
   - Implement Platt scaling for calibration
   - Compare well-calibrated vs poorly-calibrated models
"""


if __name__ == "__main__":
    main()
