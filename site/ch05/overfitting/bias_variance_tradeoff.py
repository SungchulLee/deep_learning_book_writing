"""
Bias-Variance Tradeoff Demonstration
====================================
This script demonstrates the bias-variance tradeoff concept
by decomposing the expected prediction error.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)

def generate_data(n_samples=200, noise_std=0.3):
    """Generate synthetic data with a known function"""
    X = np.linspace(0, 10, n_samples)
    # True function: sin(x)
    y_true = np.sin(X)
    # Add noise
    y = y_true + np.random.normal(0, noise_std, n_samples)
    return X.reshape(-1, 1), y, y_true

def compute_bias_variance(X_train, y_train, X_test, y_test_true, 
                          max_depth, n_iterations=100):
    """
    Compute bias and variance for a model by training multiple times
    on bootstrap samples.
    
    Expected Error = Bias² + Variance + Irreducible Error
    """
    predictions = []
    
    for i in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Train model
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=i)
        model.fit(X_boot, y_boot)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # Compute bias
    mean_prediction = np.mean(predictions, axis=0)
    bias = np.mean((mean_prediction - y_test_true) ** 2)
    
    # Compute variance
    variance = np.mean(np.var(predictions, axis=0))
    
    # Total expected error (bias² + variance)
    expected_error = bias + variance
    
    return bias, variance, expected_error, predictions, mean_prediction

def analyze_complexity_range(X_train, y_train, X_test, y_test_true):
    """Analyze bias-variance tradeoff across different model complexities"""
    max_depths = range(1, 16)
    biases = []
    variances = []
    total_errors = []
    
    print("Analyzing bias-variance tradeoff...")
    for depth in max_depths:
        print(f"Processing max_depth={depth}...")
        bias, variance, total_error, _, _ = compute_bias_variance(
            X_train, y_train, X_test, y_test_true, 
            max_depth=depth, n_iterations=50
        )
        biases.append(bias)
        variances.append(variance)
        total_errors.append(total_error)
    
    return max_depths, biases, variances, total_errors

def plot_bias_variance_tradeoff(max_depths, biases, variances, total_errors):
    """Plot the bias-variance tradeoff"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Bias-Variance Tradeoff
    plt.subplot(1, 2, 1)
    plt.plot(max_depths, biases, 'b-o', label='Bias²', linewidth=2)
    plt.plot(max_depths, variances, 'r-s', label='Variance', linewidth=2)
    plt.plot(max_depths, total_errors, 'g-^', label='Total Error (Bias² + Variance)', 
             linewidth=2, markersize=8)
    
    # Find optimal complexity
    optimal_idx = np.argmin(total_errors)
    optimal_depth = max_depths[optimal_idx]
    plt.axvline(x=optimal_depth, color='purple', linestyle='--', 
                label=f'Optimal Complexity (depth={optimal_depth})')
    
    plt.xlabel('Model Complexity (Max Depth)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Component Analysis
    plt.subplot(1, 2, 2)
    width = 0.35
    x = np.arange(len(max_depths))
    
    plt.bar(x, biases, width, label='Bias²', alpha=0.8)
    plt.bar(x, variances, width, bottom=biases, label='Variance', alpha=0.8)
    
    plt.xlabel('Model Complexity (Max Depth)', fontsize=12)
    plt.ylabel('Error Components', fontsize=12)
    plt.title('Stacked Error Components', fontsize=14, fontweight='bold')
    plt.xticks(x[::2], [max_depths[i] for i in range(0, len(max_depths), 2)])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_predictions_by_complexity(X_train, y_train, X_test, y_test_true):
    """Visualize predictions for different complexity levels"""
    complexities = [1, 3, 7, 15]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, depth in enumerate(complexities):
        ax = axes[idx]
        
        bias, variance, total_error, predictions, mean_pred = compute_bias_variance(
            X_train, y_train, X_test, y_test_true, 
            max_depth=depth, n_iterations=30
        )
        
        # Sort for plotting
        sort_idx = np.argsort(X_test.ravel())
        
        # Plot individual predictions (showing model variability)
        for pred in predictions[:10]:  # Show only first 10 for clarity
            ax.plot(X_test[sort_idx], pred[sort_idx], 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot mean prediction
        ax.plot(X_test[sort_idx], mean_pred[sort_idx], 'b-', 
                label='Mean Prediction', linewidth=2)
        
        # Plot true function
        ax.plot(X_test[sort_idx], y_test_true[sort_idx], 'g-', 
                label='True Function', linewidth=2)
        
        # Plot training data
        ax.scatter(X_train, y_train, alpha=0.3, s=20, c='red', label='Training Data')
        
        ax.set_title(f'Max Depth = {depth}\n'
                    f'Bias² = {bias:.4f}, Variance = {variance:.4f}\n'
                    f'Total Error = {total_error:.4f}',
                    fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        if bias > variance:
            interpretation = "HIGH BIAS\n(Underfitting)"
            color = 'orange'
        elif variance > bias * 2:
            interpretation = "HIGH VARIANCE\n(Overfitting)"
            color = 'red'
        else:
            interpretation = "BALANCED"
            color = 'green'
        
        ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("Bias-Variance Tradeoff Analysis")
    print("="*70)
    
    # Generate data
    X, y, y_true = generate_data(n_samples=200, noise_std=0.3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Get true values for test set
    y_test_true = np.sin(X_test.ravel())
    
    # Analyze bias-variance across different complexities
    max_depths, biases, variances, total_errors = analyze_complexity_range(
        X_train, y_train, X_test, y_test_true
    )
    
    # Print summary
    print("\n" + "="*70)
    print("Bias-Variance Analysis Summary")
    print("="*70)
    print(f"{'Max Depth':<12} {'Bias²':<15} {'Variance':<15} {'Total Error':<15}")
    print("-"*70)
    
    for depth, bias, var, total in zip(max_depths, biases, variances, total_errors):
        print(f"{depth:<12} {bias:<15.4f} {var:<15.4f} {total:<15.4f}")
    
    optimal_idx = np.argmin(total_errors)
    print("="*70)
    print(f"Optimal Complexity: Max Depth = {max_depths[optimal_idx]}")
    print(f"Minimum Total Error: {total_errors[optimal_idx]:.4f}")
    print("="*70)
    
    # Create visualizations
    fig1 = plot_bias_variance_tradeoff(max_depths, biases, variances, total_errors)
    plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("\nBias-variance tradeoff plot saved as 'bias_variance_tradeoff.png'")
    
    fig2 = visualize_predictions_by_complexity(X_train, y_train, X_test, y_test_true)
    plt.savefig('bias_variance_predictions.png', dpi=150, bbox_inches='tight')
    print("Predictions visualization saved as 'bias_variance_predictions.png'")
    
    plt.show()
    
    # Key concepts
    print("\n" + "="*70)
    print("Key Concepts:")
    print("="*70)
    print("• BIAS: Error from incorrect assumptions (underfitting)")
    print("  - High bias → model too simple → systematic errors")
    print("  - Low complexity models have high bias")
    print("\n• VARIANCE: Error from sensitivity to training data (overfitting)")
    print("  - High variance → model too complex → unstable predictions")
    print("  - High complexity models have high variance")
    print("\n• TRADEOFF: As complexity increases:")
    print("  - Bias decreases (model captures more patterns)")
    print("  - Variance increases (model becomes more sensitive)")
    print("\n• OPTIMAL MODEL: Minimizes (Bias² + Variance)")
    print("="*70)
