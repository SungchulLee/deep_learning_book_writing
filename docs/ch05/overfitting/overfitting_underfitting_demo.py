"""
Overfitting and Underfitting Demonstration
==========================================
This script demonstrates the concepts of overfitting and underfitting
using polynomial regression with different degrees.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=100, noise=0.5):
    """Generate synthetic data with a non-linear pattern"""
    X = np.linspace(0, 10, n_samples)
    y = np.sin(X) + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y

# Train polynomial regression models
def train_polynomial_models(X_train, y_train, X_test, y_test, degrees):
    """Train polynomial regression models with different degrees"""
    results = {}
    
    for degree in degrees:
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results[degree] = {
            'model': model,
            'poly_features': poly_features,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
    
    return results

# Visualization
def plot_results(X_train, y_train, X_test, y_test, results, degrees):
    """Plot the results for different polynomial degrees"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        result = results[degree]
        
        # Sort data for plotting smooth curves
        sort_idx_train = np.argsort(X_train.ravel())
        sort_idx_test = np.argsort(X_test.ravel())
        
        # Plot training data
        ax.scatter(X_train, y_train, alpha=0.5, label='Train data', color='blue')
        ax.plot(X_train[sort_idx_train], result['y_train_pred'][sort_idx_train], 
                'b-', label='Train prediction', linewidth=2)
        
        # Plot test data
        ax.scatter(X_test, y_test, alpha=0.5, label='Test data', color='red')
        ax.plot(X_test[sort_idx_test], result['y_test_pred'][sort_idx_test], 
                'r-', label='Test prediction', linewidth=2)
        
        # Add title with metrics
        ax.set_title(f'Degree {degree}\n'
                    f'Train MSE: {result["train_mse"]:.4f}, Test MSE: {result["test_mse"]:.4f}\n'
                    f'Train R²: {result["train_r2"]:.4f}, Test R²: {result["test_r2"]:.4f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        if result['test_mse'] > 0.5 and degree <= 2:
            ax.text(0.5, 0.95, 'UNDERFITTING', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='orange',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif result['test_mse'] > result['train_mse'] * 2 and degree >= 10:
            ax.text(0.5, 0.95, 'OVERFITTING', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        else:
            ax.text(0.5, 0.95, 'GOOD FIT', transform=ax.transAxes,
                   ha='center', va='top', fontsize=12, color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(n_samples=100, noise=0.5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 5, 10, 15]
    
    # Train models
    print("Training polynomial regression models...")
    results = train_polynomial_models(X_train, y_train, X_test, y_test, degrees)
    
    # Print results
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Degree':<10} {'Train MSE':<15} {'Test MSE':<15} {'Interpretation':<20}")
    print("-"*70)
    
    for degree in degrees:
        result = results[degree]
        train_mse = result['train_mse']
        test_mse = result['test_mse']
        
        # Interpretation
        if test_mse > 0.5 and degree <= 2:
            interpretation = "Underfitting"
        elif test_mse > train_mse * 2 and degree >= 10:
            interpretation = "Overfitting"
        else:
            interpretation = "Good fit"
        
        print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {interpretation:<20}")
    
    print("="*70)
    
    # Create visualization
    fig = plot_results(X_train, y_train, X_test, y_test, results, degrees)
    plt.savefig('overfitting_underfitting.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'overfitting_underfitting.png'")
    plt.show()
    
    # Key takeaways
    print("\nKey Takeaways:")
    print("- Underfitting: High training and test error (model too simple)")
    print("- Overfitting: Low training error but high test error (model too complex)")
    print("- Good fit: Similar training and test errors (balanced complexity)")
