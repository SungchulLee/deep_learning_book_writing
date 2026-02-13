"""
Learning Curves Demonstration
=============================
This script demonstrates how to use learning curves to diagnose
overfitting and underfitting problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Set random seed
np.random.seed(42)

def generate_data(n_samples=1000, noise=0.3):
    """Generate synthetic data with non-linear pattern"""
    X = np.linspace(0, 10, n_samples)
    y = np.sin(X) + 0.5 * X + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y

def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curve for a given estimator
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    
    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training error', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation error', linewidth=2)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(f'Learning Curve - {title}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    final_train_error = train_mean[-1]
    final_val_error = val_mean[-1]
    gap = final_val_error - final_train_error
    
    if final_val_error > 0.5 and gap < 0.2:
        interpretation = "HIGH BIAS (Underfitting)\n• Both errors are high\n• Small gap between curves"
        color = 'orange'
    elif gap > 0.5:
        interpretation = "HIGH VARIANCE (Overfitting)\n• Large gap between curves\n• Low training error"
        color = 'red'
    else:
        interpretation = "GOOD FIT\n• Low errors\n• Small gap"
        color = 'green'
    
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig, train_mean, val_mean

def compare_learning_curves(X, y):
    """Compare learning curves for different models"""
    models = [
        ('Linear Regression (Underfitting)', 
         LinearRegression()),
        
        ('Polynomial Regression (Degree 3)', 
         Pipeline([
             ('poly', PolynomialFeatures(degree=3)),
             ('linear', LinearRegression())
         ])),
        
        ('Decision Tree (max_depth=2, Underfitting)',
         DecisionTreeRegressor(max_depth=2, random_state=42)),
        
        ('Decision Tree (max_depth=20, Overfitting)',
         DecisionTreeRegressor(max_depth=20, random_state=42)),
        
        ('Random Forest (Good Fit)',
         RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
        
        ('Ridge Regression (Degree 10, Regularized)',
         Pipeline([
             ('poly', PolynomialFeatures(degree=10)),
             ('ridge', Ridge(alpha=1.0))
         ]))
    ]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    results = []
    
    for idx, (name, model) in enumerate(models):
        print(f"\nProcessing: {name}...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, train_sizes=train_sizes,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Convert to positive MSE
        train_scores = -train_scores
        val_scores = -val_scores
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        ax = axes[idx]
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', 
                label='Training error', linewidth=2)
        ax.fill_between(train_sizes_abs, train_mean - train_std, 
                        train_mean + train_std, alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', 
                label='Validation error', linewidth=2)
        ax.fill_between(train_sizes_abs, val_mean - val_std, 
                        val_mean + val_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('Mean Squared Error', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Store results
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_val - final_train
        
        results.append({
            'name': name,
            'train_error': final_train,
            'val_error': final_val,
            'gap': gap
        })
        
        # Add diagnosis
        if final_val > 0.5 and gap < 0.2:
            diagnosis = "Underfitting"
            color = 'orange'
        elif gap > 0.5:
            diagnosis = "Overfitting"
            color = 'red'
        else:
            diagnosis = "Good Fit"
            color = 'green'
        
        ax.text(0.95, 0.95, diagnosis, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    return fig, results

def diagnose_from_learning_curve(train_error, val_error):
    """Diagnose model issues from learning curve metrics"""
    gap = val_error - train_error
    
    print("\nDiagnosis:")
    print(f"Training Error: {train_error:.4f}")
    print(f"Validation Error: {val_error:.4f}")
    print(f"Gap: {gap:.4f}")
    print()
    
    if val_error > 0.5 and gap < 0.2:
        print("→ HIGH BIAS (Underfitting)")
        print("  Symptoms:")
        print("    • Both training and validation errors are high")
        print("    • Small gap between training and validation curves")
        print("    • Curves plateau at high error")
        print("  Solutions:")
        print("    • Use more complex model")
        print("    • Add more features")
        print("    • Reduce regularization")
        print("    • Train longer")
    elif gap > 0.5:
        print("→ HIGH VARIANCE (Overfitting)")
        print("  Symptoms:")
        print("    • Large gap between training and validation errors")
        print("    • Low training error but high validation error")
        print("    • Validation error doesn't improve with more data")
        print("  Solutions:")
        print("    • Get more training data")
        print("    • Use simpler model")
        print("    • Add regularization")
        print("    • Use ensemble methods")
        print("    • Feature selection")
    else:
        print("→ GOOD FIT")
        print("  • Both errors are reasonably low")
        print("  • Small gap between curves")
        print("  • Model generalizes well")

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("Learning Curves Analysis")
    print("="*70)
    
    # Generate data
    X, y = generate_data(n_samples=1000, noise=0.3)
    
    # Example 1: Underfitting model
    print("\n1. UNDERFITTING EXAMPLE - Linear Regression")
    print("-"*70)
    model_underfit = LinearRegression()
    fig1, train_mean, val_mean = plot_learning_curve(
        model_underfit, 'Linear Regression (Underfitting)', X, y
    )
    plt.savefig('learning_curve_underfitting.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # Example 2: Overfitting model
    print("\n2. OVERFITTING EXAMPLE - Deep Decision Tree")
    print("-"*70)
    model_overfit = DecisionTreeRegressor(max_depth=20, random_state=42)
    fig2, train_mean, val_mean = plot_learning_curve(
        model_overfit, 'Decision Tree (Overfitting)', X, y
    )
    plt.savefig('learning_curve_overfitting.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # Example 3: Good fit model
    print("\n3. GOOD FIT EXAMPLE - Random Forest")
    print("-"*70)
    model_good = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    fig3, train_mean, val_mean = plot_learning_curve(
        model_good, 'Random Forest (Good Fit)', X, y
    )
    plt.savefig('learning_curve_good_fit.png', dpi=150, bbox_inches='tight')
    print(f"Final Training Error: {train_mean[-1]:.4f}")
    print(f"Final Validation Error: {val_mean[-1]:.4f}")
    diagnose_from_learning_curve(train_mean[-1], val_mean[-1])
    
    # Comparison plot
    print("\n4. COMPARING MULTIPLE MODELS")
    print("-"*70)
    fig4, results = compare_learning_curves(X, y)
    plt.savefig('learning_curves_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"{'Model':<50} {'Train Error':<15} {'Val Error':<15} {'Gap':<10}")
    print("-"*70)
    for result in results:
        print(f"{result['name']:<50} {result['train_error']:<15.4f} "
              f"{result['val_error']:<15.4f} {result['gap']:<10.4f}")
    print("="*70)
    
    plt.show()
    
    print("\n" + "="*70)
    print("Key Insights from Learning Curves:")
    print("="*70)
    print("""
1. HIGH BIAS (Underfitting):
   • Both curves plateau at high error
   • Small gap between training and validation
   • More data won't help much
   → Solution: Increase model complexity

2. HIGH VARIANCE (Overfitting):
   • Large gap between curves
   • Training error is low, validation error is high
   • More data can help
   → Solution: Get more data or reduce complexity

3. GOOD FIT:
   • Both errors are low
   • Small gap between curves
   • Curves converge
   → Model is working well!
    """)
    print("="*70)
