"""
Practical Solutions for Overfitting and Underfitting
====================================================
This script demonstrates practical techniques to address
overfitting and underfitting problems including:
- Regularization (L1, L2)
- Cross-validation
- Feature engineering
- Early stopping
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, validation_curve, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

def generate_data(n_samples=200, n_features=1, noise=0.3):
    """Generate synthetic data"""
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, noise, n_samples)
    return X, y

# ============================================================================
# 1. REGULARIZATION TECHNIQUES
# ============================================================================

def compare_regularization(X_train, X_test, y_train, y_test):
    """Compare different regularization techniques"""
    
    # Create polynomial features
    degree = 15
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Different models
    models = {
        'No Regularization': LinearRegression(),
        'Ridge (L2, α=0.1)': Ridge(alpha=0.1),
        'Ridge (L2, α=1.0)': Ridge(alpha=1.0),
        'Ridge (L2, α=10.0)': Ridge(alpha=10.0),
        'Lasso (L1, α=0.01)': Lasso(alpha=0.01, max_iter=10000),
        'Lasso (L1, α=0.1)': Lasso(alpha=0.1, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train_poly, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Count non-zero coefficients (for Lasso)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_nonzero': n_nonzero,
            'y_test_pred': y_test_pred
        }
    
    return results, poly

def plot_regularization_comparison(X_train, X_test, y_train, y_test, results):
    """Visualize regularization effects"""
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx >= 8:
            break
            
        ax = axes[idx]
        
        # Sort for plotting
        sort_idx = np.argsort(X_test.ravel())
        
        # Plot
        ax.scatter(X_train, y_train, alpha=0.5, s=30, label='Train', color='blue')
        ax.scatter(X_test, y_test, alpha=0.5, s=30, label='Test', color='red')
        ax.plot(X_test[sort_idx], result['y_test_pred'][sort_idx], 
                'g-', linewidth=2, label='Prediction')
        
        ax.set_title(f"{name}\nTrain MSE: {result['train_mse']:.3f}, "
                    f"Test MSE: {result['test_mse']:.3f}\n"
                    f"Non-zero coef: {result['n_nonzero']}", 
                    fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 2. VALIDATION CURVES
# ============================================================================

def plot_validation_curve_alpha(X, y, model_class, param_name, param_range):
    """Plot validation curve for hyperparameter tuning"""
    
    # Create pipeline with polynomial features
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('scaler', StandardScaler()),
        ('model', model_class())
    ])
    
    train_scores, val_scores = validation_curve(
        pipeline, X, y,
        param_name=f'model__{param_name}',
        param_range=param_range,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Convert to positive MSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, 'o-', label='Training error', linewidth=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
    
    plt.semilogx(param_range, val_mean, 'o-', label='Validation error', linewidth=2)
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)
    
    # Find optimal parameter
    optimal_idx = np.argmin(val_mean)
    optimal_param = param_range[optimal_idx]
    plt.axvline(x=optimal_param, color='red', linestyle='--', 
                label=f'Optimal α = {optimal_param:.4f}')
    
    plt.xlabel(f'{param_name} (log scale)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title(f'Validation Curve - {model_class.__name__}', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf(), optimal_param

# ============================================================================
# 3. CROSS-VALIDATION
# ============================================================================

def compare_models_cv(X, y):
    """Compare models using cross-validation"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial (deg=3)': Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ]),
        'Ridge (α=1.0)': Pipeline([
            ('poly', PolynomialFeatures(degree=10)),
            ('ridge', Ridge(alpha=1.0))
        ]),
        'Lasso (α=0.1)': Pipeline([
            ('poly', PolynomialFeatures(degree=10)),
            ('lasso', Lasso(alpha=0.1, max_iter=10000))
        ])
    }
    
    results = {}
    
    print("\nCross-Validation Results:")
    print("="*70)
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, cv=5, 
            scoring='neg_mean_squared_error'
        )
        cv_scores = -cv_scores  # Convert to positive MSE
        
        results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
        
        print(f"{name:<30} MSE: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("="*70)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = range(len(results))
    means = [r['mean'] for r in results.values()]
    stds = [r['std'] for r in results.values()]
    labels = list(results.keys())
    
    ax.bar(positions, means, yerr=stds, capsize=10, alpha=0.7, color='steelblue')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Model Comparison using 5-Fold Cross-Validation', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_idx = np.argmin(means)
    ax.bar(best_idx, means[best_idx], color='green', alpha=0.7, 
           label='Best Model')
    ax.legend()
    
    plt.tight_layout()
    return fig, results

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def demonstrate_feature_engineering(X_train, X_test, y_train, y_test):
    """Show the impact of feature engineering"""
    
    configs = [
        ('Original Features', 1, None),
        ('Polynomial (deg=3)', 3, None),
        ('Polynomial (deg=5)', 5, None),
        ('Polynomial (deg=10)', 10, None),
        ('Polynomial (deg=10) + Ridge', 10, Ridge(alpha=1.0)),
        ('Polynomial (deg=15) + Ridge', 15, Ridge(alpha=0.5))
    ]
    
    results = {}
    
    for name, degree, model in configs:
        # Create features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Train model
        if model is None:
            model = LinearRegression()
        
        model.fit(X_train_poly, y_train)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
        test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
        
        results[name] = {
            'n_features': X_train_poly.shape[1],
            'train_mse': train_mse,
            'test_mse': test_mse,
            'overfitting': test_mse - train_mse
        }
    
    # Print results
    print("\nFeature Engineering Impact:")
    print("="*80)
    print(f"{'Configuration':<35} {'# Features':<12} {'Train MSE':<12} "
          f"{'Test MSE':<12} {'Gap':<10}")
    print("-"*80)
    
    for name, result in results.items():
        print(f"{name:<35} {result['n_features']:<12} "
              f"{result['train_mse']:<12.4f} {result['test_mse']:<12.4f} "
              f"{result['overfitting']:<10.4f}")
    
    print("="*80)
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Practical Solutions for Overfitting and Underfitting")
    print("="*70)
    
    # Generate data
    X, y = generate_data(n_samples=200, noise=0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 1. Regularization
    print("\n" + "="*70)
    print("1. REGULARIZATION TECHNIQUES")
    print("="*70)
    results, poly = compare_regularization(X_train, X_test, y_train, y_test)
    
    print(f"\n{'Model':<30} {'Train MSE':<12} {'Test MSE':<12} {'Gap':<10} {'# Coef':<10}")
    print("-"*70)
    for name, result in results.items():
        gap = result['test_mse'] - result['train_mse']
        print(f"{name:<30} {result['train_mse']:<12.4f} {result['test_mse']:<12.4f} "
              f"{gap:<10.4f} {result['n_nonzero']:<10}")
    
    fig1 = plot_regularization_comparison(X_train, X_test, y_train, y_test, results)
    plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
    print("\nRegularization comparison saved as 'regularization_comparison.png'")
    
    # 2. Validation Curves
    print("\n" + "="*70)
    print("2. HYPERPARAMETER TUNING WITH VALIDATION CURVES")
    print("="*70)
    
    alphas = np.logspace(-4, 2, 30)
    fig2, optimal_alpha = plot_validation_curve_alpha(X, y, Ridge, 'alpha', alphas)
    plt.savefig('validation_curve_ridge.png', dpi=150, bbox_inches='tight')
    print(f"Optimal Ridge alpha: {optimal_alpha:.4f}")
    print("Validation curve saved as 'validation_curve_ridge.png'")
    
    # 3. Cross-Validation
    print("\n" + "="*70)
    print("3. MODEL SELECTION WITH CROSS-VALIDATION")
    print("="*70)
    fig3, cv_results = compare_models_cv(X, y)
    plt.savefig('cross_validation_comparison.png', dpi=150, bbox_inches='tight')
    print("Cross-validation comparison saved as 'cross_validation_comparison.png'")
    
    # 4. Feature Engineering
    print("\n" + "="*70)
    print("4. FEATURE ENGINEERING IMPACT")
    print("="*70)
    fe_results = demonstrate_feature_engineering(X_train, X_test, y_train, y_test)
    
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Solutions for Common Problems")
    print("="*70)
    print("""
UNDERFITTING (High Bias):
✓ Increase model complexity (higher degree polynomial)
✓ Add more features
✓ Reduce regularization strength
✓ Train longer (more iterations)
✓ Use more powerful model architecture

OVERFITTING (High Variance):
✓ Get more training data
✓ Reduce model complexity
✓ Add regularization:
  • Ridge (L2): Shrinks coefficients
  • Lasso (L1): Feature selection
  • ElasticNet: Combination of L1 and L2
✓ Use cross-validation for model selection
✓ Early stopping
✓ Dropout (for neural networks)
✓ Ensemble methods

BEST PRACTICES:
• Always use cross-validation
• Plot learning curves to diagnose issues
• Use validation curves for hyperparameter tuning
• Start simple, then increase complexity
• Regularization is your friend!
    """)
    print("="*70)
