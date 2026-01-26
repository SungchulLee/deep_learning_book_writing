# Bias-Variance Decomposition

## Learning Objectives

By the end of this section, you will be able to:

- Understand the fundamental decomposition of prediction error into bias, variance, and irreducible noise
- Derive the bias-variance decomposition mathematically from first principles
- Recognize how model complexity affects bias and variance in opposite directions
- Apply this framework to diagnose and improve machine learning models
- Implement empirical bias-variance estimation using bootstrap methods in PyTorch

## Prerequisites

- Understanding of expected value and variance from probability theory
- Familiarity with supervised learning and regression problems
- Basic PyTorch tensor operations
- Knowledge of training/test splits and model evaluation

---

## 1. Introduction: The Fundamental Challenge

Every supervised learning problem faces a core challenge: **how do we build a model that performs well on data it has never seen?** The bias-variance decomposition provides a mathematical framework for understanding this challenge and the inherent tradeoffs involved.

When we train a model $\hat{f}(x)$ to approximate an unknown true function $f(x)$, our predictions will inevitably contain error. This error comes from three distinct sources:

1. **Bias**: Systematic error from incorrect assumptions in the learning algorithm
2. **Variance**: Error from sensitivity to fluctuations in the training data  
3. **Irreducible Error**: Noise inherent in the data that no model can eliminate

Understanding this decomposition is essential for diagnosing model problems and making informed decisions about model complexity, regularization, and data collection.

---

## 2. Problem Setup

### 2.1 The Data Generating Process

Assume our data follows the relationship:

$$y = f(x) + \epsilon$$

where:
- $x \in \mathbb{R}^d$ is the input feature vector
- $y \in \mathbb{R}$ is the target variable
- $f(x)$ is the **true underlying function** (unknown)
- $\epsilon$ is random noise with $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$

The noise $\epsilon$ is independent of $x$ and represents irreducible randomness in the data generating process.

### 2.2 The Learning Problem

Given a training dataset $\mathcal{D} = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ drawn from this distribution, we want to learn a function $\hat{f}(x; \mathcal{D})$ that approximates $f(x)$.

!!! note "Key Insight"
    The learned function $\hat{f}$ depends on the specific training data $\mathcal{D}$. Different training sets would produce different learned functions. This randomness in the training data is the source of **variance**.

### 2.3 Measuring Prediction Error

For a new test point $x_0$, we measure the expected squared prediction error:

$$\text{Expected Error} = \mathbb{E}_{\mathcal{D}, \epsilon}\left[(y_0 - \hat{f}(x_0; \mathcal{D}))^2\right]$$

This expectation is taken over:
1. The randomness in the training data $\mathcal{D}$
2. The randomness in the test observation noise $\epsilon$

---

## 3. The Bias-Variance Decomposition

### 3.1 Theorem Statement

**Theorem (Bias-Variance Decomposition):** For any fixed test point $x_0$, the expected squared prediction error decomposes as:

$$\boxed{\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}}$$

Or more compactly:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

### 3.2 Component Definitions

**Bias** measures how far off the average prediction is from the true value:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}_{\mathcal{D}}[\hat{f}(x; \mathcal{D})] - f(x)$$

- High bias indicates the model is systematically wrong
- This happens when the model is **too simple** to capture the true relationship
- Bias arises from restrictive modeling assumptions

**Variance** measures how much the predictions vary across different training sets:

$$\text{Var}[\hat{f}(x)] = \mathbb{E}_{\mathcal{D}}\left[(\hat{f}(x; \mathcal{D}) - \mathbb{E}[\hat{f}(x)])^2\right]$$

- High variance indicates the model is **sensitive to the training data**
- This happens when the model is **too complex** relative to the data
- Variance arises from overfitting to training set idiosyncrasies

**Irreducible Error** is the inherent noise in the data:

$$\sigma^2 = \text{Var}(\epsilon)$$

- This cannot be reduced by any model
- It represents the fundamental limit of prediction accuracy
- It sets the lower bound on achievable error

---

## 4. Mathematical Derivation

### 4.1 Setting Up the Derivation

Let's derive the decomposition rigorously. We start with the expected squared error and work through the algebra step by step.

**Step 1: Expand the squared term**

$$\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(f + \epsilon - \hat{f})^2]$$

Let $\bar{f} = \mathbb{E}[\hat{f}]$ denote the expected prediction (averaged over training sets).

**Step 2: Add and subtract $\bar{f}$**

$$= \mathbb{E}[(f - \bar{f} + \bar{f} - \hat{f} + \epsilon)^2]$$

**Step 3: Expand the square**

$$= \mathbb{E}[(f - \bar{f})^2 + (\bar{f} - \hat{f})^2 + \epsilon^2 + 2(f - \bar{f})(\bar{f} - \hat{f}) + 2(f - \bar{f})\epsilon + 2(\bar{f} - \hat{f})\epsilon]$$

### 4.2 Analyzing Each Term

**Term 1:** $(f - \bar{f})^2$

This is deterministic (given $x$), so:
$$\mathbb{E}[(f - \bar{f})^2] = (f - \bar{f})^2 = \text{Bias}^2$$

**Term 2:** $(\bar{f} - \hat{f})^2$

$$\mathbb{E}[(\bar{f} - \hat{f})^2] = \mathbb{E}[(\hat{f} - \bar{f})^2] = \text{Var}(\hat{f})$$

**Term 3:** $\epsilon^2$

$$\mathbb{E}[\epsilon^2] = \sigma^2$$

**Term 4:** $2(f - \bar{f})(\bar{f} - \hat{f})$

Since $(f - \bar{f})$ is deterministic:
$$\mathbb{E}[2(f - \bar{f})(\bar{f} - \hat{f})] = 2(f - \bar{f})\mathbb{E}[\bar{f} - \hat{f}] = 2(f - \bar{f}) \cdot 0 = 0$$

**Term 5:** $2(f - \bar{f})\epsilon$

Since $\epsilon$ is independent of everything else:
$$\mathbb{E}[2(f - \bar{f})\epsilon] = 2(f - \bar{f})\mathbb{E}[\epsilon] = 0$$

**Term 6:** $2(\bar{f} - \hat{f})\epsilon$

Since $\epsilon$ is independent of $\hat{f}$:
$$\mathbb{E}[2(\bar{f} - \hat{f})\epsilon] = 2\mathbb{E}[\bar{f} - \hat{f}]\mathbb{E}[\epsilon] = 0$$

### 4.3 Final Result

Combining all terms:

$$\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2 \quad \blacksquare$$

---

## 5. The Bias-Variance Tradeoff

### 5.1 The Core Insight

The bias-variance decomposition reveals a fundamental tradeoff in machine learning:

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Very Low (underfitting) | High | Low | High |
| Optimal | Medium | Medium | Minimum |
| Very High (overfitting) | Low | High | High |

This relationship is known as the **bias-variance tradeoff**.

### 5.2 Why the Tradeoff Exists

**Simple models (high bias, low variance):**
- Make strong assumptions about the data
- Cannot capture complex patterns → high bias
- Are stable across different training sets → low variance
- Example: Linear regression on non-linear data

**Complex models (low bias, high variance):**
- Make few assumptions about the data
- Can fit almost any pattern → low bias
- Are sensitive to training data noise → high variance
- Example: Deep neural networks with insufficient data

### 5.3 Visual Intuition

Consider fitting polynomial models of increasing degree to noisy data:

```
Degree 1 (Linear):     Degree 4 (Optimal):     Degree 15 (Overfit):
                       
    *  *                   *   *                    * ~ * 
  *      *  *           *        *              *~      ~*
 *   ___      *       *    ___     *          *~   ~~~    ~*
    /                     /   \                  /\  /\  
   /                     /     \                /  \/  \/
  
High Bias              Balanced                High Variance
Low Variance           Bias & Variance         Low Bias
```

---

## 6. PyTorch Implementation

### 6.1 Empirical Bias-Variance Estimation

We can estimate bias and variance empirically using **bootstrap sampling**:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_data(n_samples=200, noise_std=0.3):
    """
    Generate synthetic data from a known true function.
    
    y = sin(x) + noise
    
    This allows us to compute true bias since we know f(x).
    """
    X = torch.linspace(0, 10, n_samples)
    y_true = torch.sin(X)
    y = y_true + torch.randn(n_samples) * noise_std
    return X.reshape(-1, 1), y, y_true

def compute_bias_variance(X_train, y_train, X_test, y_test_true, 
                          max_depth, n_iterations=100):
    """
    Compute bias and variance using bootstrap sampling.
    
    The key idea: train the same model architecture on many different
    bootstrap samples, then analyze how predictions vary.
    
    Args:
        X_train: Training features (n_train, d)
        y_train: Training targets (n_train,)
        X_test: Test features (n_test, d)
        y_test_true: True function values at test points (n_test,)
        max_depth: Tree depth (controls model complexity)
        n_iterations: Number of bootstrap samples
    
    Returns:
        bias_squared: Squared bias at test points
        variance: Variance of predictions at test points
        predictions: All predictions (n_iterations, n_test)
    """
    predictions = []
    
    # Convert to numpy for sklearn
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    
    for i in range(n_iterations):
        # Bootstrap sample: sample with replacement
        indices = np.random.choice(len(X_train_np), 
                                   size=len(X_train_np), 
                                   replace=True)
        X_boot = X_train_np[indices]
        y_boot = y_train_np[indices]
        
        # Train model on bootstrap sample
        model = DecisionTreeRegressor(max_depth=max_depth, 
                                      random_state=i)
        model.fit(X_boot, y_boot)
        
        # Predict on test set
        y_pred = model.predict(X_test_np)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)  # (n_iterations, n_test)
    
    # Compute bias: E[f_hat(x)] - f(x)
    mean_prediction = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_prediction - y_test_true.numpy()) ** 2)
    
    # Compute variance: E[(f_hat(x) - E[f_hat(x)])^2]
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias_squared, variance, predictions, mean_prediction
```

### 6.2 Analyzing the Tradeoff Across Complexities

```python
def analyze_bias_variance_tradeoff(X_train, y_train, X_test, y_test_true,
                                    noise_variance=0.09):
    """
    Analyze how bias and variance change with model complexity.
    
    Args:
        noise_variance: Known σ² (irreducible error)
    """
    max_depths = range(1, 16)
    results = {
        'depths': list(max_depths),
        'bias_squared': [],
        'variance': [],
        'total_error': []
    }
    
    print("="*60)
    print("Bias-Variance Analysis Across Model Complexities")
    print("="*60)
    print(f"{'Depth':<8} {'Bias²':<12} {'Variance':<12} {'Total':<12}")
    print("-"*60)
    
    for depth in max_depths:
        bias_sq, var, _, _ = compute_bias_variance(
            X_train, y_train, X_test, y_test_true,
            max_depth=depth, n_iterations=50
        )
        
        # Total error = Bias² + Variance + σ²
        total = bias_sq + var + noise_variance
        
        results['bias_squared'].append(bias_sq)
        results['variance'].append(var)
        results['total_error'].append(total)
        
        print(f"{depth:<8} {bias_sq:<12.4f} {var:<12.4f} {total:<12.4f}")
    
    # Find optimal complexity
    optimal_idx = np.argmin(results['total_error'])
    optimal_depth = results['depths'][optimal_idx]
    
    print("="*60)
    print(f"Optimal Complexity: max_depth = {optimal_depth}")
    print(f"Minimum Total Error: {results['total_error'][optimal_idx]:.4f}")
    print("="*60)
    
    return results, optimal_depth
```

### 6.3 Visualization

```python
def plot_bias_variance_tradeoff(results, noise_variance=0.09):
    """
    Create the classic bias-variance tradeoff plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    depths = results['depths']
    bias_sq = results['bias_squared']
    variance = results['variance']
    total = results['total_error']
    
    # Plot 1: Line plot
    ax1 = axes[0]
    ax1.plot(depths, bias_sq, 'b-o', label='Bias²', linewidth=2, markersize=6)
    ax1.plot(depths, variance, 'r-s', label='Variance', linewidth=2, markersize=6)
    ax1.plot(depths, total, 'g-^', label='Total Error', linewidth=2, markersize=8)
    ax1.axhline(y=noise_variance, color='gray', linestyle=':', 
                label=f'Irreducible Error (σ²={noise_variance})')
    
    # Mark optimal point
    optimal_idx = np.argmin(total)
    ax1.axvline(x=depths[optimal_idx], color='purple', linestyle='--',
                label=f'Optimal Depth = {depths[optimal_idx]}')
    
    ax1.set_xlabel('Model Complexity (Max Depth)', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.5, 15.5])
    
    # Plot 2: Stacked bar chart
    ax2 = axes[1]
    x = np.arange(len(depths))
    
    # Stack: Irreducible + Bias² + Variance
    irreducible = [noise_variance] * len(depths)
    
    ax2.bar(x, irreducible, label='Irreducible Error', alpha=0.8, color='gray')
    ax2.bar(x, bias_sq, bottom=irreducible, label='Bias²', alpha=0.8, color='blue')
    bottom_for_var = [i + b for i, b in zip(irreducible, bias_sq)]
    ax2.bar(x, variance, bottom=bottom_for_var, label='Variance', alpha=0.8, color='red')
    
    ax2.set_xlabel('Model Complexity (Max Depth)', fontsize=12)
    ax2.set_ylabel('Error Components', fontsize=12)
    ax2.set_title('Error Decomposition', fontsize=14, fontweight='bold')
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([depths[i] for i in range(0, len(depths), 2)])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    set_seed(42)
    
    # Generate data with known true function
    X, y, y_true = generate_data(n_samples=200, noise_std=0.3)
    noise_variance = 0.3 ** 2  # Known irreducible error
    
    # Split data
    n_train = 140
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    y_test_true = y_true[n_train:]
    
    # Analyze bias-variance tradeoff
    results, optimal_depth = analyze_bias_variance_tradeoff(
        X_train, y_train, X_test, y_test_true, noise_variance
    )
    
    # Visualize
    fig = plot_bias_variance_tradeoff(results, noise_variance)
    plt.savefig('bias_variance_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 7. Practical Implications

### 7.1 Diagnostic Framework

The bias-variance decomposition provides a framework for diagnosing model problems:

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High train error, high test error | High Bias | Increase model complexity |
| Low train error, high test error | High Variance | Regularize or get more data |
| Low train error, low test error | Good balance | Model is working well |

### 7.2 Strategies for Reducing Bias

When your model has high bias (underfitting):

1. **Increase model complexity**: More layers, more parameters, higher polynomial degree
2. **Add features**: Engineer more informative features
3. **Reduce regularization**: Lower L1/L2 penalty
4. **Train longer**: Ensure convergence

### 7.3 Strategies for Reducing Variance

When your model has high variance (overfitting):

1. **Get more training data**: More data reduces variance
2. **Reduce model complexity**: Fewer parameters, shallower networks
3. **Add regularization**: L1, L2, dropout, early stopping
4. **Use ensemble methods**: Bagging averages over multiple models

### 7.4 The Double Descent Phenomenon

!!! warning "Modern Deep Learning"
    In highly overparameterized neural networks, classical bias-variance analysis breaks down. The **double descent** phenomenon shows that test error can decrease again as model complexity increases far beyond the interpolation threshold. This is an active area of research.

---

## 8. Connection to Other Concepts

### 8.1 Regularization

Regularization explicitly trades off bias and variance:

- **L2 regularization (Ridge)**: Adds $\lambda \|\mathbf{w}\|^2$ to the loss
  - Increases bias (shrinks weights toward zero)
  - Decreases variance (reduces model flexibility)
  
- **L1 regularization (Lasso)**: Adds $\lambda \|\mathbf{w}\|_1$ to the loss
  - Performs feature selection
  - Can reduce both bias (by selecting relevant features) and variance

### 8.2 Ensemble Methods

**Bagging** (Bootstrap Aggregating):
- Trains multiple models on bootstrap samples
- Averages predictions
- Primarily **reduces variance** while keeping bias approximately constant

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B \hat{f}_b\right) \approx \frac{1}{B}\text{Var}(\hat{f})$$

**Boosting**:
- Sequentially fits models to residuals
- Primarily **reduces bias** by building increasingly complex combinations

### 8.3 Cross-Validation

Cross-validation provides an estimate of **total expected error** (bias² + variance + irreducible error). It doesn't directly decompose these components, but:

- High CV error with low training error → high variance
- High CV error with high training error → high bias

---

## 9. Summary

### Key Takeaways

1. **The Decomposition**: Expected error = Bias² + Variance + Irreducible Error

2. **Bias** comes from model assumptions being wrong; **Variance** comes from sensitivity to training data

3. **The Tradeoff**: Increasing model complexity decreases bias but increases variance

4. **Optimal Complexity**: The best model minimizes total error, balancing bias and variance

5. **Practical Diagnostics**: Compare training and test error to identify the dominant error source

### What's Next

In the next section, we'll derive the bias-variance decomposition from an alternative perspective using statistical learning theory and explore its connections to the generalization bounds in learning theory.

---

## 10. Exercises

### Exercise 1: Manual Derivation
Derive the bias-variance decomposition for the special case of a linear model $\hat{f}(x) = \mathbf{w}^T \mathbf{x}$ with squared error loss. Show how the covariance matrix of the weights affects the variance term.

### Exercise 2: Bootstrap Implementation
Modify the PyTorch implementation to use neural networks instead of decision trees. Analyze how network depth affects the bias-variance tradeoff.

### Exercise 3: Theoretical Analysis
For a polynomial regression model of degree $d$ fit to $n$ data points:
- What happens to bias as $d \to \infty$?
- What happens to variance as $d \to n$?
- Find the optimal $d$ as a function of $n$ and noise level.

### Exercise 4: Double Descent
Implement an experiment that demonstrates double descent in a neural network. Plot test error as a function of model width and identify the interpolation threshold.

---

## References

1. Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural Networks and the Bias/Variance Dilemma. *Neural Computation*, 4(1), 1-58.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 7.

3. Belkin, M., et al. (2019). Reconciling modern machine learning practice and the bias-variance trade-off. *PNAS*, 116(32), 15849-15854.
