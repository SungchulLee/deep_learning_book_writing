# Overfitting Detection

## Learning Objectives

By the end of this section, you will be able to:

- Recognize the symptoms of overfitting and underfitting in machine learning models
- Use learning curves as a diagnostic tool to identify model problems
- Distinguish between high bias and high variance situations
- Apply systematic approaches to diagnose and fix model issues
- Implement learning curve analysis in PyTorch

## Prerequisites

- Understanding of bias-variance decomposition
- Familiarity with training/validation/test splits
- Basic PyTorch training loops
- Experience with model evaluation metrics

---

## 1. Introduction: The Generalization Challenge

The fundamental goal of machine learning is **generalization**: performing well on data the model has never seen. Overfitting occurs when a model learns the training data too well—including its noise and peculiarities—at the expense of generalization.

### 1.1 Defining the Problems

**Overfitting (High Variance):**
The model captures noise in the training data rather than the underlying pattern.

- Training error: Very low
- Test error: Much higher than training error
- The model has memorized the training data

**Underfitting (High Bias):**
The model is too simple to capture the underlying pattern in the data.

- Training error: High
- Test error: Also high (similar to training error)
- The model cannot learn the relationship

**Good Fit:**
The model captures the underlying pattern without fitting the noise.

- Training error: Reasonably low
- Test error: Close to training error
- The model generalizes well

---

## 2. Diagnostic Symptoms

### 2.1 The Error Gap Diagnostic

The most reliable diagnostic is comparing training and validation/test error:

| Situation | Training Error | Test Error | Gap | Diagnosis |
|-----------|---------------|------------|-----|-----------|
| Underfitting | High | High | Small | High Bias |
| Good Fit | Low | Low | Small | Balanced |
| Overfitting | Low | High | Large | High Variance |

### 2.2 Visual Patterns

**Underfitting Example:**
```
True Function:    Model Fit:
    ●               ____
   ● ●             /
  ●   ●           /
 ●     ●         /
●       ●       /

Model is too simple (linear) for nonlinear data
```

**Overfitting Example:**
```
True Function:    Model Fit:
    ●                 ∿
   ● ●              ∿   ∿
  ●   ●           ∿  ●  ∿
 ●     ●         ∿ ●  ● ∿
●       ●       ∿●      ●∿

Model fits every point (including noise)
```

### 2.3 Quantitative Thresholds

While context-dependent, general rules of thumb:

- **Large gap (potential overfitting):** Test error > 1.5 × Training error
- **Both errors high (potential underfitting):** Errors significantly above expected irreducible noise
- **Validation error increasing:** Training too long (overfitting)

---

## 3. Learning Curves: The Primary Diagnostic Tool

### 3.1 What Are Learning Curves?

Learning curves plot model performance as a function of training set size. They reveal how training and validation errors evolve.

```
Error
  ↑
  │ ╲
  │  ╲ _________________ Validation Error
  │   ╲_______________/
  │    ╲
  │     ╲______________ Training Error
  │
  └─────────────────────→ Training Set Size
```

### 3.2 Interpreting Learning Curves

**Pattern 1: High Bias (Underfitting)**

```
Error
  ↑
  │ ────────────────── Validation (high plateau)
  │ ────────────────── Training (high plateau)
  │
  └────────────────────→ Training Set Size
```

**Characteristics:**
- Both curves plateau at high error
- Small gap between curves
- More data won't help

**Solutions:**
- Increase model complexity
- Add more features
- Reduce regularization

**Pattern 2: High Variance (Overfitting)**

```
Error
  ↑
  │ ╲
  │  ╲_________________ Validation (high)
  │
  │ ____________________Training (low)
  └────────────────────→ Training Set Size
```

**Characteristics:**
- Large gap between curves
- Training error is low
- Validation error stays high
- Gap may narrow with more data

**Solutions:**
- Get more training data
- Reduce model complexity
- Add regularization
- Use dropout/early stopping

**Pattern 3: Good Fit**

```
Error
  ↑
  │ ╲
  │  ╲______╲__________ Validation
  │   ╲______╲_________Training
  │          (converging)
  └────────────────────→ Training Set Size
```

**Characteristics:**
- Both curves converge to low error
- Small gap between curves
- Model generalizes well

### 3.3 Learning Curves Over Training Time

Another diagnostic plots error vs. training epochs:

```
Error
  ↑
  │╲
  │ ╲
  │  ╲______________________ Validation (starts rising!)
  │   ╲
  │    ╲__________________ Training (keeps decreasing)
  │
  └───────────────────────→ Epochs
         ↑
    Early stopping point
```

When validation error starts increasing while training error decreases, the model is overfitting.

---

## 4. PyTorch Implementation

### 4.1 Learning Curve Generator

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

class FlexibleMLP(nn.Module):
    """
    MLP with configurable complexity for demonstrating overfitting.
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

def generate_synthetic_data(n_samples=1000, noise_std=0.3):
    """
    Generate synthetic regression data with a known pattern.
    
    y = sin(x) + 0.5*x + noise
    """
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = np.sin(X.ravel()) + 0.5 * X.ravel()
    y = y_true + np.random.normal(0, noise_std, n_samples)
    
    return torch.FloatTensor(X), torch.FloatTensor(y), y_true

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, lr=0.01, verbose=False):
    """
    Train model and track both training and validation loss.
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
    
    return train_losses, val_losses

def compute_learning_curve(model_fn, X, y, train_sizes, cv_folds=5):
    """
    Compute learning curves by varying training set size.
    
    Args:
        model_fn: Function that returns a fresh model instance
        X: Full feature tensor
        y: Full target tensor
        train_sizes: List of training set sizes to evaluate
        cv_folds: Number of cross-validation folds
        
    Returns:
        train_sizes: Actual training sizes used
        train_means: Mean training error for each size
        train_stds: Std of training error
        val_means: Mean validation error for each size
        val_stds: Std of validation error
    """
    train_means = []
    train_stds = []
    val_means = []
    val_stds = []
    
    for train_size in train_sizes:
        train_scores = []
        val_scores = []
        
        for fold in range(cv_folds):
            # Create train/val split
            np.random.seed(fold)
            indices = np.random.permutation(len(X))
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + len(indices)//5]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Train model
            model = model_fn()
            train_losses, val_losses = train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=100, verbose=False
            )
            
            train_scores.append(train_losses[-1])
            val_scores.append(val_losses[-1])
        
        train_means.append(np.mean(train_scores))
        train_stds.append(np.std(train_scores))
        val_means.append(np.mean(val_scores))
        val_stds.append(np.std(val_scores))
    
    return (np.array(train_sizes), 
            np.array(train_means), np.array(train_stds),
            np.array(val_means), np.array(val_stds))
```

### 4.2 Visualization Functions

```python
def plot_learning_curve(train_sizes, train_means, train_stds, 
                        val_means, val_stds, title="Learning Curve"):
    """
    Plot learning curve with confidence bands.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Training curve
    ax.plot(train_sizes, train_means, 'o-', color='blue', 
            label='Training Error', linewidth=2)
    ax.fill_between(train_sizes, 
                    train_means - train_stds,
                    train_means + train_stds,
                    alpha=0.2, color='blue')
    
    # Validation curve
    ax.plot(train_sizes, val_means, 'o-', color='red',
            label='Validation Error', linewidth=2)
    ax.fill_between(train_sizes,
                    val_means - val_stds,
                    val_means + val_stds,
                    alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add diagnosis
    final_gap = val_means[-1] - train_means[-1]
    final_val = val_means[-1]
    
    if final_val > 0.3 and final_gap < 0.1:
        diagnosis = "HIGH BIAS (Underfitting)"
        color = 'orange'
    elif final_gap > 0.15:
        diagnosis = "HIGH VARIANCE (Overfitting)"
        color = 'red'
    else:
        diagnosis = "GOOD FIT"
        color = 'green'
    
    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    return fig

def plot_training_dynamics(train_losses, val_losses, title="Training Dynamics"):
    """
    Plot loss curves over training epochs.
    Useful for detecting when overfitting begins.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Detect potential early stopping point
    min_val_idx = np.argmin(val_losses)
    ax.axvline(min_val_idx + 1, color='green', linestyle='--',
               label=f'Best Validation (Epoch {min_val_idx + 1})')
    
    # Check if validation increased after minimum
    if val_losses[-1] > val_losses[min_val_idx] * 1.1:
        ax.fill_betweenx([0, max(val_losses)],
                         min_val_idx + 1, len(epochs),
                         alpha=0.2, color='red',
                         label='Overfitting Region')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### 4.3 Complete Diagnostic Suite

```python
def diagnose_model(model_fn, X, y, model_name="Model"):
    """
    Run complete overfitting diagnosis for a model.
    
    Produces:
    1. Learning curves (error vs. training size)
    2. Training dynamics (error vs. epochs)
    3. Numerical diagnosis
    """
    print("="*60)
    print(f"Diagnosing: {model_name}")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, random_state=42
    )
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    
    # Train single model for dynamics
    model = model_fn()
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        epochs=200, verbose=True
    )
    
    # Compute learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    train_sizes = (train_sizes * len(X_train)).astype(int)
    
    sizes, train_means, train_stds, val_means, val_stds = compute_learning_curve(
        model_fn, X, y, train_sizes, cv_folds=3
    )
    
    # Diagnosis
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    gap = final_val - final_train
    
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    print(f"Final Training Loss:   {final_train:.4f}")
    print(f"Final Validation Loss: {final_val:.4f}")
    print(f"Gap:                   {gap:.4f}")
    print()
    
    if final_val > 0.3 and gap < 0.1:
        print("→ HIGH BIAS (Underfitting)")
        print("  Symptoms:")
        print("    • Both training and validation errors are high")
        print("    • Small gap between curves")
        print("    • Learning curves plateau early")
        print("  Recommendations:")
        print("    • Increase model capacity (more layers/neurons)")
        print("    • Add more features")
        print("    • Reduce regularization")
        print("    • Train longer")
    elif gap > 0.15:
        print("→ HIGH VARIANCE (Overfitting)")
        print("  Symptoms:")
        print("    • Large gap between training and validation")
        print("    • Low training error, high validation error")
        print("    • Validation may worsen over training")
        print("  Recommendations:")
        print("    • Get more training data")
        print("    • Reduce model complexity")
        print("    • Add regularization (L2, dropout)")
        print("    • Use early stopping")
        print("    • Use data augmentation")
    else:
        print("→ GOOD FIT")
        print("  • Model is generalizing well")
        print("  • Consider small improvements or deployment")
    
    print("="*60)
    
    # Create visualizations
    fig1 = plot_training_dynamics(train_losses, val_losses, 
                                  f"{model_name}: Training Dynamics")
    fig2 = plot_learning_curve(sizes, train_means, train_stds,
                               val_means, val_stds,
                               f"{model_name}: Learning Curve")
    
    return fig1, fig2, {'train_loss': final_train, 'val_loss': final_val, 'gap': gap}
```

### 4.4 Demonstration

```python
def demonstrate_overfitting_detection():
    """
    Demonstrate overfitting detection with three model types.
    """
    set_seed(42)
    X, y, _ = generate_synthetic_data(n_samples=500, noise_std=0.3)
    
    # Model 1: Underfitting (too simple)
    def simple_model():
        return FlexibleMLP(input_dim=1, hidden_dims=[4], dropout=0.0)
    
    # Model 2: Overfitting (too complex)
    def complex_model():
        return FlexibleMLP(input_dim=1, hidden_dims=[128, 128, 128], dropout=0.0)
    
    # Model 3: Good fit (appropriate complexity with regularization)
    def balanced_model():
        return FlexibleMLP(input_dim=1, hidden_dims=[32, 32], dropout=0.2)
    
    models = [
        (simple_model, "Simple Model (High Bias)"),
        (complex_model, "Complex Model (High Variance)"),
        (balanced_model, "Balanced Model (Good Fit)")
    ]
    
    all_results = []
    all_figures = []
    
    for model_fn, name in models:
        fig1, fig2, results = diagnose_model(model_fn, X, y, name)
        all_results.append(results)
        all_figures.extend([fig1, fig2])
        
        # Save figures
        fig1.savefig(f'{name.replace(" ", "_")}_dynamics.png', 
                     dpi=150, bbox_inches='tight')
        fig2.savefig(f'{name.replace(" ", "_")}_learning_curve.png',
                     dpi=150, bbox_inches='tight')
    
    plt.show()
    return all_results

if __name__ == "__main__":
    results = demonstrate_overfitting_detection()
```

---

## 5. Advanced Diagnostic Techniques

### 5.1 Validation Curves

Validation curves show how performance changes with a hyperparameter:

```python
def plot_validation_curve(model_fn, X, y, param_name, param_range):
    """
    Plot validation curve for hyperparameter tuning.
    
    Shows how training and validation error change with a hyperparameter.
    """
    train_scores = []
    val_scores = []
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    for param_value in param_range:
        # Create model with this parameter
        model = model_fn(param_value)
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=100, verbose=False
        )
        train_scores.append(train_losses[-1])
        val_scores.append(val_losses[-1])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_range, train_scores, 'b-o', label='Training Error')
    ax.plot(param_range, val_scores, 'r-o', label='Validation Error')
    
    # Mark optimal
    opt_idx = np.argmin(val_scores)
    ax.axvline(param_range[opt_idx], color='green', linestyle='--',
               label=f'Optimal {param_name} = {param_range[opt_idx]}')
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title(f'Validation Curve: {param_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, param_range[opt_idx]
```

### 5.2 Residual Analysis

Examining prediction residuals can reveal patterns:

```python
def analyze_residuals(model, X_train, y_train, X_test, y_test):
    """
    Analyze residuals for signs of underfitting.
    
    Random residuals → good fit
    Systematic patterns → underfitting (model missing structure)
    """
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)
    
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals vs. predicted
    axes[0].scatter(y_test_pred.numpy(), test_residuals.numpy(), alpha=0.5)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs. Predicted')
    
    # Residual histogram
    axes[1].hist(test_residuals.numpy(), bins=30, edgecolor='black')
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    
    # QQ plot (simplified)
    sorted_residuals = np.sort(test_residuals.numpy())
    n = len(sorted_residuals)
    theoretical_quantiles = np.linspace(-2, 2, n)
    axes[2].scatter(theoretical_quantiles, sorted_residuals, alpha=0.5)
    axes[2].plot([-2, 2], [-2, 2], 'r--')
    axes[2].set_xlabel('Theoretical Quantiles')
    axes[2].set_ylabel('Sample Quantiles')
    axes[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    return fig
```

### 5.3 Cross-Validation Analysis

```python
def cross_validation_analysis(model_fn, X, y, n_folds=5):
    """
    Perform k-fold cross-validation and analyze variance across folds.
    
    High variance across folds → high model variance (overfitting tendency)
    """
    fold_size = len(X) // n_folds
    fold_train_errors = []
    fold_val_errors = []
    
    for fold in range(n_folds):
        # Create fold indices
        val_start = fold * fold_size
        val_end = val_start + fold_size
        
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, len(X)))
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Train
        model = model_fn()
        train_losses, val_losses = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=100, verbose=False
        )
        
        fold_train_errors.append(train_losses[-1])
        fold_val_errors.append(val_losses[-1])
    
    # Analysis
    print(f"\n{'Fold':<6} {'Train Error':<15} {'Val Error':<15}")
    print("-" * 40)
    for i, (te, ve) in enumerate(zip(fold_train_errors, fold_val_errors)):
        print(f"{i+1:<6} {te:<15.4f} {ve:<15.4f}")
    print("-" * 40)
    print(f"{'Mean':<6} {np.mean(fold_train_errors):<15.4f} "
          f"{np.mean(fold_val_errors):<15.4f}")
    print(f"{'Std':<6} {np.std(fold_train_errors):<15.4f} "
          f"{np.std(fold_val_errors):<15.4f}")
    
    # High std in validation → high variance model
    if np.std(fold_val_errors) > 0.05:
        print("\n⚠️ High variance across folds - model may be overfitting")
    
    return fold_train_errors, fold_val_errors
```

---

## 6. Quick Diagnostic Checklist

### 6.1 Initial Assessment

- [ ] Compare training and validation/test error
- [ ] Calculate the gap: $\text{gap} = \text{test error} - \text{train error}$
- [ ] Check if both errors are acceptable (low enough for the task)

### 6.2 If Gap is Large (Potential Overfitting)

- [ ] Plot learning curves with training set size
- [ ] Check if validation error decreases with more data
- [ ] Plot training dynamics over epochs
- [ ] Identify when validation error starts increasing
- [ ] Check model complexity relative to data size

### 6.3 If Both Errors are High (Potential Underfitting)

- [ ] Check if model converged (enough epochs)
- [ ] Plot residuals for systematic patterns
- [ ] Verify feature representation is adequate
- [ ] Check if learning rate is appropriate

### 6.4 Before Declaring Good Fit

- [ ] Run cross-validation to check stability
- [ ] Test on held-out test set
- [ ] Check performance across different data subsets
- [ ] Verify no data leakage

---

## 7. Summary

### Key Diagnostic Tools

1. **Train/Test Error Comparison**: The fundamental diagnostic
2. **Learning Curves**: How error changes with training set size
3. **Training Dynamics**: How error changes over epochs
4. **Validation Curves**: How error changes with hyperparameters
5. **Residual Analysis**: Patterns indicate model deficiencies
6. **Cross-Validation**: Stability of error estimates

### Diagnostic Decision Tree

```
                 Is training error acceptable?
                        /            \
                      No              Yes
                      |                |
              Underfitting      Is test error similar?
                                    /        \
                                  Yes         No
                                   |           |
                              Good Fit     Overfitting
```

### When in Doubt

- Plot learning curves first
- Check both error magnitude AND gap
- Consider the irreducible noise level
- Use cross-validation for reliable estimates

---

## Exercises

### Exercise 1: Learning Curve Analysis
Generate a learning curve for a decision tree classifier on a binary classification problem. Vary the `max_depth` parameter and show how the learning curves change.

### Exercise 2: Early Stopping Implementation
Modify the training function to implement early stopping based on validation loss. Use a patience parameter to decide when to stop.

### Exercise 3: Diagnostic Report
Create a function that automatically generates a PDF report with all diagnostic plots and recommendations for a given model and dataset.

### Exercise 4: Real Data Analysis
Apply the diagnostic techniques to a real dataset (e.g., California Housing). Identify if the initial model overfits or underfits, then apply appropriate remedies.

---

## References

1. Ng, A. (2018). Machine Learning Yearning. deeplearning.ai.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer. Chapter 7.

3. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly. Chapter 4.
