# Random Search

## Overview

Random Search is a hyperparameter optimization technique that samples configurations randomly from a defined search space. Unlike Grid Search's exhaustive enumeration, Random Search draws a fixed number of parameter combinations from specified distributions, making it significantly more efficient for high-dimensional spaces.

**Key Insight**: Bergstra and Bengio (2012) demonstrated that Random Search is more efficient than Grid Search because hyperparameters often have varying importance. Random Search explores diverse values of important parameters while Grid Search wastes evaluations on unimportant ones.

**Key Characteristics:**

- **Stochastic Sampling**: Randomly samples from parameter distributions
- **Continuous Support**: Can search continuous parameter spaces directly
- **Efficiency**: Explores diverse parameter combinations with fewer evaluations
- **Scalability**: Computational cost grows linearly with sample size, not grid size

## Mathematical Formulation

### Search Space Definition

For $d$ hyperparameters, define a probability distribution $p_i(\theta_i)$ over each parameter's domain $\Theta_i$. The joint search distribution is:

$$p(\boldsymbol{\theta}) = \prod_{i=1}^{d} p_i(\theta_i)$$

This independence assumption simplifies sampling while allowing different distribution types per parameter.

### Sampling Process

Random Search draws $N$ independent samples:

$$\boldsymbol{\theta}^{(j)} \sim p(\boldsymbol{\theta}), \quad j = 1, 2, \ldots, N$$

### Optimization Objective

The best configuration is:

$$\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta}^{(j)}, \; j \in \{1,\ldots,N\}}{\arg\max} \; \hat{J}(\boldsymbol{\theta}^{(j)})$$

where $\hat{J}(\boldsymbol{\theta})$ is the cross-validated performance estimate.

### Theoretical Justification

**Probability of Finding Good Configuration**: Given that a fraction $\gamma$ of the search space contains "good" configurations (above threshold performance), the probability of finding at least one good configuration in $N$ samples is:

$$P(\text{success}) = 1 - (1 - \gamma)^N$$

For example, if $\gamma = 0.05$ (5% of space is good) and $N = 60$ samples:

$$P(\text{success}) = 1 - (0.95)^{60} \approx 0.954$$

**Grid Search Comparison**: For a $d$-dimensional grid with $k$ points per dimension, finding a configuration within the top $1/k$ of each parameter requires all parameters aligned, giving probability $1/k^d$. Random Search samples independently, giving probability approximately $d/k$ for finding a good value in at least one important dimension.

## Algorithm

```
Algorithm: Random Search with Cross-Validation
──────────────────────────────────────────────
Input: Parameter distributions P, Dataset D, Number of samples N,
       Number of folds K, Scoring metric L, Random seed s
Output: Best parameters θ*, Best score J*

1. Initialize: best_score ← -∞, best_params ← None
2. Set random seed: seed(s)
3. For j = 1 to N:
   a. Sample configuration: θ^(j) ~ P
   b. Split D into K folds: {(D_train^k, D_val^k)}_{k=1}^K
   c. scores ← []
   d. For k = 1 to K:
      i.   Train model f_{θ^(j)} on D_train^k
      ii.  Evaluate: score_k ← L(f_{θ^(j)}, D_val^k)
      iii. Append score_k to scores
   e. mean_score ← mean(scores)
   f. If mean_score > best_score:
      i.  best_score ← mean_score
      ii. best_params ← θ^(j)
4. Return best_params, best_score
```

## Common Distributions

### Continuous Distributions

| Distribution | Use Case | scipy.stats |
|--------------|----------|-------------|
| Uniform | Linear scale parameters | `uniform(loc, scale)` |
| Log-Uniform | Learning rates, regularization | `loguniform(low, high)` |
| Normal | Parameters with known center | `norm(loc, scale)` |
| Truncated Normal | Bounded parameters | `truncnorm(a, b, loc, scale)` |

### Discrete Distributions

| Distribution | Use Case | scipy.stats |
|--------------|----------|-------------|
| Randint | Integer parameters | `randint(low, high)` |
| Choice | Categorical options | List of values |

### Log-Uniform Distribution

For parameters spanning multiple orders of magnitude (learning rate, regularization strength), log-uniform sampling ensures equal probability per order of magnitude:

$$\theta \sim \text{LogUniform}(a, b) \implies \log(\theta) \sim \text{Uniform}(\log a, \log b)$$

```python
from scipy.stats import loguniform

# Sample learning rate from [0.0001, 0.1]
learning_rate_dist = loguniform(1e-4, 1e-1)

# Verify log-uniform property
samples = learning_rate_dist.rvs(10000)
import numpy as np
log_samples = np.log10(samples)
# log_samples should be approximately uniform in [-4, -1]
```

## Implementation

### Basic Random Search with scikit-learn

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 500),           # Integer [50, 499]
    'max_depth': [None, 10, 20, 30, 40, 50],    # Categorical
    'min_samples_split': randint(2, 20),         # Integer [2, 19]
    'min_samples_leaf': randint(1, 10),          # Integer [1, 9]
    'max_features': ['sqrt', 'log2', None],      # Categorical
    'bootstrap': [True, False]                   # Binary
}

# Create Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,              # Number of random samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,         # Reproducibility
    return_train_score=True
)

# Execute search
random_search.fit(X_train, y_train)

# Results
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")
print(f"Test Score: {random_search.score(X_test, y_test):.4f}")
```

### Using Log-Uniform for Learning Rates

```python
from scipy.stats import loguniform
from sklearn.ensemble import GradientBoostingClassifier

# Parameters with appropriate distributions
gb_distributions = {
    'n_estimators': randint(50, 300),
    'learning_rate': loguniform(0.01, 0.3),    # Log-uniform: [0.01, 0.3]
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'subsample': uniform(0.6, 0.4),            # Uniform: [0.6, 1.0]
    'max_features': ['sqrt', 'log2', None]
}

random_search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=gb_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search_gb.fit(X_train, y_train)
print(f"Best learning_rate: {random_search_gb.best_params_['learning_rate']:.6f}")
```

### Analyzing Random Search Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert results to DataFrame
results_df = pd.DataFrame(random_search.cv_results_)

# View top configurations
results_df = results_df.sort_values('rank_test_score')
print("\nTop 10 Configurations:")
print(results_df[['params', 'mean_test_score', 'std_test_score']].head(10))

# Scatter plot: Parameter value vs. Score
def plot_param_vs_score(results_df, param_name):
    """Visualize relationship between parameter and performance."""
    param_col = f'param_{param_name}'
    
    if param_col not in results_df.columns:
        print(f"Parameter {param_name} not found")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Handle categorical parameters
    param_values = results_df[param_col]
    if param_values.dtype == object or isinstance(param_values.iloc[0], str):
        # Categorical: box plot
        unique_vals = param_values.unique()
        data = [results_df[results_df[param_col] == v]['mean_test_score'] 
                for v in unique_vals]
        plt.boxplot(data, labels=unique_vals)
        plt.xlabel(param_name)
    else:
        # Numerical: scatter plot
        plt.scatter(
            results_df[param_col], 
            results_df['mean_test_score'],
            alpha=0.6, c=results_df['rank_test_score'], 
            cmap='viridis_r'
        )
        plt.colorbar(label='Rank')
        plt.xlabel(param_name)
    
    plt.ylabel('Mean CV Score')
    plt.title(f'Effect of {param_name} on Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualize key parameters
for param in ['n_estimators', 'max_depth', 'min_samples_split']:
    plot_param_vs_score(results_df, param)
```

### Comparing Different Sample Sizes

```python
def compare_n_iter(X, y, param_distributions, n_iters, cv=3):
    """Compare Random Search performance across different sample sizes."""
    results = []
    
    for n_iter in n_iters:
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        import time
        start = time.time()
        random_search.fit(X, y)
        elapsed = time.time() - start
        
        results.append({
            'n_iter': n_iter,
            'best_score': random_search.best_score_,
            'time': elapsed
        })
        
        print(f"n_iter={n_iter:3d}: Score={random_search.best_score_:.4f}, "
              f"Time={elapsed:.2f}s")
    
    return pd.DataFrame(results)

# Compare sample sizes
n_iters = [10, 25, 50, 100, 200]
comparison = compare_n_iter(X_train, y_train, param_distributions, n_iters)

# Visualize diminishing returns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(comparison['n_iter'], comparison['best_score'], 'bo-', linewidth=2)
ax1.set_xlabel('Number of Iterations')
ax1.set_ylabel('Best CV Score')
ax1.set_title('Score vs. Sample Size')
ax1.grid(True, alpha=0.3)

ax2.plot(comparison['n_iter'], comparison['time'], 'ro-', linewidth=2)
ax2.set_xlabel('Number of Iterations')
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Time vs. Sample Size')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Custom Random Search Implementation

Understanding the internals helps with extensions:

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, List, Callable

class CustomRandomSearch:
    """Custom Random Search with flexible sampling."""
    
    def __init__(self, estimator, param_distributions: Dict[str, Any],
                 n_iter: int = 100, cv: int = 5, scoring: str = 'accuracy',
                 random_state: int = None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def _sample_params(self) -> Dict[str, Any]:
        """Sample a single configuration from distributions."""
        params = {}
        for param, dist in self.param_distributions.items():
            if hasattr(dist, 'rvs'):
                # scipy distribution
                params[param] = dist.rvs(random_state=self.rng)
            elif isinstance(dist, list):
                # Categorical: uniform choice
                params[param] = self.rng.choice(dist)
            else:
                params[param] = dist
        return params
    
    def fit(self, X, y):
        """Execute random search."""
        self.rng = np.random.RandomState(self.random_state)
        
        results = []
        best_score = -np.inf
        
        for i in range(self.n_iter):
            params = self._sample_params()
            
            # Clone estimator with new params
            model = clone(self.estimator)
            model.set_params(**params)
            
            # Cross-validate
            scores = cross_val_score(
                model, X, y, cv=self.cv, scoring=self.scoring
            )
            mean_score = scores.mean()
            std_score = scores.std()
            
            results.append({
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = best_score
        
        self.cv_results_ = pd.DataFrame(results)
        
        # Fit final model with best params
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        return self
    
    def score(self, X, y):
        """Score on test data."""
        return self.best_estimator_.score(X, y)

from sklearn.base import clone

# Use custom implementation
custom_search = CustomRandomSearch(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    random_state=42
)
custom_search.fit(X_train, y_train)
print(f"Custom Search Best Score: {custom_search.best_score_:.4f}")
```

## Halving Random Search

For faster convergence, successively halve the number of candidates while increasing resources:

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# Halving Random Search
halving_search = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_candidates=100,        # Initial number of candidates
    factor=3,                # Halving factor
    resource='n_estimators', # Resource to increase
    min_resources=50,        # Minimum resource per candidate
    max_resources=500,       # Maximum resource
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

halving_search.fit(X_train, y_train)

print(f"\nHalving Random Search Results:")
print(f"Best Parameters: {halving_search.best_params_}")
print(f"Best Score: {halving_search.best_score_:.4f}")
print(f"Number of iterations: {halving_search.n_iterations_}")
```

## Random Search vs. Grid Search Comparison

```python
from sklearn.model_selection import GridSearchCV
import time

# Equivalent parameter space
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15, 20],
    'max_features': ['sqrt', 'log2', None]
}

total_grid = np.prod([len(v) for v in param_grid.values()])
print(f"Grid Search total configurations: {total_grid}")

# Grid Search
start = time.time()
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start

# Random Search with same number of iterations
start = time.time()
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions={
        'n_estimators': randint(50, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'max_features': ['sqrt', 'log2', None]
    },
    n_iter=50,  # Only 50 samples vs 540 grid points
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
random_time = time.time() - start

print(f"\nComparison Results:")
print(f"{'Method':<15} {'Score':<10} {'Time':<10} {'Evaluations':<12}")
print("-" * 50)
print(f"{'Grid Search':<15} {grid_search.best_score_:.4f}     {grid_time:.2f}s     {total_grid}")
print(f"{'Random Search':<15} {random_search.best_score_:.4f}     {random_time:.2f}s     50")
print(f"\nTime Savings: {(1 - random_time/grid_time)*100:.1f}%")
print(f"Score Difference: {abs(grid_search.best_score_ - random_search.best_score_):.4f}")
```

## Advantages and Limitations

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Efficiency** | Better exploration per evaluation than Grid Search |
| **Scalability** | Linear cost in sample size, not grid size |
| **Continuous Support** | Direct sampling from continuous distributions |
| **Flexibility** | Custom distributions for different parameter types |
| **Anytime Property** | Can stop early and use best result so far |

### Limitations

| Limitation | Description |
|------------|-------------|
| **Non-Adaptive** | Does not learn from previous evaluations |
| **No Guarantees** | May miss optimal configuration |
| **Reproducibility** | Requires random seed for reproducibility |
| **Uniform Exploration** | Samples all parameters equally regardless of importance |

## Guidelines for Effective Random Search

### Choosing n_iter

```python
def recommended_n_iter(d: int, gamma: float = 0.05, confidence: float = 0.95):
    """
    Calculate recommended sample size.
    
    Parameters:
    -----------
    d : int
        Number of hyperparameters
    gamma : float
        Assumed fraction of "good" configurations
    confidence : float
        Desired probability of finding good configuration
    
    Returns:
    --------
    int : Recommended number of iterations
    """
    import math
    n = math.ceil(math.log(1 - confidence) / math.log(1 - gamma))
    # Add buffer for higher-dimensional spaces
    n = int(n * (1 + 0.1 * d))
    return n

# Example calculations
for d in [3, 5, 10]:
    n = recommended_n_iter(d, gamma=0.05, confidence=0.95)
    print(f"Parameters: {d}, Recommended n_iter: {n}")
```

### Distribution Selection Guidelines

| Parameter Type | Recommended Distribution | Example |
|----------------|-------------------------|---------|
| Learning rate | Log-uniform | `loguniform(1e-5, 1e-1)` |
| Regularization | Log-uniform | `loguniform(1e-6, 1e0)` |
| Number of units | Discrete | `randint(32, 512)` |
| Dropout rate | Uniform | `uniform(0.0, 0.5)` |
| Kernel size | Choice | `[3, 5, 7]` |
| Activation | Choice | `['relu', 'tanh', 'elu']` |

## Summary

Random Search is a powerful, efficient alternative to Grid Search for hyperparameter optimization. Its key advantages include linear computational scaling, native support for continuous distributions, and theoretical guarantees for finding good configurations. Best practices include using log-uniform distributions for scale-sensitive parameters, choosing appropriate sample sizes based on confidence requirements, and analyzing results to understand parameter importance. For complex optimization problems, Random Search often serves as an excellent baseline before considering more sophisticated methods like Bayesian Optimization.
