# Bayesian Optimization

## Overview

Bayesian Optimization is a sequential model-based optimization technique that intelligently explores the hyperparameter space by building a probabilistic surrogate model of the objective function. Unlike Grid Search or Random Search, Bayesian Optimization learns from previous evaluations to focus sampling on promising regions, making it highly sample-efficient for expensive objective functions.

**Core Principle**: Build a probabilistic model of $f(\boldsymbol{\theta})$ (the objective function), use it to decide where to sample next, then update the model with the new observation. This sequential process balances exploration (sampling uncertain regions) and exploitation (sampling near known good values).

**Key Characteristics:**

- **Sample Efficient**: Requires fewer evaluations than random or grid search
- **Sequential**: Each evaluation informs subsequent sampling decisions
- **Probabilistic**: Quantifies uncertainty in the objective function
- **Adaptive**: Automatically balances exploration and exploitation

## Mathematical Framework

### Problem Formulation

We seek to maximize an expensive black-box objective function $f: \mathcal{X} \rightarrow \mathbb{R}$:

$$\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta} \in \mathcal{X}}{\arg\max} \; f(\boldsymbol{\theta})$$

where evaluating $f(\boldsymbol{\theta})$ is computationally expensive (e.g., cross-validation score of a neural network).

### Surrogate Model

We model $f$ using a probabilistic surrogate $p(f | \mathcal{D}_n)$, where $\mathcal{D}_n = \{(\boldsymbol{\theta}_i, y_i)\}_{i=1}^n$ are previous observations with $y_i = f(\boldsymbol{\theta}_i) + \epsilon$ and $\epsilon$ is observation noise.

**Gaussian Process (GP)**: The most common surrogate model, a GP defines a distribution over functions:

$$f(\boldsymbol{\theta}) \sim \mathcal{GP}(m(\boldsymbol{\theta}), k(\boldsymbol{\theta}, \boldsymbol{\theta}'))$$

where $m(\boldsymbol{\theta})$ is the mean function (often zero) and $k(\boldsymbol{\theta}, \boldsymbol{\theta}')$ is the covariance (kernel) function.

**GP Posterior**: Given observations $\mathcal{D}_n$, the posterior predictive distribution at a new point $\boldsymbol{\theta}^*$ is:

$$p(f^* | \boldsymbol{\theta}^*, \mathcal{D}_n) = \mathcal{N}(\mu_n(\boldsymbol{\theta}^*), \sigma_n^2(\boldsymbol{\theta}^*))$$

with:
$$\mu_n(\boldsymbol{\theta}^*) = \mathbf{k}^T (\mathbf{K} + \sigma_\epsilon^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma_n^2(\boldsymbol{\theta}^*) = k(\boldsymbol{\theta}^*, \boldsymbol{\theta}^*) - \mathbf{k}^T (\mathbf{K} + \sigma_\epsilon^2 \mathbf{I})^{-1} \mathbf{k}$$

where $\mathbf{k} = [k(\boldsymbol{\theta}_1, \boldsymbol{\theta}^*), \ldots, k(\boldsymbol{\theta}_n, \boldsymbol{\theta}^*)]^T$ and $\mathbf{K}_{ij} = k(\boldsymbol{\theta}_i, \boldsymbol{\theta}_j)$.

### Acquisition Functions

An acquisition function $\alpha(\boldsymbol{\theta})$ quantifies the utility of evaluating $f$ at $\boldsymbol{\theta}$. The next evaluation point is:

$$\boldsymbol{\theta}_{n+1} = \underset{\boldsymbol{\theta} \in \mathcal{X}}{\arg\max} \; \alpha(\boldsymbol{\theta} | \mathcal{D}_n)$$

**Expected Improvement (EI)**: The expected improvement over the current best $f^+ = \max_{i \leq n} y_i$:

$$\text{EI}(\boldsymbol{\theta}) = \mathbb{E}\left[\max(f(\boldsymbol{\theta}) - f^+, 0)\right]$$

Closed-form solution:
$$\text{EI}(\boldsymbol{\theta}) = (\mu_n(\boldsymbol{\theta}) - f^+ - \xi) \Phi(Z) + \sigma_n(\boldsymbol{\theta}) \phi(Z)$$

where $Z = \frac{\mu_n(\boldsymbol{\theta}) - f^+ - \xi}{\sigma_n(\boldsymbol{\theta})}$, $\Phi$ is the standard normal CDF, $\phi$ is the PDF, and $\xi$ is an exploration parameter.

**Upper Confidence Bound (UCB)**: Optimistic estimate balancing mean and uncertainty:

$$\text{UCB}(\boldsymbol{\theta}) = \mu_n(\boldsymbol{\theta}) + \kappa \sigma_n(\boldsymbol{\theta})$$

where $\kappa > 0$ controls exploration-exploitation trade-off.

**Probability of Improvement (PI)**: Probability of exceeding current best:

$$\text{PI}(\boldsymbol{\theta}) = P(f(\boldsymbol{\theta}) > f^+ + \xi) = \Phi(Z)$$

## Algorithm

```
Algorithm: Bayesian Optimization
────────────────────────────────
Input: Objective f, Search space X, Acquisition function α,
       Initial samples n₀, Total budget N
Output: Best parameters θ*, Best value f*

1. Initialize: Sample n₀ points uniformly from X
2. Evaluate: y_i ← f(θ_i) for i = 1, ..., n₀
3. D_n ← {(θ_i, y_i)}_{i=1}^{n₀}

4. For n = n₀ to N-1:
   a. Fit surrogate model p(f | D_n)
   b. Find next point: θ_{n+1} ← argmax_θ α(θ | D_n)
   c. Evaluate: y_{n+1} ← f(θ_{n+1})
   d. Augment data: D_{n+1} ← D_n ∪ {(θ_{n+1}, y_{n+1})}

5. Return: θ* = argmax_i y_i, f* = max_i y_i
```

## Implementation with Optuna

Optuna is a modern hyperparameter optimization framework that implements efficient Bayesian optimization algorithms including TPE (Tree-structured Parzen Estimator).

### Basic Optuna Example

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    """
    Optuna objective function.
    
    The trial object suggests parameter values and Optuna
    optimizes based on the returned score.
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', None]
        ),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    
    # Create and evaluate model
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    return scores.mean()

# Create study (optimization session)
study = optuna.create_study(
    direction='maximize',  # Maximize accuracy
    sampler=optuna.samplers.TPESampler(seed=42)  # TPE sampler
)

# Optimize
study.optimize(
    objective, 
    n_trials=100,  # Number of trials
    show_progress_bar=True
)

# Results
print(f"\nBest Parameters: {study.best_params}")
print(f"Best CV Score: {study.best_value:.4f}")

# Train final model
best_model = RandomForestClassifier(
    **study.best_params, random_state=42, n_jobs=-1
)
best_model.fit(X_train, y_train)
print(f"Test Score: {best_model.score(X_test, y_test):.4f}")
```

### Log-Scale and Conditional Parameters

```python
from sklearn.ensemble import GradientBoostingClassifier

def objective_gb(trial):
    """
    Objective with log-scale and conditional parameters.
    """
    # Log-scale for learning rate (important!)
    learning_rate = trial.suggest_float(
        'learning_rate', 1e-4, 1e-1, log=True
    )
    
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    # Conditional parameter: subsample only if using bootstrap
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    params = {
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'subsample': subsample,
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    
    return scores.mean()

# Study with TPE sampler
study_gb = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study_gb.optimize(objective_gb, n_trials=50, show_progress_bar=True)

print(f"\nGradient Boosting Best:")
print(f"  learning_rate: {study_gb.best_params['learning_rate']:.6f}")
print(f"  Best Score: {study_gb.best_value:.4f}")
```

### Pruning for Early Stopping

Pruning terminates unpromising trials early, saving computation:

```python
def objective_with_pruning(trial):
    """
    Objective function with intermediate value reporting for pruning.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Report intermediate values for each CV fold
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = []
    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        scores.append(score)
        
        # Report intermediate value
        trial.report(score, step)
        
        # Prune if performing poorly
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Create study with pruner
study_pruned = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Minimum trials before pruning
        n_warmup_steps=2,      # Steps before pruning within a trial
        interval_steps=1
    )
)

study_pruned.optimize(objective_with_pruning, n_trials=100, show_progress_bar=True)

# Pruning statistics
pruned_trials = [t for t in study_pruned.trials 
                 if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study_pruned.trials 
                   if t.state == optuna.trial.TrialState.COMPLETE]

print(f"\nPruning Statistics:")
print(f"  Completed trials: {len(complete_trials)}")
print(f"  Pruned trials: {len(pruned_trials)}")
print(f"  Pruning rate: {len(pruned_trials)/len(study_pruned.trials)*100:.1f}%")
```

### Visualization

```python
# Optuna provides built-in visualizations
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Parameter importances
fig = vis.plot_param_importances(study)
fig.show()

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Slice plot for individual parameters
fig = vis.plot_slice(study)
fig.show()

# Contour plot for parameter interactions
fig = vis.plot_contour(study, params=['n_estimators', 'max_depth'])
fig.show()
```

## Comparing Different Samplers

Optuna supports multiple sampling strategies:

```python
def compare_samplers(objective, n_trials=50):
    """Compare different Optuna samplers."""
    
    samplers = {
        'TPE': optuna.samplers.TPESampler(seed=42),
        'Random': optuna.samplers.RandomSampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
        'GP': optuna.samplers.GPSampler(seed=42),  # Gaussian Process
    }
    
    results = {}
    
    for name, sampler in samplers.items():
        print(f"\nTesting {name} sampler...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        import time
        start = time.time()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        elapsed = time.time() - start
        
        results[name] = {
            'best_value': study.best_value,
            'time': elapsed,
            'study': study
        }
        
        print(f"  Best: {study.best_value:.4f}, Time: {elapsed:.2f}s")
    
    return results

# Run comparison
sampler_results = compare_samplers(objective, n_trials=50)

# Plot comparison
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

names = list(sampler_results.keys())
scores = [r['best_value'] for r in sampler_results.values()]
times = [r['time'] for r in sampler_results.values()]

ax1.bar(names, scores)
ax1.set_ylabel('Best Score')
ax1.set_title('Sampler Comparison: Score')

ax2.bar(names, times)
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Sampler Comparison: Time')

plt.tight_layout()
plt.show()
```

## Tree-structured Parzen Estimator (TPE)

TPE is Optuna's default sampler and an efficient alternative to GP-based Bayesian optimization.

### TPE Algorithm

Instead of modeling $p(y | \boldsymbol{\theta})$ directly, TPE models:

$$p(\boldsymbol{\theta} | y) = \begin{cases} 
\ell(\boldsymbol{\theta}) & \text{if } y < y^* \\
g(\boldsymbol{\theta}) & \text{if } y \geq y^*
\end{cases}$$

where $y^*$ is a quantile of observed values (typically 15th percentile), $\ell(\boldsymbol{\theta})$ is a density fitted to "good" observations, and $g(\boldsymbol{\theta})$ is fitted to remaining observations.

The acquisition function maximizes:

$$\frac{\ell(\boldsymbol{\theta})}{g(\boldsymbol{\theta})} \propto p(y < y^* | \boldsymbol{\theta})$$

### Advantages of TPE

| Advantage | Description |
|-----------|-------------|
| **Scalability** | Handles high-dimensional spaces better than GP |
| **Categorical** | Naturally handles categorical parameters |
| **Parallel** | Supports parallel/asynchronous optimization |
| **Tree-structured** | Efficient for conditional parameters |

## Neural Network Hyperparameter Optimization

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def create_pytorch_objective(X_train, y_train, X_val, y_val):
    """
    Create objective function for neural network hyperparameter tuning.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    
    def objective(trial):
        # Architectural hyperparameters
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = [
            trial.suggest_int(f'hidden_size_{i}', 16, 256, log=True)
            for i in range(n_layers)
        ]
        dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        optimizer_name = trial.suggest_categorical(
            'optimizer', ['Adam', 'AdamW', 'SGD']
        )
        
        # Build model
        layers = []
        input_size = X_train.shape[1]
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, len(np.unique(y_train))))
        model = nn.Sequential(*layers).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, momentum=0.9
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop with pruning
        n_epochs = 50
        for epoch in range(n_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation accuracy
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_accuracy = correct / total
            
            # Report for pruning
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return val_accuracy
    
    return objective

# Create validation split
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Run optimization
nn_objective = create_pytorch_objective(X_tr, y_tr, X_val, y_val)

study_nn = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)

study_nn.optimize(nn_objective, n_trials=100, show_progress_bar=True)

print(f"\nNeural Network Best Parameters:")
for key, value in study_nn.best_params.items():
    print(f"  {key}: {value}")
print(f"\nBest Validation Accuracy: {study_nn.best_value:.4f}")
```

## Multi-Objective Optimization

Optuna supports Pareto optimization for multiple objectives:

```python
def multi_objective(trial):
    """
    Optimize for both accuracy and training time.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    import time
    start = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=3)
    training_time = time.time() - start
    
    accuracy = scores.mean()
    
    # Return multiple objectives
    return accuracy, -training_time  # Maximize accuracy, minimize time

# Multi-objective study
study_mo = optuna.create_study(
    directions=['maximize', 'maximize']  # Both maximize (time is negated)
)
study_mo.optimize(multi_objective, n_trials=50, show_progress_bar=True)

# Get Pareto front
pareto_front = study_mo.best_trials
print(f"\nPareto Front ({len(pareto_front)} solutions):")
for trial in pareto_front:
    print(f"  Accuracy: {trial.values[0]:.4f}, Time: {-trial.values[1]:.3f}s")

# Visualize Pareto front
fig = vis.plot_pareto_front(
    study_mo, 
    target_names=['Accuracy', 'Training Time (s)']
)
fig.show()
```

## Advantages and Limitations

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Sample Efficiency** | Requires fewer evaluations than random/grid search |
| **Intelligent Search** | Learns from previous evaluations |
| **Uncertainty Quantification** | Provides confidence estimates |
| **Handles Noise** | Robust to noisy objective functions |
| **Flexible** | Works with discrete, continuous, and categorical parameters |

### Limitations

| Limitation | Description |
|------------|-------------|
| **Overhead** | Model fitting adds per-iteration cost |
| **Scalability** | GP-based methods struggle in high dimensions (>20) |
| **Sequential** | Limited parallelization compared to random search |
| **Local Optima** | May get stuck if acquisition function is poorly tuned |
| **Complexity** | More complex to implement and tune than simpler methods |

## When to Use Bayesian Optimization

**Good Use Cases:**

- Expensive objective evaluations (neural network training)
- Medium-dimensional parameter spaces (5-20 parameters)
- Limited computational budget
- Need for sample efficiency
- Continuous optimization with smooth objective

**Avoid When:**

- Very cheap objective evaluations
- Very high-dimensional spaces (>50 parameters)
- Highly non-smooth or discontinuous objectives
- Need for massive parallelization
- Simple problems where random search suffices

## Summary

Bayesian Optimization provides a principled approach to hyperparameter tuning by building probabilistic models of the objective function and using acquisition functions to balance exploration and exploitation. Key frameworks like Optuna make implementation straightforward, offering features like pruning, multi-objective optimization, and various samplers including the efficient TPE algorithm. While more complex than random or grid search, Bayesian Optimization is the method of choice when function evaluations are expensive and sample efficiency is paramount.
