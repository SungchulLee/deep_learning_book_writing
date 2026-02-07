# SHAP: SHapley Additive exPlanations

## Introduction

SHAP (SHapley Additive exPlanations) is a unified approach to feature attribution based on Shapley values from cooperative game theory. SHAP provides a theoretically grounded method for explaining individual predictions while satisfying several desirable properties that other methods lack.

The key insight is that feature attribution can be framed as a fair division problem: **How do we fairly distribute the prediction among the features that contributed to it?**

## Theoretical Foundation

### Shapley Values from Game Theory

In cooperative game theory, Shapley values answer: "How should we fairly divide a collective payoff among players based on their contributions?"

For a game with $N$ players and a value function $v(S)$ for any coalition $S \subseteq N$, the Shapley value for player $i$ is:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[v(S \cup \{i\}) - v(S)\right]$$

This averages the marginal contribution of player $i$ over all possible orderings of players.

### SHAP for Machine Learning

For a model $f$ with input features $x_1, ..., x_d$:
- **Players** = features
- **Payoff** = model prediction $f(x)$
- **Coalition value** $v(S)$ = expected prediction when only features in $S$ are known

The SHAP value for feature $i$ is:

$$\phi_i(x) = \sum_{S \subseteq \{1,...,d\} \setminus \{i\}} \frac{|S|!(d-|S|-1)!}{d!} \left[f_x(S \cup \{i\}) - f_x(S)\right]$$

where $f_x(S)$ is the expected prediction when only features in $S$ have their observed values.

## Properties of SHAP

SHAP uniquely satisfies three desirable properties:

### 1. Local Accuracy (Efficiency)

The feature attributions sum to the difference between the prediction and the expected prediction:

$$f(x) = \phi_0 + \sum_{i=1}^{d} \phi_i(x)$$

where $\phi_0 = \mathbb{E}[f(X)]$ is the base value.

### 2. Missingness

Features that don't change the output get zero attribution:

$$x_i = x'_i \Rightarrow \phi_i(x) = 0$$

### 3. Consistency

If a feature's contribution increases in a new model, its SHAP value should not decrease:

$$f'_x(S \cup \{i\}) - f'_x(S) \geq f_x(S \cup \{i\}) - f_x(S)$$

for all $S$ implies $\phi'_i(x) \geq \phi_i(x)$.

## SHAP Algorithms

### Kernel SHAP

Model-agnostic approximation using weighted linear regression:

```python
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression

class KernelSHAP:
    """
    Kernel SHAP - model-agnostic SHAP approximation.
    """
    
    def __init__(self, model, background_data):
        """
        Args:
            model: Prediction function
            background_data: Reference data for computing expectations
        """
        self.model = model
        self.background_data = background_data
        self.base_value = model(background_data).mean()
    
    def _shap_kernel(self, n_features, coalition_size):
        """SHAP kernel weight for coalition of given size."""
        if coalition_size == 0 or coalition_size == n_features:
            return float('inf')  # Will be handled separately
        
        from math import comb
        return (n_features - 1) / (comb(n_features, coalition_size) * 
                                    coalition_size * (n_features - coalition_size))
    
    def _compute_coalition_value(self, instance, coalition, background):
        """
        Compute expected model output for a coalition.
        
        Features in coalition take values from instance;
        features not in coalition are marginalized over background.
        """
        n_samples = len(background)
        samples = np.tile(instance, (n_samples, 1))
        
        # Replace non-coalition features with background values
        mask = np.ones(len(instance), dtype=bool)
        mask[list(coalition)] = False
        samples[:, mask] = background[:, mask]
        
        return self.model(samples).mean()
    
    def explain(
        self,
        instance: np.ndarray,
        num_samples: int = 2048
    ) -> np.ndarray:
        """
        Compute SHAP values for a single instance.
        
        Args:
            instance: Input to explain
            num_samples: Number of coalition samples
            
        Returns:
            SHAP values for each feature
        """
        n_features = len(instance)
        
        # Sample coalitions and compute values
        coalitions = []
        coalition_values = []
        weights = []
        
        # Sample random coalitions
        for _ in range(num_samples):
            # Random coalition size
            size = np.random.randint(0, n_features + 1)
            coalition = tuple(sorted(
                np.random.choice(n_features, size, replace=False)
            ))
            
            if coalition not in coalitions:
                coalitions.append(coalition)
                value = self._compute_coalition_value(
                    instance, coalition, self.background_data
                )
                coalition_values.append(value)
                weights.append(self._shap_kernel(n_features, len(coalition)))
        
        # Create binary coalition matrix
        Z = np.zeros((len(coalitions), n_features))
        for i, coalition in enumerate(coalitions):
            Z[i, list(coalition)] = 1
        
        # Weighted linear regression
        W = np.diag(np.clip(weights, 0, 1e10))
        
        # Solve: minimize ||sqrt(W)(Zφ - y)||^2
        # Solution: φ = (Z'WZ)^{-1} Z'Wy
        y = np.array(coalition_values) - self.base_value
        
        ZtWZ = Z.T @ W @ Z + 1e-6 * np.eye(n_features)
        ZtWy = Z.T @ W @ y
        
        shap_values = np.linalg.solve(ZtWZ, ZtWy)
        
        return shap_values
```

### Tree SHAP

Exact and fast algorithm for tree-based models:

```python
class TreeSHAP:
    """
    TreeSHAP for tree ensemble models (simplified).
    
    The actual algorithm recursively computes Shapley values
    by leveraging the tree structure.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Tree ensemble (e.g., XGBoost, LightGBM)
        """
        self.model = model
        
    def explain(self, instance):
        """
        Compute SHAP values using tree structure.
        
        For production use, use the shap library which has
        optimized C++ implementation.
        """
        # Placeholder - actual TreeSHAP is complex
        # Use: shap.TreeExplainer(model).shap_values(instance)
        raise NotImplementedError("Use shap.TreeExplainer")
```

### Deep SHAP

SHAP for neural networks using DeepLIFT approximation:

```python
import torch
import torch.nn as nn

class DeepSHAP:
    """
    Deep SHAP using DeepLIFT-style backpropagation.
    """
    
    def __init__(self, model: nn.Module, background: torch.Tensor):
        """
        Args:
            model: PyTorch neural network
            background: Background samples for computing expectations
        """
        self.model = model
        self.background = background
        
        with torch.no_grad():
            self.base_output = model(background).mean(dim=0)
    
    def _deep_lift_gradient(self, x, baseline, target_class):
        """Compute DeepLIFT-style attributions."""
        x.requires_grad_(True)
        baseline.requires_grad_(True)
        
        output_x = self.model(x)
        output_baseline = self.model(baseline)
        
        diff = output_x[:, target_class] - output_baseline[:, target_class]
        
        # Compute gradients
        grads_x = torch.autograd.grad(
            diff.sum(), x, create_graph=False
        )[0]
        
        # Attribution = gradient * (input - baseline)
        attr = grads_x * (x - baseline)
        
        return attr
    
    def explain(
        self,
        instance: torch.Tensor,
        target_class: int = None,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Compute SHAP values for neural network.
        
        Args:
            instance: Input tensor
            target_class: Target class for explanation
            n_samples: Number of background samples to use
            
        Returns:
            SHAP values tensor
        """
        self.model.eval()
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(instance)
                target_class = output.argmax(dim=1).item()
        
        # Sample from background
        idx = torch.randperm(len(self.background))[:n_samples]
        baselines = self.background[idx]
        
        # Average attribution over baselines
        shap_values = torch.zeros_like(instance)
        
        for baseline in baselines:
            baseline = baseline.unsqueeze(0)
            attr = self._deep_lift_gradient(
                instance, baseline, target_class
            )
            shap_values += attr
        
        shap_values /= n_samples
        
        return shap_values
```

## Using the SHAP Library

```python
import shap
import torch
import numpy as np

def shap_for_tabular(model, X_train, X_test):
    """SHAP explanation for tabular data."""
    
    # Create explainer with background data
    explainer = shap.KernelExplainer(
        model.predict, 
        shap.sample(X_train, 100)
    )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test[:100])
    
    # Summary plot
    shap.summary_plot(shap_values, X_test[:100])
    
    # Force plot for single instance
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_test[0]
    )
    
    return shap_values


def shap_for_tree_model(model, X_train, X_test):
    """Fast SHAP for tree ensembles."""
    
    # TreeExplainer is much faster than KernelExplainer
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_test)
    
    # Interaction values (second-order)
    interaction_values = explainer.shap_interaction_values(X_test[:10])
    
    return shap_values, interaction_values


def shap_for_pytorch(model, background, test_samples):
    """SHAP for PyTorch models."""
    
    # Wrap PyTorch model
    def predict_fn(x):
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            return model(tensor).numpy()
    
    # Use DeepExplainer for neural networks
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    return shap_values
```

## Visualization

### Summary Plot

```python
def create_summary_plot(shap_values, features, feature_names):
    """Create SHAP summary plot showing feature importance."""
    
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Beeswarm-style plot
    for i, idx in enumerate(sorted_idx[::-1]):
        y = np.ones(len(shap_values)) * i + 0.1 * np.random.randn(len(shap_values))
        colors = features[:, idx]
        
        scatter = ax.scatter(
            shap_values[:, idx], y,
            c=colors, cmap='coolwarm',
            alpha=0.5, s=10
        )
    
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
    ax.set_xlabel('SHAP Value (impact on prediction)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.colorbar(scatter, label='Feature Value')
    plt.tight_layout()
    
    return fig
```

### Waterfall Plot

```python
def waterfall_plot(
    base_value: float,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list,
    max_display: int = 10
):
    """Create waterfall plot showing how features push prediction."""
    
    # Sort by absolute SHAP value
    order = np.argsort(np.abs(shap_values))[::-1][:max_display]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(max_display + 1)
    cumsum = base_value
    
    values = [base_value]
    labels = ['Base Value']
    
    for idx in order[::-1]:
        cumsum += shap_values[idx]
        values.append(shap_values[idx])
        labels.append(f'{feature_names[idx]} = {feature_values[idx]:.2f}')
    
    values.append(cumsum)
    labels.append(f'Prediction')
    
    colors = ['gray'] + ['red' if v < 0 else 'blue' for v in values[1:-1]] + ['gray']
    
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Model Output')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig
```

## Interaction Effects

SHAP can capture feature interactions:

```python
def compute_shap_interactions(model, X, n_samples=100):
    """
    Compute SHAP interaction values.
    
    Interaction values decompose prediction into main effects
    and pairwise interactions.
    """
    # For tree models, use TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # interaction_values[i, j, k] = interaction between features j and k
    # for sample i
    interaction_values = explainer.shap_interaction_values(X[:n_samples])
    
    return interaction_values


def plot_interaction_matrix(interaction_values, feature_names):
    """Plot average interaction effects as heatmap."""
    
    # Average absolute interactions
    mean_interactions = np.abs(interaction_values).mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        mean_interactions,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='Blues',
        ax=ax
    )
    
    ax.set_title('SHAP Feature Interactions')
    plt.tight_layout()
    
    return fig
```

## Applications in Finance

### Credit Risk Model Explanation

```python
def explain_credit_model(model, applicant, feature_names, background_data):
    """
    Comprehensive SHAP explanation for credit decision.
    """
    # Create explainer
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    
    # Get SHAP values (for probability of default)
    shap_values = explainer.shap_values(applicant.reshape(1, -1))
    
    # For binary classification, shap_values is list [class_0, class_1]
    default_shap = shap_values[1][0]  # SHAP values for default class
    
    # Generate explanation report
    print("=" * 60)
    print("CREDIT DECISION EXPLANATION")
    print("=" * 60)
    
    prediction = model.predict_proba(applicant.reshape(1, -1))[0, 1]
    print(f"\nPredicted Default Probability: {prediction:.1%}")
    print(f"Base Rate: {explainer.expected_value[1]:.1%}")
    
    print("\nTop Contributing Factors:")
    print("-" * 40)
    
    sorted_idx = np.argsort(np.abs(default_shap))[::-1]
    
    for idx in sorted_idx[:10]:
        feature = feature_names[idx]
        value = applicant[idx]
        shap_val = default_shap[idx]
        
        if shap_val > 0:
            direction = "increases default risk"
        else:
            direction = "decreases default risk"
        
        print(f"{feature:30s} = {value:8.2f} -> {shap_val:+.3f} ({direction})")
    
    return shap_values
```

### Portfolio Risk Attribution

```python
def portfolio_risk_attribution(
    risk_model,
    portfolio_weights,
    factor_exposures,
    factor_names
):
    """
    Use SHAP to attribute portfolio risk to factors.
    """
    # Risk model predicts portfolio variance/VaR
    def risk_fn(exposures):
        return risk_model.predict(exposures)
    
    # Background: typical factor exposures
    background = np.random.randn(100, len(factor_names))
    
    explainer = shap.KernelExplainer(risk_fn, background)
    
    shap_values = explainer.shap_values(
        factor_exposures.reshape(1, -1)
    )
    
    print("Portfolio Risk Attribution:")
    print("-" * 40)
    
    total_risk = risk_fn(factor_exposures.reshape(1, -1))[0]
    print(f"Total Portfolio Risk: {total_risk:.2%}")
    
    for i, (factor, shap_val) in enumerate(zip(factor_names, shap_values[0])):
        contribution = shap_val / total_risk * 100
        print(f"{factor:20s}: {contribution:+.1f}%")
    
    return shap_values
```

## Advantages and Limitations

### Advantages

| Property | Benefit |
|----------|---------|
| Theoretical foundation | Based on game-theoretic axioms |
| Local accuracy | Explanations are locally faithful |
| Consistency | Monotonic relationship with contribution |
| Unified approach | Works across model types |
| Interaction detection | Can decompose feature interactions |

### Limitations

| Limitation | Impact |
|------------|--------|
| Computational cost | Exact SHAP is exponential in features |
| Correlation handling | Shapley values can be unintuitive with correlated features |
| Background choice | Results depend on reference distribution |
| Causality | SHAP measures association, not causation |

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.

2. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." Nature Machine Intelligence.

3. Shapley, L. S. (1953). "A Value for n-Person Games." Contributions to the Theory of Games.

4. Molnar, C. (2020). "Interpretable Machine Learning." Chapter on SHAP.
