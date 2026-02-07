# Feature Interaction Effects

## Introduction

Most interpretability methods compute **main effects**—the independent contribution of each feature to the prediction. However, many real-world phenomena involve **interactions**: the combined effect of two or more features that cannot be decomposed into individual contributions. Understanding interaction effects is critical in quantitative finance where factor interactions drive portfolio dynamics, risk concentrations, and non-linear market behavior.

## Mathematical Foundation

### Defining Interactions

For a model $f(\mathbf{x})$, the prediction can be decomposed into main effects and interactions:

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_i(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x}) + \text{higher-order terms}
$$

The **SHAP interaction value** $\Phi_{ij}$ for features $i$ and $j$ is:

$$
\Phi_{ij}(\mathbf{x}) = \sum_{S \subseteq N \setminus \{i,j\}} \frac{|S|!(d - |S| - 2)!}{2(d-1)!} \nabla_{ij}(S)
$$

where the discrete second derivative is:

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

### Properties

1. **Symmetry**: $\Phi_{ij} = \Phi_{ji}$
2. **Completeness**: $\sum_j \Phi_{ij} = \phi_i$
3. **Diagonal = main effect**: $\Phi_{ii}$ captures the main effect after removing interactions

### Friedman's H-Statistic

Measures the fraction of variance explained by an interaction:

$$
H^2_{ij} = \frac{\sum_k \left[\hat{f}_{ij}(x_i^{(k)}, x_j^{(k)}) - \hat{f}_i(x_i^{(k)}) - \hat{f}_j(x_j^{(k)})\right]^2}{\sum_k \hat{f}_{ij}^2(x_i^{(k)}, x_j^{(k)})}
$$

## Computing Interaction Effects

### SHAP Interaction Values

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_shap_interactions(model, X, feature_names, n_samples=100):
    """
    Compute SHAP interaction values for tree models.
    
    Returns interaction_values with shape [n_samples, n_features, n_features].
    """
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X[:n_samples])
    return interaction_values


def top_interactions(interaction_values, feature_names, k=10):
    """Find top-k feature interactions by average magnitude."""
    mean_interactions = np.abs(interaction_values).mean(axis=0)
    n_features = len(feature_names)
    
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions.append({
                'feature_i': feature_names[i],
                'feature_j': feature_names[j],
                'strength': mean_interactions[i, j]
            })
    
    interactions.sort(key=lambda x: x['strength'], reverse=True)
    return interactions[:k]
```

### Visualization

```python
def plot_interaction_matrix(interaction_values, feature_names):
    """Plot average interaction effects as heatmap."""
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


def plot_interaction_dependence(
    interaction_values, X, feature_names, feature_i, feature_j
):
    """
    Plot how the interaction between two features varies
    with their values.
    """
    i = feature_names.index(feature_i)
    j = feature_names.index(feature_j)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Interaction value vs feature_i, colored by feature_j
    scatter = axes[0].scatter(
        X[:len(interaction_values), i],
        interaction_values[:, i, j],
        c=X[:len(interaction_values), j],
        cmap='coolwarm', alpha=0.6, s=20
    )
    axes[0].set_xlabel(feature_i)
    axes[0].set_ylabel(f'Interaction ({feature_i} × {feature_j})')
    plt.colorbar(scatter, ax=axes[0], label=feature_j)
    
    # Interaction value vs feature_j, colored by feature_i
    scatter = axes[1].scatter(
        X[:len(interaction_values), j],
        interaction_values[:, i, j],
        c=X[:len(interaction_values), i],
        cmap='coolwarm', alpha=0.6, s=20
    )
    axes[1].set_xlabel(feature_j)
    axes[1].set_ylabel(f'Interaction ({feature_i} × {feature_j})')
    plt.colorbar(scatter, ax=axes[1], label=feature_i)
    
    plt.tight_layout()
    return fig
```

## Applications in Quantitative Finance

### Factor Interaction Analysis

```python
def analyze_factor_interactions(
    model,
    factor_data: np.ndarray,
    factor_names: list
):
    """
    Analyze interactions between financial factors.
    
    Important interactions in finance:
    - Momentum × Volatility (momentum crashes in volatile markets)
    - Value × Quality (value traps vs quality-at-reasonable-price)  
    - Size × Liquidity (small-cap illiquidity premium)
    """
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(factor_data[:200])
    
    # Main effects vs interactions
    n_factors = len(factor_names)
    main_effects = np.zeros(n_factors)
    interaction_effects = np.zeros((n_factors, n_factors))
    
    for i in range(n_factors):
        main_effects[i] = np.abs(interactions[:, i, i]).mean()
        for j in range(n_factors):
            if i != j:
                interaction_effects[i, j] = np.abs(interactions[:, i, j]).mean()
    
    # Ratio of interaction to main effect
    print("Factor Main Effects vs Interactions:")
    print("-" * 60)
    for i in range(n_factors):
        total_interaction = interaction_effects[i].sum()
        ratio = total_interaction / (main_effects[i] + 1e-10)
        print(f"{factor_names[i]:20s}: main={main_effects[i]:.4f}, "
              f"interaction={total_interaction:.4f}, ratio={ratio:.2f}")
    
    return interactions
```

### Risk Concentration Detection

```python
def detect_risk_concentrations(
    risk_model,
    portfolio_features: np.ndarray,
    feature_names: list
):
    """
    Use interaction effects to detect hidden risk concentrations.
    
    Large interactions between risk factors indicate that 
    diversification benefits may be overstated.
    """
    interactions = compute_shap_interactions(
        risk_model, portfolio_features, feature_names
    )
    
    top = top_interactions(interactions, feature_names, k=5)
    
    print("Potential Risk Concentrations (Top Interactions):")
    print("-" * 60)
    for inter in top:
        print(f"  {inter['feature_i']} × {inter['feature_j']}: "
              f"strength={inter['strength']:.4f}")
    
    return top
```

## Summary

Feature interaction analysis reveals the non-additive structure in model predictions. In quantitative finance, interactions capture regime-dependent behavior, factor crowding, and non-linear risk dynamics that main effects alone cannot explain.

**Key equation:**

$$
\nabla_{ij}(S) = f_x(S \cup \{i,j\}) - f_x(S \cup \{i\}) - f_x(S \cup \{j\}) + f_x(S)
$$

## References

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Friedman, J. H., & Popescu, B. E. (2008). "Predictive Learning via Rule Ensembles." *Annals of Applied Statistics*.

3. Tsang, M., et al. (2020). "How Does This Interaction Affect Me? Interpretable Estimation of Individual-Level Interaction Effects." *AAAI*.
