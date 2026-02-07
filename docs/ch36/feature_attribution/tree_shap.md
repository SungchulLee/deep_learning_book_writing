# Tree SHAP

## Introduction

**Tree SHAP** is an exact, polynomial-time algorithm for computing Shapley values on tree-based models (decision trees, random forests, gradient boosted trees). While Kernel SHAP requires $O(TL2^M)$ evaluations for exact computation, Tree SHAP exploits the recursive structure of decision trees to compute exact Shapley values in $O(TLD^2)$ time, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum depth.

This efficiency makes Tree SHAP the method of choice for explaining XGBoost, LightGBM, CatBoost, and scikit-learn tree models.

## Theoretical Foundation

### Shapley Values for Trees

For a tree ensemble with prediction function $f(\mathbf{x}) = \sum_{t=1}^{T} f_t(\mathbf{x})$, the SHAP value for feature $i$ decomposes across trees:

$$
\phi_i(\mathbf{x}) = \sum_{t=1}^{T} \phi_i^{(t)}(\mathbf{x})
$$

For a single decision tree, the coalition value $f_x(S)$ (prediction when only features in $S$ are known) can be computed by following the tree structure:

- At each internal node splitting on feature $j$:
  - If $j \in S$: follow the appropriate branch based on $x_j$
  - If $j \notin S$: follow **both** branches, weighted by the proportion of training samples in each

### The Tree SHAP Algorithm

The key algorithmic insight is tracking the set of features used along each path from root to leaf, and efficiently computing the weighted contribution of each feature.

For each path $p$ from root to leaf with value $v_p$:

1. Let $D_p = \{j_1, j_2, \ldots, j_k\}$ be the set of features used in decisions along path $p$
2. For each feature $j_i \in D_p$, compute its marginal contribution considering all orderings
3. The contribution accounts for the fraction of training data at each node

### Complexity

| Method | Complexity | Exact? |
|--------|-----------|--------|
| Brute-force Shapley | $O(2^d \cdot T \cdot D)$ | Yes |
| Kernel SHAP | $O(N_{\text{samples}} \cdot N_{\text{background}})$ | No |
| **Tree SHAP** | $O(T \cdot L \cdot D^2)$ | **Yes** |

For a typical XGBoost model with 500 trees, depth 6, and 64 leaves: Tree SHAP computes exact values in milliseconds per sample.

## Implementation

### Using the SHAP Library

The recommended approach is the optimized C++ implementation in the `shap` library:

```python
import shap
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification

def tree_shap_example():
    """Complete Tree SHAP example with XGBoost."""
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20,
        n_informative=10, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, random_state=42
    )
    model.fit(X[:800], y[:800])
    
    # Tree SHAP explainer - exact and fast
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for test set
    shap_values = explainer.shap_values(X[800:])
    
    # Base value (expected prediction)
    print(f"Base value: {explainer.expected_value:.4f}")
    
    # Verify completeness for first sample
    prediction = model.predict_proba(X[800:801])[0, 1]
    from scipy.special import expit
    shap_sum = shap_values[0].sum() + explainer.expected_value
    print(f"Prediction (prob):  {prediction:.4f}")
    print(f"SHAP reconstruction: {expit(shap_sum):.4f}")
    
    return shap_values, explainer


def tree_shap_for_lightgbm(model, X_train, X_test, feature_names):
    """Tree SHAP for LightGBM models."""
    import lightgbm as lgb
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For multi-class, shap_values is a list per class
    if isinstance(shap_values, list):
        print(f"Number of classes: {len(shap_values)}")
        for c, sv in enumerate(shap_values):
            importance = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(importance)[::-1][:5]
            print(f"\nClass {c} top features:")
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {importance[idx]:.4f}")
    
    return shap_values
```

### SHAP Interaction Values

A unique capability of Tree SHAP is computing **exact interaction values**—second-order Shapley values that decompose the prediction into main effects and pairwise interactions:

$$
f(\mathbf{x}) = \phi_0 + \sum_i \phi_{ii}(\mathbf{x}) + \sum_{i < j} \phi_{ij}(\mathbf{x})
$$

where $\phi_{ii}$ is the main effect of feature $i$ and $\phi_{ij}$ captures the interaction between features $i$ and $j$.

```python
def compute_tree_shap_interactions(model, X_test, feature_names):
    """
    Compute SHAP interaction values for tree models.
    
    interaction_values[i, j, k] = interaction between features j and k
    for sample i.
    """
    explainer = shap.TreeExplainer(model)
    
    # Interaction values - more expensive than main effects
    interaction_values = explainer.shap_interaction_values(X_test[:100])
    
    # Average absolute interactions
    mean_interactions = np.abs(interaction_values).mean(axis=0)
    
    # Top interactions
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
    
    print("Top Feature Interactions:")
    print("-" * 50)
    for inter in interactions[:10]:
        print(f"{inter['feature_i']:20s} × {inter['feature_j']:20s}: "
              f"{inter['strength']:.4f}")
    
    return interaction_values
```

## Visualization

### Summary Plot

```python
def tree_shap_visualization(shap_values, X_test, feature_names):
    """Standard SHAP visualizations for tree models."""
    
    # Beeswarm summary - shows distribution of SHAP values per feature
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='dot'
    )
    
    # Bar plot - mean absolute SHAP values
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='bar'
    )
    
    # Dependence plot - SHAP value vs feature value
    # Automatically detects interaction feature
    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    shap.dependence_plot(
        top_feature_idx, shap_values, X_test,
        feature_names=feature_names
    )
```

### Waterfall Plot

```python
def explain_single_prediction(explainer, X_instance, feature_names):
    """Detailed waterfall explanation for a single prediction."""
    
    shap_values = explainer.shap_values(X_instance.reshape(1, -1))
    
    # Waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_instance,
            feature_names=feature_names
        )
    )
```

## Interventional vs Observational Tree SHAP

Tree SHAP supports two modes:

### Observational (Path-Dependent)

Uses the conditional distribution $P(X_{\bar{S}} \mid X_S)$ implied by the tree structure. Features that are correlated receive shared credit.

```python
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
```

### Interventional

Uses the marginal distribution $P(X_{\bar{S}})$ regardless of observed features. Better reflects causal contributions.

```python
explainer = shap.TreeExplainer(model, X_background, feature_perturbation='interventional')
```

| Mode | Pros | Cons |
|------|------|------|
| Observational | Fast, no background needed | Credits correlated features |
| Interventional | Causal interpretation | Requires background data, slower |

## Applications in Quantitative Finance

### Credit Scoring with Tree Models

```python
def explain_credit_tree_model(
    model,  # XGBoost/LightGBM credit scoring model
    applicant_features: np.ndarray,
    feature_names: list,
    X_train: np.ndarray
):
    """
    Generate regulatory-compliant explanations for credit decisions.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(applicant_features.reshape(1, -1))
    
    # If binary classification, shap_values may be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Default class
    
    values = shap_values[0]
    
    # Adverse action reasons (regulatory requirement)
    negative_factors = []
    for idx in np.argsort(values)[::-1]:
        if values[idx] > 0:  # Increases default risk
            negative_factors.append({
                'factor': feature_names[idx],
                'impact': values[idx],
                'value': applicant_features[idx]
            })
    
    print("Adverse Action Reasons:")
    for i, factor in enumerate(negative_factors[:4], 1):
        print(f"  {i}. {factor['factor']}: value={factor['value']:.2f}, "
              f"impact={factor['impact']:+.4f}")
    
    return shap_values
```

## Summary

Tree SHAP provides exact Shapley values for tree-based models in polynomial time, making it the gold standard for explaining gradient boosted models in production.

**Key properties:**

- **Exact**: No approximation error (unlike Kernel SHAP)
- **Fast**: $O(TLD^2)$ per sample
- **Interactions**: Can compute exact pairwise interaction values
- **Two modes**: Observational (fast) and interventional (causal)

## References

1. Lundberg, S. M., et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees." *Nature Machine Intelligence*.

2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

3. Lundberg, S. M., et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles." *arXiv:1802.03888*.
