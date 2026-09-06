# Kernel SHAP
## Introduction

**Kernel SHAP** is a model-agnostic method for approximating SHAP values using weighted linear regression. It bridges the gap between LIME's local surrogate approach and SHAP's game-theoretic foundation by formulating Shapley value computation as a specially weighted regression problem. This makes it applicable to any black-box model while providing the theoretical guarantees of Shapley values.

## Theoretical Foundation

### From SHAP to Regression

Recall that SHAP values are defined as the unique additive feature attributions satisfying local accuracy, missingness, and consistency. Lundberg and Lee (2017) showed that these can be computed by solving a weighted least squares problem.

For a model $f$ and input $\mathbf{x}$ with $d$ features, define a simplified input $z' \in \{0, 1\}^d$ where $z'_i = 1$ means feature $i$ takes its observed value and $z'_i = 0$ means it is "absent" (marginalized out).

The SHAP values $\phi_1, \ldots, \phi_d$ minimize:

$$
\sum_{z' \in \{0,1\}^d} \pi_x(z') \left[ f_x(z') - \left(\phi_0 + \sum_{i=1}^{d} \phi_i z'_i \right) \right]^2
$$

### The SHAP Kernel

The critical insight is the specific weighting kernel:

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

where $|z'| = \sum_i z'_i$ is the coalition size. This kernel gives infinite weight to the empty and full coalitions (enforcing exact predictions at the extremes) and high weight to small and large coalitions (where marginal contributions are most informative).

| Coalition Size $|z'|$ | Weight (d=10) | Interpretation |
|----------------------|---------------|----------------|
| 0 or $d$ | $\infty$ | Exact boundary conditions |
| 1 or $d-1$ | High | Single-feature marginal effect |
| $d/2$ | Low | Most coalitions, least informative individually |

### Computing Coalition Values

For a coalition $S$ (set of features with $z'_i = 1$), the coalition value is:

$$
f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]
$$

In practice, this expectation is approximated by marginalizing over a background dataset:

$$
f_x(S) \approx \frac{1}{N} \sum_{j=1}^{N} f(x_S, x^{(j)}_{\bar{S}})
$$

where $x^{(j)}_{\bar{S}}$ are background values for absent features.

## Implementation

### Complete Kernel SHAP

```python
import numpy as np
from math import comb
from sklearn.linear_model import LinearRegression

class KernelSHAP:
    """
    Kernel SHAP - model-agnostic SHAP approximation.
    
    Approximates Shapley values using weighted linear regression
    with the SHAP kernel.
    """
    
    def __init__(self, model, background_data):
        """
        Args:
            model: Prediction function (numpy array -> predictions)
            background_data: Reference data for computing expectations
        """
        self.model = model
        self.background_data = background_data
        self.base_value = model(background_data).mean()
    
    def _shap_kernel(self, n_features, coalition_size):
        """SHAP kernel weight for coalition of given size."""
        if coalition_size == 0 or coalition_size == n_features:
            return 1e6  # Large weight for boundary coalitions
        
        return (n_features - 1) / (
            comb(n_features, coalition_size) * 
            coalition_size * (n_features - coalition_size)
        )
    
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
            instance: Input to explain (1D array)
            num_samples: Number of coalition samples
            
        Returns:
            SHAP values for each feature
        """
        n_features = len(instance)
        
        # Sample coalitions and compute values
        coalitions = []
        coalition_values = []
        weights = []
        
        for _ in range(num_samples):
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
        y = np.array(coalition_values) - self.base_value
        
        ZtWZ = Z.T @ W @ Z + 1e-6 * np.eye(n_features)
        ZtWy = Z.T @ W @ y
        
        shap_values = np.linalg.solve(ZtWZ, ZtWy)
        
        return shap_values
```

### Using the SHAP Library

```python
import shap

def kernel_shap_explain(model, X_train, X_test, feature_names):
    """
    Production Kernel SHAP using the shap library.
    """
    # Create explainer with background data
    explainer = shap.KernelExplainer(
        model.predict, 
        shap.sample(X_train, 100)  # Subsample background for efficiency
    )
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test[:100])
    
    # Summary plot
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    
    # Force plot for single instance
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X_test[0],
        feature_names=feature_names
    )
    
    return shap_values
```

## Computational Considerations

### Complexity

Exact Shapley values require evaluating all $2^d$ coalitions. Kernel SHAP approximates this by sampling coalitions, with complexity:

$$
O(N_{\text{samples}} \cdot N_{\text{background}} \cdot C_{\text{model}})
$$

where $C_{\text{model}}$ is the cost of a single model evaluation.

### Variance Reduction

Kernel SHAP's sampling introduces variance. Strategies to reduce it:

| Strategy | Effect | Cost |
|----------|--------|------|
| More coalition samples | Lower variance | More model evaluations |
| Paired sampling | Cancel first-order variance | Same evaluations |
| Stratified sampling | Better coverage of coalition sizes | Slight overhead |
| Larger background set | Better marginal estimates | More memory |

### When Kernel SHAP vs Alternatives

| Scenario | Recommendation |
|----------|---------------|
| Tree-based model | Use Tree SHAP (exact, fast) |
| Neural network | Use Deep SHAP or Gradient SHAP |
| Any black-box model | Kernel SHAP |
| Small feature set ($d < 15$) | Kernel SHAP works well |
| Large feature set ($d > 100$) | Consider approximate methods |

## Applications in Finance

```python
def explain_portfolio_allocation(
    allocation_model,
    market_features,
    feature_names,
    background_data
):
    """
    Explain portfolio allocation decisions using Kernel SHAP.
    """
    explainer = shap.KernelExplainer(
        allocation_model.predict, 
        background_data
    )
    
    shap_values = explainer.shap_values(market_features)
    
    print("Portfolio Allocation Explanation:")
    print("-" * 50)
    
    sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1]
    for idx in sorted_idx[:10]:
        direction = "↑ allocation" if shap_values[0][idx] > 0 else "↓ allocation"
        print(f"{feature_names[idx]:30s}: {shap_values[0][idx]:+.4f} ({direction})")
    
    return shap_values
```

## Summary

Kernel SHAP provides model-agnostic Shapley value approximation through weighted linear regression with a specially designed kernel. It inherits all theoretical properties of Shapley values while being applicable to any black-box model.

**Key equation:**

$$
\pi_x(z') = \frac{d - 1}{\binom{d}{|z'|} \cdot |z'| \cdot (d - |z'|)}
$$

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Covert, I., & Lee, S. I. (2021). "Improving KernelSHAP: Practical Shapley Value Estimation Using Linear Regression." *AISTATS*.

3. Shapley, L. S. (1953). "A Value for n-Person Games." *Contributions to the Theory of Games*.

## Exercises

**Exercise 1.**
Apply the interpretability method described in this section to a 2-layer neural network with ReLU activations classifying XOR inputs. Compute the explanation for the input $x = [1, 1]$.

??? success "Solution to Exercise 1"
    For a trained XOR network with weights $W_1, b_1, W_2, b_2$, the output is $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$. The explanation method produces attributions for each input feature. For $x = [1, 1]$ (class 0), both features contribute to the negative classification. The specific attribution values depend on the method: gradient-based methods compute $\partial f / \partial x_i$; perturbation-based methods measure output change when features are masked. The XOR problem demonstrates that linear explanation methods can mislead because the decision boundary is non-linear. $\square$

---

**Exercise 2.**
Prove or disprove that the explanation method in this section satisfies the completeness axiom: the sum of all feature attributions equals $f(x) - f(x_0)$ for some baseline $x_0$.

??? success "Solution to Exercise 2"
    The completeness axiom (also called efficiency in Shapley value theory) states that attributions sum to the difference between the model output at the input and at the baseline. Whether this method satisfies completeness depends on its formulation. Gradient methods do not satisfy completeness (gradients are local, not path-integrated). Integrated Gradients satisfies completeness by construction (fundamental theorem of calculus along the path). SHAP values satisfy efficiency by the Shapley axiom. Methods that violate completeness may over- or under-attribute, making the total attribution unreliable as a global explanation. $\square$

---

**Exercise 3.**
Design an experiment to evaluate the faithfulness of the explanations produced by this method. Use insertion and deletion curves to measure whether highlighted features are truly important to the model.

??? success "Solution to Exercise 3"
    Protocol: (1) Compute feature attributions for each test image. (2) Deletion: progressively mask features in order of decreasing attribution, recording the model confidence drop. Faithful explanations cause rapid confidence decrease. (3) Insertion: progressively reveal features in order of decreasing attribution from a blank baseline, recording confidence increase. Faithful explanations cause rapid confidence increase. (4) Compute AUC for both curves. (5) Compare against random ordering (baseline) and other methods. A faithful method should have low deletion AUC and high insertion AUC. Repeat over 1000+ test samples for statistical reliability. $\square$

---

**Exercise 4.**
Discuss how this interpretability method could be applied to a financial model predicting credit default. What regulatory requirements must the explanations satisfy?

??? success "Solution to Exercise 4"
    For credit models, regulations (ECOA, GDPR Article 22) require individualized explanations for adverse decisions. The method must produce: (1) the top factors contributing to the denial (adverse action reasons); (2) explanations that are consistent (similar applicants get similar explanations); (3) explanations that are actionable (the applicant understands what to change). The interpretability method from this section can identify feature importances, but must be validated for stability (small input changes should not drastically alter the explanation) and correctness (removing important features should change the prediction). Protected attributes must be handled carefully to avoid revealing proxy discrimination. $\square$
