# Factor Attribution
## Introduction

Factor attribution decomposes portfolio returns and risk into contributions from systematic factors. Interpretability methods—particularly SHAP—provide a rigorous framework for understanding how factor exposures drive model predictions, enabling portfolio managers to validate economic intuition and identify hidden dependencies.

## Understanding Factor Exposures

### Linear Factor Model Interpretation

For linear factor models, coefficients are directly interpretable:

$$
r_p = \alpha + \sum_{k=1}^{K} \beta_k f_k + \epsilon
$$

where $\beta_k$ is the exposure to factor $k$ and $f_k$ is the factor return.

### Non-Linear Factor Models

When using neural networks or tree-based models for return prediction, factor contributions are no longer simple coefficients. SHAP values provide the appropriate decomposition:

$$
\hat{r}_p = \phi_0 + \sum_{k=1}^{K} \phi_k
$$

where $\phi_k$ is the SHAP value for factor $k$.

## Implementation

### Factor Model Explainer

```python
import numpy as np
import shap
import matplotlib.pyplot as plt

class FactorModelExplainer:
    """Interpret factor model predictions."""
    
    def __init__(self, model, factor_names):
        self.model = model
        self.factor_names = factor_names
    
    def explain_return_forecast(self, factor_exposures):
        """Explain predicted return decomposition."""
        predicted_return = self.model.predict(
            factor_exposures.reshape(1, -1)
        )[0]
        
        if hasattr(self.model, 'coef_'):
            # Linear model: direct interpretation
            factor_contributions = self.model.coef_ * factor_exposures
            intercept = self.model.intercept_
        else:
            # Non-linear model: use SHAP
            explainer = shap.Explainer(self.model)
            shap_values = explainer(factor_exposures.reshape(1, -1))
            factor_contributions = shap_values.values[0]
            intercept = shap_values.base_values[0]
        
        return {
            'predicted_return': predicted_return,
            'alpha': intercept,
            'factor_contributions': dict(
                zip(self.factor_names, factor_contributions)
            )
        }
    
    def visualize_decomposition(self, explanation):
        """Create waterfall chart of return decomposition."""
        factors = list(explanation['factor_contributions'].keys())
        contributions = list(explanation['factor_contributions'].values())
        
        sorted_idx = np.argsort(np.abs(contributions))[::-1]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cumsum = explanation['alpha']
        positions = []
        
        for i, idx in enumerate(sorted_idx):
            contrib = contributions[idx]
            left = cumsum if contrib > 0 else cumsum + contrib
            width = abs(contrib)
            color = 'green' if contrib > 0 else 'red'
            
            ax.barh(i, width, left=left, color=color, alpha=0.7)
            ax.text(left + width/2, i, f'{contrib:.2%}', 
                   ha='center', va='center')
            
            cumsum += contrib
            positions.append(factors[idx])
        
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(positions)
        ax.set_xlabel('Contribution to Return')
        ax.set_title(f'Return Decomposition (Total: {explanation["predicted_return"]:.2%})')
        ax.axvline(x=0, color='black', linestyle='-')
        
        return fig
    
    def factor_interaction_analysis(self, factor_data):
        """Analyze interactions between factors."""
        explainer = shap.TreeExplainer(self.model)
        interactions = explainer.shap_interaction_values(factor_data[:100])
        
        mean_interactions = np.abs(interactions).mean(axis=0)
        
        # Off-diagonal: interactions; diagonal: main effects
        main_effects = np.diag(mean_interactions)
        
        print("Factor Main Effects vs Interaction Strength:")
        print("-" * 60)
        for i, name in enumerate(self.factor_names):
            interaction_total = mean_interactions[i].sum() - main_effects[i]
            ratio = interaction_total / (main_effects[i] + 1e-10)
            print(f"{name:20s}: main={main_effects[i]:.4f}, "
                  f"interaction={interaction_total:.4f}, ratio={ratio:.2f}")
        
        return interactions
```

## Time-Varying Attribution

Factor contributions change over time. Tracking SHAP values across a rolling window reveals regime shifts:

```python
def rolling_factor_attribution(model, factor_data, factor_names, window=60):
    """Compute time-varying factor attribution."""
    n_periods = len(factor_data) - window
    attributions = np.zeros((n_periods, len(factor_names)))
    
    explainer = shap.Explainer(model)
    
    for t in range(n_periods):
        shap_values = explainer(factor_data[t + window:t + window + 1])
        attributions[t] = shap_values.values[0]
    
    return attributions
```

## Summary

Factor attribution using SHAP values provides a theoretically grounded decomposition of return predictions, applicable to both linear and non-linear models. Time-varying analysis reveals regime-dependent factor dynamics.

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
2. Feng, G., Giglio, S., & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors." *Journal of Finance*.

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
