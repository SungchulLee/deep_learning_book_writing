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
