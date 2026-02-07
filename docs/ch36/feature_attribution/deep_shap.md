# Deep SHAP

## Introduction

**Deep SHAP** combines DeepLIFT's efficient backpropagation-based attribution with SHAP's game-theoretic framework to compute approximate Shapley values for neural networks. By averaging DeepLIFT attributions over multiple reference points sampled from a background distribution, Deep SHAP inherits the theoretical guarantees of Shapley values while maintaining computational efficiency.

## Theoretical Foundation

### DeepLIFT as Building Block

DeepLIFT (Shrikumar et al., 2017) explains neural network predictions by comparing neuron activations to their "reference" activations. For an input $\mathbf{x}$ and reference $\mathbf{x}^0$:

$$
\Delta y = f(\mathbf{x}) - f(\mathbf{x}^0)
$$

DeepLIFT computes contribution scores $C_{\Delta x_i \Delta y}$ satisfying the **summation-to-delta property**:

$$
\sum_i C_{\Delta x_i \Delta y} = \Delta y
$$

### The Rescale Rule

For nonlinear activations, DeepLIFT uses the rescale rule to propagate contributions:

$$
m_{\Delta x \Delta y} = \frac{\Delta y}{\Delta x}
$$

This avoids the **saturation problem** where gradients become zero for saturated activations (sigmoid at extremes, ReLU for negative inputs), even when inputs clearly matter.

### From DeepLIFT to Deep SHAP

Deep SHAP extends DeepLIFT by averaging over multiple references sampled from a background distribution $D$:

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

This averaging approximates the Shapley value computation, which considers all possible coalitions. The key insight: marginalizing over references is equivalent to marginalizing over feature coalitions under independence assumptions.

### Connection to Shapley Values

For a network with piecewise-linear activations (ReLU), DeepLIFT with a single reference computes exact Shapley values for a **linearized** version of the model around the path from reference to input. Deep SHAP improves this by:

1. Averaging over multiple references → better approximation of the full Shapley expectation
2. Handling non-linear activations → the rescale rule captures non-linear effects that gradients miss

## Implementation

### PyTorch Deep SHAP

```python
import torch
import torch.nn as nn
import numpy as np

class DeepSHAP:
    """
    Deep SHAP using DeepLIFT-style backpropagation
    averaged over multiple background samples.
    """
    
    def __init__(self, model: nn.Module, background: torch.Tensor):
        """
        Args:
            model: PyTorch neural network
            background: Background samples [N, features] for reference distribution
        """
        self.model = model
        self.background = background
        
        with torch.no_grad():
            self.base_output = model(background).mean(dim=0)
    
    def _deep_lift_gradient(self, x, baseline, target_class):
        """Compute DeepLIFT-style attributions for one baseline."""
        x = x.clone().requires_grad_(True)
        baseline = baseline.clone().requires_grad_(True)
        
        output_x = self.model(x)
        output_baseline = self.model(baseline)
        
        diff = output_x[:, target_class] - output_baseline[:, target_class]
        
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
        Compute Deep SHAP values.
        
        Args:
            instance: Input tensor [1, features]
            target_class: Target class for explanation
            n_samples: Number of background samples to average over
            
        Returns:
            SHAP values tensor [1, features]
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
    
    def verify_completeness(
        self,
        instance: torch.Tensor,
        shap_values: torch.Tensor,
        target_class: int
    ) -> float:
        """
        Verify that SHAP values sum to prediction - base value.
        """
        with torch.no_grad():
            prediction = self.model(instance)[0, target_class].item()
        
        base = self.base_output[target_class].item()
        shap_sum = shap_values.sum().item()
        expected_sum = prediction - base
        
        error = abs(shap_sum - expected_sum)
        print(f"Prediction: {prediction:.4f}")
        print(f"Base value: {base:.4f}")
        print(f"SHAP sum:   {shap_sum:.4f}")
        print(f"Expected:   {expected_sum:.4f}")
        print(f"Error:      {error:.6f}")
        
        return error
```

### Using the SHAP Library

```python
import shap
import torch

def deep_shap_with_library(model, background, test_samples):
    """
    Deep SHAP using the shap library's optimized implementation.
    """
    # DeepExplainer implements Deep SHAP
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    return shap_values


def gradient_shap_alternative(model, test_samples, baselines):
    """
    GradientSHAP via Captum - another approach to neural network SHAP.
    """
    from captum.attr import GradientShap
    
    gradient_shap = GradientShap(model)
    attribution = gradient_shap.attribute(
        test_samples,
        baselines=baselines,
        target=0  # target class
    )
    
    return attribution
```

## Handling the Saturation Problem

A key advantage of Deep SHAP over pure gradient methods is handling saturated activations.

Consider a sigmoid neuron with input $x = 10$ and reference $x^0 = 0$:

| Method | Attribution | Issue |
|--------|------------|-------|
| Gradient | $\sigma'(10) \approx 0.000045$ | Nearly zero despite large activation difference |
| DeepLIFT/Deep SHAP | $(\sigma(10) - \sigma(0)) / (10 - 0) \approx 0.05$ | Captures meaningful contribution |

For ReLU with $x = -10$, $x^0 = 0$:

| Method | Attribution | Issue |
|--------|------------|-------|
| Gradient | $0$ (inactive ReLU) | Misses the fact that being inactive is informative |
| DeepLIFT/Deep SHAP | $0$ | Correctly attributes zero (same output as reference) |

## Comparison with Other Neural Network Attribution Methods

| Method | Completeness | Saturation Handling | Speed | Multiple Baselines |
|--------|-------------|-------------------|-------|-------------------|
| Vanilla Gradients | No | Poor | Very fast | N/A |
| Integrated Gradients | Yes | Good | Slow (many steps) | Single path |
| DeepLIFT | Yes | Good | Fast | Single reference |
| **Deep SHAP** | **Yes** | **Good** | **Moderate** | **Yes (averaged)** |
| Gradient SHAP | Approximate | Moderate | Fast | Yes |

## Background Selection

The background distribution significantly affects Deep SHAP results:

| Data Type | Recommended Background |
|-----------|----------------------|
| Images | Training set subsample, zero images, or Gaussian noise |
| Tabular | Representative training subsample (100-1000 samples) |
| Time series | Historical baseline periods |
| Text | Padding token embeddings |
| Financial | Market-neutral state or historical average |

**Guidelines:**

- Use at least 100 background samples for stable estimates
- Background should represent the "uninformed" baseline state
- Avoid using the test instance's neighbors as background (leaks information)

## Applications in Quantitative Finance

```python
def explain_neural_risk_model(
    risk_model: nn.Module,
    portfolio_features: torch.Tensor,
    background: torch.Tensor,
    feature_names: list
):
    """
    Explain neural network risk predictions using Deep SHAP.
    """
    explainer = DeepSHAP(risk_model, background)
    
    shap_values = explainer.explain(
        portfolio_features.unsqueeze(0),
        target_class=0,  # Risk score
        n_samples=200
    )
    
    values = shap_values.squeeze().detach().cpu().numpy()
    
    sorted_idx = np.argsort(np.abs(values))[::-1]
    
    print("Risk Factor Attribution (Deep SHAP):")
    print("-" * 50)
    for idx in sorted_idx[:10]:
        direction = "↑ risk" if values[idx] > 0 else "↓ risk"
        print(f"{feature_names[idx]:30s}: {values[idx]:+.6f} ({direction})")
    
    return shap_values
```

## Summary

Deep SHAP provides theoretically grounded neural network attribution by combining DeepLIFT's efficient backpropagation with SHAP's Shapley value framework. By averaging over multiple background references, it approximates true Shapley values while handling activation saturation that defeats pure gradient methods.

**Key equation:**

$$
\phi_i(\mathbf{x}) = \mathbb{E}_{\mathbf{x}^0 \sim D}\left[C_{\Delta x_i \Delta y}\right]
$$

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.

2. Shrikumar, A., et al. (2017). "Learning Important Features Through Propagating Activation Differences." *ICML*.

3. Ancona, M., et al. (2018). "Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks." *ICLR*.

4. Erion, G., et al. (2021). "Improving Performance of Deep Learning Models with Axiomatic Attribution Priors and Expected Gradients." *Nature Machine Intelligence*.
