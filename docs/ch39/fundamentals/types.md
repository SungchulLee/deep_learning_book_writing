# Types of Uncertainty

## Overview

Understanding the landscape of uncertainty in deep learning requires distinguishing between fundamentally different sources of unpredictability. This section introduces the taxonomy of uncertainty types and their mathematical characterization.

## Uncertainty Taxonomy

Uncertainty in neural network predictions arises from multiple sources, each requiring different modeling approaches and leading to different practical responses.

### Parameter Uncertainty

Parameter uncertainty reflects our ignorance about the true values of model weights. Given finite training data, many weight configurations are consistent with the observations. The posterior distribution $p(\mathbf{w}|\mathcal{D})$ captures this uncertainty—wider posteriors indicate greater parameter uncertainty.

In standard training, we collapse this distribution to a point estimate $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}} p(\mathbf{w}|\mathcal{D})$, discarding all information about parameter uncertainty.

### Structural Uncertainty

Even with infinite data, the model architecture may be wrong. A linear model fit to nonlinear data has structural uncertainty—no amount of training data can fix the mismatch. This form of uncertainty is difficult to quantify within a single model but can be addressed through Bayesian model comparison or ensemble methods that combine diverse architectures.

### Data Uncertainty

Data uncertainty encompasses noise in the observations themselves: measurement error, label noise, inherent stochasticity in the data-generating process, and class overlap in classification problems. This uncertainty persists even with a perfect model and infinite data.

### Distributional Uncertainty

When test inputs differ from the training distribution, the model extrapolates beyond its experience. Distributional uncertainty captures the model's unreliability on out-of-distribution (OOD) inputs—a critical concern in financial applications where market regimes shift.

## The Practical Taxonomy

For practical purposes, these sources collapse into two fundamental categories:

$$\text{Total Uncertainty} = \text{Epistemic Uncertainty} + \text{Aleatoric Uncertainty}$$

**Epistemic uncertainty** (from Greek *episteme*, knowledge) encompasses parameter and structural uncertainty—things the model doesn't know that it could learn with more data or better architecture.

**Aleatoric uncertainty** (from Latin *alea*, dice) encompasses data uncertainty—irreducible randomness inherent in the observations.

The next section develops this decomposition rigorously.

## Uncertainty Measures for Classification

For classification problems, several measures capture different aspects of uncertainty:

### Softmax Confidence

The simplest uncertainty proxy is one minus the maximum softmax probability:

$$u(\mathbf{x}) = 1 - \max_c p(y = c | \mathbf{x})$$

This is easy to compute but poorly calibrated in modern networks.

### Predictive Entropy

Entropy of the predictive distribution captures total uncertainty:

$$\mathbb{H}[y|\mathbf{x}, \mathcal{D}] = -\sum_{c=1}^{C} p(y=c|\mathbf{x}, \mathcal{D}) \log p(y=c|\mathbf{x}, \mathcal{D})$$

### Mutual Information

Mutual information between the prediction and model parameters isolates epistemic uncertainty:

$$\mathbb{I}[y; \mathbf{w} | \mathbf{x}, \mathcal{D}] = \mathbb{H}[y|\mathbf{x}, \mathcal{D}] - \mathbb{E}_{p(\mathbf{w}|\mathcal{D})}[\mathbb{H}[y|\mathbf{x}, \mathbf{w}]]$$

This equals the entropy of the mean prediction minus the mean of individual entropies. High mutual information means model members disagree—a hallmark of epistemic uncertainty.

## Uncertainty Measures for Regression

### Predictive Variance

For regression, the total predictive variance decomposes via the law of total variance:

$$\text{Var}[y|\mathbf{x}, \mathcal{D}] = \underbrace{\mathbb{E}_{\mathbf{w}}[\text{Var}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\mathbf{w}}[\mathbb{E}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Epistemic}}$$

### Prediction Intervals vs Confidence Intervals

**Prediction intervals** bound future observations: "The next return will be in $[-2\%, 3\%]$ with 95% probability." These incorporate both aleatoric and epistemic uncertainty.

**Confidence intervals** bound the estimated mean prediction: "The expected return is in $[0.1\%, 0.8\%]$ with 95% probability." These capture only epistemic uncertainty about the mean.

For risk management in finance, prediction intervals are typically more relevant.

## PyTorch Implementation

```python
import torch
import torch.nn.functional as F
from typing import Dict


def compute_classification_uncertainty(
    mc_probs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute uncertainty measures from MC predictions.
    
    Args:
        mc_probs: (T, batch_size, n_classes) softmax probabilities
                  from T forward passes
    
    Returns:
        Dictionary with uncertainty measures
    """
    epsilon = 1e-10
    
    # Mean predictive distribution
    mean_probs = mc_probs.mean(dim=0)  # (batch, classes)
    
    # Confidence (1 - max prob)
    confidence = mean_probs.max(dim=-1).values
    
    # Predictive entropy (total uncertainty)
    predictive_entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + epsilon), dim=-1
    )
    
    # Expected entropy (aleatoric uncertainty)
    sample_entropies = -torch.sum(
        mc_probs * torch.log(mc_probs + epsilon), dim=-1
    )  # (T, batch)
    expected_entropy = sample_entropies.mean(dim=0)
    
    # Mutual information (epistemic uncertainty)
    mutual_info = predictive_entropy - expected_entropy
    
    # Variance of predictions across samples
    prediction_variance = mc_probs.var(dim=0).mean(dim=-1)
    
    return {
        'confidence': confidence,
        'predictive_entropy': predictive_entropy,
        'expected_entropy': expected_entropy,
        'mutual_information': mutual_info,
        'prediction_variance': prediction_variance,
    }


def compute_regression_uncertainty(
    mc_means: torch.Tensor,
    mc_log_vars: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """
    Compute regression uncertainty decomposition.
    
    Args:
        mc_means: (T, batch_size, output_dim) predicted means
        mc_log_vars: (T, batch_size, output_dim) predicted log-variances
                     (for heteroscedastic models)
    """
    # Epistemic: variance of means across samples
    epistemic_var = mc_means.var(dim=0)
    pred_mean = mc_means.mean(dim=0)
    
    if mc_log_vars is not None:
        aleatoric_var = mc_log_vars.exp().mean(dim=0)
    else:
        aleatoric_var = torch.zeros_like(epistemic_var)
    
    total_var = epistemic_var + aleatoric_var
    
    return {
        'mean': pred_mean,
        'epistemic_var': epistemic_var,
        'aleatoric_var': aleatoric_var,
        'total_var': total_var,
        'total_std': torch.sqrt(total_var),
    }
```

## Key Takeaways

!!! success "Summary"
    1. **Multiple sources** of uncertainty exist: parameter, structural, data, and distributional
    2. **Two practical categories**: epistemic (reducible) and aleatoric (irreducible)
    3. **Classification uncertainty** is measured by entropy, mutual information, and softmax confidence
    4. **Regression uncertainty** decomposes via the law of total variance
    5. **Prediction intervals** and **confidence intervals** capture different things
