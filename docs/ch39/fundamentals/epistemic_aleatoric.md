# Epistemic vs Aleatoric Uncertainty

## Introduction

Understanding the distinction between epistemic and aleatoric uncertainty is fundamental to designing uncertainty-aware systems. These two types of uncertainty have different sources, different implications, and require different modeling approaches.

## Epistemic Uncertainty (Model Uncertainty)

**Definition**: Uncertainty arising from lack of knowledge about the true model or its parameters.

**Characteristics**:

- Results from limited training data
- Can be reduced with more data
- High in regions far from training distribution
- Represents what the model "doesn't know it doesn't know"

**Mathematical representation**:

$$\text{Epistemic} = \text{Var}_{\mathbf{w} \sim p(\mathbf{w}|\mathcal{D})}[\mathbb{E}[y|\mathbf{x}, \mathbf{w}]]$$

This captures how much the model's predictions vary across different plausible parameter settings.

## Aleatoric Uncertainty (Data Uncertainty)

**Definition**: Uncertainty arising from inherent randomness in the data generation process.

**Characteristics**:

- Results from noise, ambiguity, or class overlap
- Cannot be reduced with more data
- Represents irreducible variability
- Present even with perfect model knowledge

**Mathematical representation**:

$$\text{Aleatoric} = \mathbb{E}_{\mathbf{w} \sim p(\mathbf{w}|\mathcal{D})}[\text{Var}[y|\mathbf{x}, \mathbf{w}]]$$

This captures the expected variance of the output given fixed model parameters.

## Visual Intuition

Consider a regression problem with sparse training data:

```
    y
    │     ·   ·        High Epistemic
    │   ·       ·      (no training data)
    │ ·           ·    
    │·    ████     ····················
    │   ██████████     Low Epistemic
    │  ████████████    (dense training data)
    │ ██·█·██·███·█    High Aleatoric
    │  ████████████    (noisy labels)
    └──────────────────── x
```

**Region with sparse data**: High epistemic uncertainty (model unsure).  
**Region with dense but noisy data**: High aleatoric uncertainty (data inherently noisy).

## Uncertainty Decomposition via Law of Total Variance

The total predictive variance decomposes exactly:

$$\text{Var}[y|\mathbf{x}, \mathcal{D}] = \underbrace{\mathbb{E}_{\mathbf{w}}[\text{Var}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}_{\mathbf{w}}[\mathbb{E}[y|\mathbf{x}, \mathbf{w}]]}_{\text{Epistemic}}$$

This decomposition provides a principled way to separate the two types of uncertainty. With an ensemble or MC Dropout, the epistemic component is approximated by the variance of predictions across members/passes, while the aleatoric component comes from each member's own predicted variance.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


def generate_uncertainty_demo_data(
    n_samples: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data demonstrating both types of uncertainty.
    
    Creates:
    - Gap region [-1, 1]: High epistemic uncertainty (no training data)
    - Right region [1, 3]: High aleatoric uncertainty (noisy labels)
    """
    np.random.seed(42)
    
    # Left region: dense data, low noise
    X_left = np.linspace(-3, -1, n_samples // 2)
    noise_left = np.random.normal(0, 0.1, len(X_left))
    
    # Right region: dense data, high noise
    X_right = np.linspace(1, 3, n_samples // 2)
    noise_right = np.random.normal(0, 0.4, len(X_right))
    
    def true_fn(x):
        return np.sin(2 * x)
    
    X = np.concatenate([X_left, X_right])
    y = np.concatenate([
        true_fn(X_left) + noise_left,
        true_fn(X_right) + noise_right
    ])
    
    return X, y


def decompose_uncertainty(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.
    
    For classification with ensemble:
        predictions: (n_models, batch_size, n_classes) - probabilities
    
    Args:
        predictions: Ensemble predictions
    
    Returns:
        Dictionary with 'epistemic', 'aleatoric', and 'total' uncertainty
    """
    # Mean prediction across ensemble
    mean_pred = predictions.mean(dim=0)  # (batch_size, n_classes)
    
    # EPISTEMIC: Variance of predictions across models
    epistemic = predictions.var(dim=0).mean(dim=-1)  # (batch_size,)
    
    # ALEATORIC approximation: entropy of mean prediction
    epsilon = 1e-10
    aleatoric = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=-1)
    
    # Total uncertainty
    total = epistemic + aleatoric
    
    return {
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total
    }


def demonstrate_decomposition():
    """
    Demonstrate uncertainty decomposition with synthetic examples.
    """
    n_models = 5
    n_samples = 100
    n_classes = 3
    
    # Case 1: High epistemic, low aleatoric
    # Models disagree, but each is confident
    preds_epistemic = []
    for i in range(n_models):
        class_idx = i % n_classes
        probs = torch.zeros(n_samples, n_classes)
        probs[:, class_idx] = 0.9
        probs[:, (class_idx + 1) % n_classes] = 0.08
        probs[:, (class_idx + 2) % n_classes] = 0.02
        preds_epistemic.append(probs)
    
    preds_epistemic = torch.stack(preds_epistemic)
    unc = decompose_uncertainty(preds_epistemic)
    print(f"Case 1 - Models DISAGREE, each CONFIDENT:")
    print(f"  Epistemic: {unc['epistemic'].mean():.4f}")
    print(f"  Aleatoric: {unc['aleatoric'].mean():.4f}")
    
    # Case 2: Low epistemic, high aleatoric
    # Models agree, but all are uncertain
    preds_aleatoric = []
    for i in range(n_models):
        probs = torch.ones(n_samples, n_classes) / n_classes
        probs += torch.randn(n_samples, n_classes) * 0.02
        probs = F.softmax(probs, dim=1)
        preds_aleatoric.append(probs)
    
    preds_aleatoric = torch.stack(preds_aleatoric)
    unc = decompose_uncertainty(preds_aleatoric)
    print(f"\nCase 2 - Models AGREE, all UNCERTAIN:")
    print(f"  Epistemic: {unc['epistemic'].mean():.4f}")
    print(f"  Aleatoric: {unc['aleatoric'].mean():.4f}")
    
    # Case 3: Low epistemic, low aleatoric
    # Models agree and are confident
    preds_confident = []
    for i in range(n_models):
        probs = torch.zeros(n_samples, n_classes)
        probs[:, 0] = 0.95
        probs[:, 1] = 0.04
        probs[:, 2] = 0.01
        preds_confident.append(probs)
    
    preds_confident = torch.stack(preds_confident)
    unc = decompose_uncertainty(preds_confident)
    print(f"\nCase 3 - Models AGREE, all CONFIDENT:")
    print(f"  Epistemic: {unc['epistemic'].mean():.4f}")
    print(f"  Aleatoric: {unc['aleatoric'].mean():.4f}")
```

## Heteroscedastic Models for Aleatoric Uncertainty

To explicitly model aleatoric uncertainty, we can have the network predict both mean and variance:

```python
class HeteroscedasticNetwork(nn.Module):
    """
    Network that predicts both mean and variance (aleatoric uncertainty).
    
    Output:
        μ(x): Predicted mean
        σ²(x): Predicted variance (aleatoric uncertainty)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean prediction head
        self.mean_head = nn.Linear(hidden_dim, 1)
        
        # Log-variance head (log for numerical stability)
        self.logvar_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: Predicted mean μ(x)
            variance: Predicted variance σ²(x)
        """
        features = self.shared(x)
        mean = self.mean_head(features)
        log_var = self.logvar_head(features)
        variance = torch.exp(log_var)  # Ensure positive variance
        return mean, variance
    
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss for heteroscedastic regression.
        
        NLL = 0.5 * log(σ²) + 0.5 * (y - μ)² / σ²
        """
        mean, variance = self.forward(x)
        nll = 0.5 * torch.log(variance) + 0.5 * (y - mean) ** 2 / variance
        return nll.mean()
```

## Practical Decision Framework

| Uncertainty Type | Action | Example |
|-----------------|--------|---------|
| High Epistemic | Collect more data | Model unsure on new region |
| High Aleatoric | Accept irreducible noise | Sensor measurement error |
| Both High | Investigate data quality | Mislabeled or ambiguous samples |
| Both Low | Trust prediction | Clear, well-represented case |

```python
def make_decision(
    epistemic: float, aleatoric: float,
    epi_threshold: float = 0.1, ale_threshold: float = 0.3
) -> str:
    """Decision framework based on uncertainty decomposition."""
    high_epi = epistemic > epi_threshold
    high_ale = aleatoric > ale_threshold
    
    if not high_epi and not high_ale:
        return "ACCEPT: Low uncertainty, trust prediction"
    elif high_epi and not high_ale:
        return "COLLECT DATA: Model needs more training examples"
    elif not high_epi and high_ale:
        return "ACCEPT WITH CAUTION: Inherent data noise"
    else:
        return "INVESTIGATE: Both types high, check data quality"
```

## Quantitative Finance Implications

The epistemic-aleatoric distinction is particularly valuable in finance:

**Epistemic uncertainty is high** during regime changes, for novel market conditions, and for thinly traded instruments. This signals the model is extrapolating and positions should be reduced.

**Aleatoric uncertainty is high** for inherently volatile assets, during earnings announcements, and for instruments with noisy price data. This is expected and should be incorporated into position sizing but doesn't indicate model failure.

A trading system that conflates the two may reduce positions during high-volatility periods (when expected returns may actually be high) or maintain large positions during regime changes (when the model is unreliable).

## Key Takeaways

!!! success "Summary"
    1. **Epistemic uncertainty** reflects model ignorance—reducible with more data
    2. **Aleatoric uncertainty** reflects data noise—irreducible even with infinite data
    3. **Total variance decomposes** exactly into these two components via the law of total variance
    4. **Different actions** are appropriate for each uncertainty type
    5. **Heteroscedastic models** explicitly predict input-dependent aleatoric uncertainty
    6. **Financial applications** benefit from the distinction: epistemic triggers position reduction, aleatoric informs risk budgeting

## References

- Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
- Der Kiureghian, A., & Ditlevsen, O. (2009). "Aleatory or epistemic? Does it matter?"
- Hüllermeier, E., & Waegeman, W. (2021). "Aleatoric and Epistemic Uncertainty with Random Forests"
