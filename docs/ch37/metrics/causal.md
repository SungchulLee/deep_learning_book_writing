# Causal Fairness Metrics

## Overview

Causal fairness metrics go beyond statistical associations to measure the *causal effect* of protected attributes on predictions. While statistical metrics detect correlation-based disparities, causal metrics distinguish between legitimate and illegitimate pathways from $A$ to $\hat{Y}$.

## Causal Framework

### Total, Direct, and Indirect Effects

Given a causal graph with protected attribute $A$, features $X$, mediators $M$, and prediction $\hat{Y}$:

$$\text{Total Effect (TE)} = \mathbb{E}[\hat{Y} \mid do(A=1)] - \mathbb{E}[\hat{Y} \mid do(A=0)]$$

$$\text{Natural Direct Effect (NDE)} = \mathbb{E}[\hat{Y}_{A=1, M_{A=0}}] - \mathbb{E}[\hat{Y}_{A=0, M_{A=0}}]$$

$$\text{Natural Indirect Effect (NIE)} = \mathbb{E}[\hat{Y}_{A=0, M_{A=1}}] - \mathbb{E}[\hat{Y}_{A=0, M_{A=0}}]$$

The NDE captures the effect of $A$ on $\hat{Y}$ holding mediators fixed, while the NIE captures the effect transmitted through mediators.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CausalFairnessMetrics:
    """Causal fairness metrics."""
    total_effect: float
    direct_effect: float
    indirect_effect: float
    path_specific_effects: Dict[str, float]

class CausalFairnessCalculator:
    """
    Estimate causal fairness metrics using interventional distributions.
    
    Approximates do-calculus quantities via simulation under
    structural causal models.
    """
    
    def estimate_total_effect(
        self,
        model: nn.Module,
        X: torch.Tensor,
        A: torch.Tensor,
        a_idx: int,
    ) -> float:
        """
        Estimate total effect: E[Ŷ|do(A=1)] - E[Ŷ|do(A=0)]
        
        Approximated by swapping A values and comparing predictions.
        """
        model.eval()
        with torch.no_grad():
            X_a0, X_a1 = X.clone(), X.clone()
            X_a0[:, a_idx] = 0.0
            X_a1[:, a_idx] = 1.0
            
            pred_a0 = model(X_a0).squeeze(-1).mean().item()
            pred_a1 = model(X_a1).squeeze(-1).mean().item()
        
        return pred_a1 - pred_a0
    
    def estimate_direct_effect(
        self,
        model: nn.Module,
        X: torch.Tensor,
        a_idx: int,
        mediator_indices: List[int],
    ) -> float:
        """
        Estimate NDE by intervening on A while holding mediators fixed.
        """
        model.eval()
        with torch.no_grad():
            # Baseline: A=0, mediators at natural values
            X_base = X.clone()
            X_base[:, a_idx] = 0.0
            
            # Intervention: A=1, but keep mediators from A=0 world
            X_direct = X.clone()
            X_direct[:, a_idx] = 1.0
            for m_idx in mediator_indices:
                X_direct[:, m_idx] = X_base[:, m_idx]
            
            pred_base = model(X_base).squeeze(-1).mean().item()
            pred_direct = model(X_direct).squeeze(-1).mean().item()
        
        return pred_direct - pred_base
    
    def compute(
        self,
        model: nn.Module,
        X: torch.Tensor,
        a_idx: int,
        mediator_indices: List[int],
    ) -> CausalFairnessMetrics:
        """Compute all causal fairness metrics."""
        te = self.estimate_total_effect(model, X, torch.zeros(len(X)), a_idx)
        nde = self.estimate_direct_effect(model, X, a_idx, mediator_indices)
        nie = te - nde  # TE ≈ NDE + NIE (approximately)
        
        return CausalFairnessMetrics(
            total_effect=te,
            direct_effect=nde,
            indirect_effect=nie,
            path_specific_effects={
                'direct': nde,
                'via_mediators': nie,
            },
        )
```

## When to Use Causal Metrics

- When you have a **known or hypothesized causal graph**
- When some pathways from $A$ to $\hat{Y}$ are considered **legitimate** (e.g., $A \to \text{qualification} \to \hat{Y}$)
- When statistical metrics are insufficient to distinguish bias from legitimate variation

## Summary

- Causal metrics decompose the effect of $A$ on $\hat{Y}$ into **direct** and **indirect** components
- Require a **causal model** (at minimum, knowledge of which features are mediators)
- Complement statistical metrics by distinguishing legitimate from illegitimate pathways

## Next Steps

- [Multi-Group Metrics](multi_group.md): Extending metrics beyond two groups
- [Intersectionality](intersectionality.md): Analyzing overlapping protected attributes
