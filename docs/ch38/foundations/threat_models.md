# Threat Models

## Introduction

A **threat model** formalizes the assumptions about what an adversary knows, can do, and seeks to achieve. Choosing the right threat model is critical: too weak and the evaluation is meaningless; too strong and the defense is impractical. Different threat models lead to qualitatively different attacks and defenses.

## Adversary Knowledge

### White-Box Attacks

In the **white-box** setting, the adversary has complete access to:

- Model architecture $f_\theta$
- Model parameters $\theta$
- Training data distribution $\mathcal{D}$
- Gradient computation $\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y)$

This represents the **strongest attacker** and worst-case scenario for defenders. White-box attacks establish upper bounds on model vulnerability.

**Implications:**

- The attacker can compute exact gradients for optimization-based attacks
- The attacker can identify and exploit architecture-specific weaknesses
- Any defense must be robust against arbitrary gradient-based perturbations
- Security evaluation should always include white-box assessment

### Black-Box Attacks

In the **black-box** setting, the adversary can only:

- Query the model with inputs and observe outputs
- Observe predicted class labels $\hat{y} = f_\theta(\mathbf{x})$
- Optionally observe confidence scores $p(\hat{y}|\mathbf{x})$

Black-box attacks use several strategies:

1. **Score-based attacks**: Use confidence scores to estimate gradients via finite differences
2. **Decision-based attacks**: Use only final predictions to explore the decision boundary
3. **Transfer attacks**: Attack a surrogate model and transfer perturbations to the target

### Gray-Box Attacks

**Gray-box** covers intermediate scenarios:

- Architecture known, weights unknown
- Training procedure known, exact model unknown
- Access to a related model (same task, different training run)

This is often the most realistic setting in practice, particularly for deployed financial systems where model architecture may be inferred from public documentation or reverse engineering.

## Adversary Goals

### Untargeted Attacks

The goal is to cause **any misclassification**:

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) \neq y, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**Optimization formulation:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y)
$$

This maximizes loss with respect to the true label, pushing predictions away from the correct class.

### Targeted Attacks

The goal is to force prediction of a **specific class** $y_{\text{target}}$:

$$
\text{Find } \boldsymbol{\delta} \text{ such that } f_\theta(\mathbf{x} + \boldsymbol{\delta}) = y_{\text{target}}, \quad \|\boldsymbol{\delta}\|_p \leq \varepsilon
$$

**Optimization formulation:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \mathcal{L}(f_\theta(\mathbf{x} + \boldsymbol{\delta}), y_{\text{target}})
$$

### Confidence Attacks

Beyond misclassification, attacks may target model **confidence**:

**High-confidence misclassification:**

$$
\boldsymbol{\delta}^* = \arg\max_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} \left[ \max_{y' \neq y} \log p(y' | \mathbf{x} + \boldsymbol{\delta}) - \log p(y | \mathbf{x} + \boldsymbol{\delta}) \right]
$$

**Low-confidence correct classification:**

$$
\boldsymbol{\delta}^* = \arg\min_{\|\boldsymbol{\delta}\|_p \leq \varepsilon} p(y | \mathbf{x} + \boldsymbol{\delta})
$$

### Comparison of Goals

| Aspect | Untargeted | Targeted | Confidence |
|--------|-----------|----------|------------|
| Optimization | Maximize loss (true label) | Minimize loss (target label) | Maximize margin |
| Gradient direction | Ascend | Descend | Varies |
| Difficulty | Easier | Harder | Moderate |
| Perturbation size | Typically smaller | Typically larger | Variable |
| Financial relevance | Evasion attacks | Impersonation attacks | Calibration attacks |

## PyTorch Implementation

### Threat Model Configuration

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class AttackerKnowledge(Enum):
    WHITE_BOX = "white_box"
    BLACK_BOX = "black_box"
    GRAY_BOX = "gray_box"

class AttackerGoal(Enum):
    UNTARGETED = "untargeted"
    TARGETED = "targeted"
    CONFIDENCE = "confidence"

@dataclass
class ThreatModel:
    """
    Specification of adversarial threat model.
    
    Attributes
    ----------
    knowledge : AttackerKnowledge
        What the attacker knows about the model
    goal : AttackerGoal
        What the attacker aims to achieve
    norm : str
        Perturbation norm constraint ('linf', 'l2', 'l1')
    epsilon : float
        Perturbation budget
    query_budget : int, optional
        Maximum queries for black-box attacks
    """
    knowledge: AttackerKnowledge
    goal: AttackerGoal
    norm: str = 'linf'
    epsilon: float = 8/255
    query_budget: Optional[int] = None
    
    def __post_init__(self):
        if self.knowledge == AttackerKnowledge.BLACK_BOX:
            if self.query_budget is None:
                self.query_budget = 10000
    
    def __repr__(self):
        return (
            f"ThreatModel(\n"
            f"  knowledge={self.knowledge.value},\n"
            f"  goal={self.goal.value},\n"
            f"  norm={self.norm},\n"
            f"  epsilon={self.epsilon:.4f}"
            + (f",\n  query_budget={self.query_budget}" 
               if self.query_budget else "") +
            f"\n)"
        )

# Standard benchmark threat models
STANDARD_CIFAR10 = ThreatModel(
    knowledge=AttackerKnowledge.WHITE_BOX,
    goal=AttackerGoal.UNTARGETED,
    norm='linf',
    epsilon=8/255
)

REALISTIC_DEPLOYMENT = ThreatModel(
    knowledge=AttackerKnowledge.BLACK_BOX,
    goal=AttackerGoal.TARGETED,
    norm='l2',
    epsilon=0.5,
    query_budget=1000
)

FINANCIAL_API = ThreatModel(
    knowledge=AttackerKnowledge.GRAY_BOX,
    goal=AttackerGoal.UNTARGETED,
    norm='linf',
    epsilon=0.05,  # Feature-space perturbation budget
    query_budget=500
)
```

## Choosing the Right Threat Model

### For Security Evaluation

- Use white-box attacks as the conservative baseline
- Consider multiple norms ($\ell_\infty$, $\ell_2$, $\ell_1$)
- Test both targeted and untargeted scenarios

### For Realistic Assessment

- Use black-box or gray-box models
- Enforce realistic query budgets
- Consider transfer attacks from public models

### For Financial Applications

| Application | Recommended Threat Model | Rationale |
|-------------|-------------------------|-----------|
| Real-time trading | Black-box, query-limited | Attacker observes outputs only |
| Batch predictions | White-box (assume model leaked) | Conservative for offline systems |
| Customer-facing API | Gray-box, score-based | Attacker can probe the endpoint |
| Internal risk models | White-box, targeted | Insider threat scenario |
| Fraud detection | Black-box, decision-based | Fraudsters see accept/reject only |

## Summary

| Concept | Key Point |
|---------|-----------|
| White-box | Full model access; strongest attacks; worst-case evaluation |
| Black-box | Query access only; realistic but weaker |
| Gray-box | Partial information; most realistic for deployment |
| Untargeted | Cause any misclassification |
| Targeted | Force a specific misclassification |
| Confidence | Manipulate prediction confidence |

Understanding threat models is the prerequisite for both attacking and defending neural networks. The appropriate model depends on the deployment context and security requirements.

## References

1. Biggio, B., & Roli, F. (2018). "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." Pattern Recognition.
2. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.
3. Gilmer, J., et al. (2018). "Motivating the Rules of the Game for Adversarial Example Research." arXiv preprint arXiv:1807.06732.
