# Statistical Detection of Adversarial Examples
## Introduction

Rather than making models robust to adversarial perturbations, an alternative approach is to **detect** adversarial inputs before they reach the classifier. Statistical detection methods analyze properties of inputs and model internals to distinguish clean from adversarial examples.

## Detection Approaches

### Feature Distribution Analysis

Adversarial examples often produce **unusual activation patterns** in intermediate network layers. Detection methods compare the activation statistics of a test input against the distribution of clean data.

**Mahalanobis Distance Detector** (Lee et al., 2018):

For each class $c$ and layer $\ell$, fit a Gaussian to clean activations:

$$
(\boldsymbol{\mu}_c^\ell, \boldsymbol{\Sigma}^\ell) = \text{fit}(\{h^\ell(\mathbf{x}) : y = c\})
$$

The detection score combines Mahalanobis distances across layers:

$$
M(\mathbf{x}) = \sum_\ell \max_c \left[ -(h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell)^\top (\boldsymbol{\Sigma}^\ell)^{-1} (h^\ell(\mathbf{x}) - \boldsymbol{\mu}_c^\ell) \right]
$$

Adversarial examples tend to have higher Mahalanobis distances (lower scores).

### Prediction Consistency

Check whether the model's predictions are **consistent** under input transformations that should not change the true class:

```python
import torch
import torch.nn as nn
from typing import Dict

class ConsistencyDetector:
    """
    Detect adversarial examples via prediction consistency
    under random transformations.
    
    Clean inputs maintain consistent predictions under
    small transformations; adversarial examples do not.
    """
    
    def __init__(
        self, model: nn.Module, num_transforms: int = 20,
        noise_std: float = 0.05, threshold: float = 0.7
    ):
        self.model = model
        self.num_transforms = num_transforms
        self.noise_std = noise_std
        self.threshold = threshold
        self.model.eval()
    
    def detect(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect adversarial examples.
        
        Returns consistency scores and binary detection decisions.
        """
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            # Original prediction
            base_pred = self.model(x).argmax(dim=1)
            
            # Predictions under random noise
            consistent = torch.zeros(len(x), device=device)
            for _ in range(self.num_transforms):
                noise = torch.randn_like(x) * self.noise_std
                noisy_pred = self.model(x + noise).argmax(dim=1)
                consistent += (noisy_pred == base_pred).float()
            
            consistency_score = consistent / self.num_transforms
        
        return {
            'consistency_score': consistency_score,
            'is_adversarial': consistency_score < self.threshold,
            'base_prediction': base_pred
        }
```

### Logit Analysis

Adversarial examples often produce logit distributions that differ from clean inputs:

- **Higher entropy**: Less confident predictions (for some attacks)
- **Unusual logit gaps**: Abnormal margins between top classes
- **Different softmax distributions**: Detectable via statistical tests

## Limitations

Statistical detectors face a fundamental challenge: they can themselves be attacked. An **adaptive adversary** who knows the detection mechanism can craft adversarial examples that also fool the detector. This creates an arms race that favors the attacker in the white-box setting.

## Summary

Statistical detection provides a complementary layer of defense, particularly effective against non-adaptive attacks. However, it should not be relied upon as the sole defense mechanism, especially against sophisticated adversaries.

## References

1. Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS.
2. Ma, X., et al. (2018). "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality." ICLR.
3. Carlini, N., & Wagner, D. (2017). "Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods." ACM Workshop on AI Security.

## Exercises

**Exercise 1.**
For a linear classifier $f(x) = w^T x + b$, compute the minimal $\ell_\infty$ perturbation needed to change the predicted class. Relate this to the robustness of neural networks.

??? success "Solution to Exercise 1"
    For a linear classifier, the distance to the decision boundary under $\ell_\infty$ norm is $\frac{|w^T x + b|}{\|w\|_1}$. The minimal perturbation is $\delta^* = \frac{|w^T x + b|}{\|w\|_1} \cdot \text{sign}(w)$. For neural networks, the local linear approximation $f(x + \delta) \approx f(x) + \nabla_x f \cdot \delta$ explains why FGSM (which uses the sign of the gradient) is effective. The vulnerability of high-dimensional models comes from the fact that $\|w\|_1$ grows with dimension while $|w^T x + b|$ does not necessarily, making the robustness margin shrink. $\square$

---

**Exercise 2.**
Implement the attack or defense described in this section for a ResNet-18 model on CIFAR-10. Report clean accuracy and robust accuracy under PGD-20 attack with $\epsilon = 8/255$.

??? success "Solution to Exercise 2"
    A standard ResNet-18 achieves $\sim$93% clean accuracy but $\sim$0% robust accuracy under PGD-20 ($\epsilon = 8/255$, step size $2/255$). After applying the method from this section, typical results depend on the specific technique: adversarial training achieves $\sim$83% clean / $\sim$50% robust; certified defenses achieve lower but provable bounds. The accuracy-robustness trade-off is fundamental: improving robustness typically costs 5--15% clean accuracy. Report results averaged over 3 random seeds with standard errors. $\square$

---

**Exercise 3.**
Prove that no defense can simultaneously achieve high accuracy on clean data and high robustness against $\ell_\infty$ perturbations without increasing model capacity, under the assumption that the data distribution has overlapping class-conditional supports within the perturbation ball.

??? success "Solution to Exercise 3"
    If two classes have support overlap within distance $\epsilon$ (i.e., $\exists x_1 \in \text{class 1}, x_2 \in \text{class 2}$ with $\|x_1 - x_2\|_\infty \leq 2\epsilon$), then any classifier robust at both $x_1$ and $x_2$ must misclassify at least one of them (the perturbation balls overlap). This is the fundamental accuracy-robustness trade-off: the fraction of overlapping support determines the unavoidable accuracy loss. For natural image distributions, significant overlap exists at $\epsilon = 8/255$, explaining the observed 10--15% accuracy drop. Increased model capacity (wider networks) can better approximate the complex robust decision boundary, partially mitigating the trade-off. $\square$

---

**Exercise 4.**
Discuss how adversarial robustness concerns manifest in a financial machine learning system (e.g., fraud detection or trading signal generation). How does the threat model differ from computer vision?

??? success "Solution to Exercise 4"
    In finance, adversaries are strategic agents (fraudsters, market manipulators) who actively adapt to detection systems. Key differences from vision: (1) the perturbation space is constrained by what is economically feasible (a fraudster cannot change their entire transaction history); (2) attacks are sequential and adaptive (the adversary observes the system's response and adjusts); (3) the cost of false positives and false negatives is asymmetric (blocking a legitimate transaction vs. missing fraud); (4) $\ell_p$ norms are not meaningful -- domain-specific perturbation models are needed. Defenses must be robust to adaptive adversaries, which rules out many detection-based approaches that can be evaded once the detection criterion is known. $\square$
