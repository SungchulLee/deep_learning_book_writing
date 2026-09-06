# AutoAttack
## Introduction

**AutoAttack** (Croce & Hein, 2020) is a parameter-free, ensemble-based attack designed for reliable adversarial robustness evaluation. It combines multiple complementary attack strategies to minimize the risk of false robustness claims due to gradient masking or suboptimal attack hyperparameters. AutoAttack has become the de facto standard for reporting robust accuracy in the adversarial robustness literature.

## Motivation

Many defenses that initially appeared robust were later broken when evaluated with stronger or more carefully tuned attacks. Common failure modes include:

- **Gradient masking**: Defenses that hide gradients rather than providing true robustness
- **Suboptimal hyperparameters**: PGD with too few steps or wrong step size
- **Single loss function**: Cross-entropy may not be the most effective attack objective for all defenses

AutoAttack addresses these issues by requiring **no hyperparameter tuning** and using an ensemble of diverse attacks.

## Components

AutoAttack combines four complementary attacks, run sequentially:

### 1. APGD-CE (Auto-PGD with Cross-Entropy)

An improved version of PGD with **adaptive step size** scheduling:

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

Key improvements over standard PGD:

- **Momentum**: $\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}$
- **Checkpoint restarts**: Resets to the best point found when step size decreases
- **No step size tuning**: Adapts automatically based on loss trajectory

### 2. APGD-DLR (Auto-PGD with Difference of Logits Ratio)

Uses the **Difference of Logits Ratio** loss instead of cross-entropy:

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

where $z_{\pi_1} \geq z_{\pi_2} \geq \ldots$ are sorted logits. The DLR loss:

- Is scale-invariant (denominator normalizes by logit range)
- Avoids saturation issues of cross-entropy near confident predictions
- Is more effective against defenses that maintain high confidence

### 3. FAB (Fast Adaptive Boundary)

A **minimum-norm attack** that finds small perturbations by iteratively projecting onto the decision boundary:

- Projects the current point onto the linearized decision boundary
- Searches across multiple target classes
- Finds perturbations that are often smaller than PGD's fixed-budget perturbations

FAB is especially useful when the defense has non-smooth loss landscapes.

### 4. Square Attack (Black-Box)

A **score-based black-box** attack using random square-shaped perturbations:

- Requires no gradients (catches gradient-masking defenses)
- Uses a localized random search strategy
- Iteratively updates square-shaped patches to maximize loss

The inclusion of a black-box attack is critical: if a defense masks gradients, APGD may fail but Square Attack can still succeed.

## Evaluation Protocol

AutoAttack runs each attack sequentially. After each attack, only the examples that survived (were not yet misclassified) are passed to the next attack:

```
Full test set → APGD-CE → survivors → APGD-DLR → survivors → FAB → survivors → Square → final robust accuracy
```

The final robust accuracy is the fraction surviving all four attacks.

### Typical Output

```
initial accuracy:  87.20%
apgd-ce:           robust accuracy: 53.40% (- 33.80%)
apgd-t (DLR):     robust accuracy: 51.20% (- 2.20%)
fab-t:             robust accuracy: 50.80% (- 0.40%)
square:            robust accuracy: 50.60% (- 0.20%)
```

## PyTorch Usage

### Standard Evaluation

```python
# Install: pip install autoattack
from autoattack import AutoAttack

# Standard evaluation (4 attacks)
adversary = AutoAttack(
    model, 
    norm='Linf', 
    eps=8/255, 
    version='standard'
)
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)

# Fast evaluation (APGD-CE + APGD-DLR only)
adversary_fast = AutoAttack(
    model, 
    norm='Linf', 
    eps=8/255, 
    version='plus'
)

# Custom attack subset
adversary_custom = AutoAttack(model, norm='Linf', eps=8/255)
adversary_custom.attacks_to_run = ['apgd-ce', 'apgd-t']
```

### Integration with Evaluation Pipeline

```python
import torch
from typing import Dict

def autoattack_evaluation(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 8/255,
    norm: str = 'Linf',
    version: str = 'standard',
    batch_size: int = 100
) -> Dict[str, float]:
    """
    Run AutoAttack evaluation.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    x, y : torch.Tensor
        Test inputs and labels
    epsilon : float
        Perturbation budget
    norm : str
        Norm type ('Linf' or 'L2')
    version : str
        'standard' (4 attacks) or 'plus' (2 attacks)
    
    Returns
    -------
    results : dict with clean_accuracy and robust_accuracy
    """
    from autoattack import AutoAttack
    
    model.eval()
    
    # Clean accuracy
    with torch.no_grad():
        device = next(model.parameters()).device
        clean_pred = model(x.to(device)).argmax(dim=1)
        clean_acc = (clean_pred == y.to(device)).float().mean().item()
    
    # AutoAttack
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version)
    x_adv = adversary.run_standard_evaluation(x, y, bs=batch_size)
    
    with torch.no_grad():
        adv_pred = model(x_adv.to(device)).argmax(dim=1)
        robust_acc = (adv_pred == y.to(device)).float().mean().item()
    
    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'attack_success_rate': 1 - robust_acc,
        'norm': norm,
        'epsilon': epsilon
    }
```

## When to Use AutoAttack

### Always Use For

- **Reporting robust accuracy** in publications or reports
- **Comparing defenses** on standardized benchmarks
- **Final evaluation** of deployed models

### Consider Alternatives For

- **Rapid prototyping**: PGD-20 is much faster for iterative development
- **Training inner loop**: Too slow for adversarial training
- **Custom threat models**: AutoAttack assumes standard $\ell_p$ balls

## Comparison with Individual Attacks

| Aspect | PGD-100 | C&W | AutoAttack |
|--------|---------|-----|------------|
| Parameter-free | No | No | **Yes** |
| Catches gradient masking | No | Partially | **Yes** |
| Multiple loss functions | No | No | **Yes** |
| Black-box component | No | No | **Yes** |
| Speed | Fast | Slow | Moderate |
| Reliability | Good | Good | **Best** |

## Summary

| Feature | Detail |
|---------|--------|
| **Purpose** | Reliable, standardized robustness evaluation |
| **Components** | APGD-CE, APGD-DLR, FAB, Square Attack |
| **Key property** | Parameter-free ensemble |
| **Use case** | Final evaluation and reporting |
| **Advantage** | Catches gradient masking, no tuning needed |

AutoAttack has become the gold standard for adversarial robustness evaluation precisely because it eliminates the ambiguity of hand-tuned attack parameters and provides comprehensive coverage of different attack strategies.

## References

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Croce, F., & Hein, M. (2020). "Minimally Distorted Adversarial Examples with a Fast Adaptive Boundary Attack." ICML.
3. Andriushchenko, M., et al. (2020). "Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search." ECCV.

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
