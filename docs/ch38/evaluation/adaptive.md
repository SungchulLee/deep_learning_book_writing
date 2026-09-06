# Adaptive Attacks and Gradient Masking
## Introduction

The history of adversarial robustness is littered with defenses that initially appeared robust but were later broken by **adaptive attacks**—attacks specifically designed to circumvent the defense mechanism. Understanding gradient masking and how to design adaptive attacks is critical for honest robustness evaluation.

## Gradient Masking

### What is Gradient Masking?

**Gradient masking** occurs when a defense obscures or corrupts the gradient signal without providing true robustness. The model appears robust against gradient-based attacks because the gradients are uninformative, not because the model is genuinely robust.

### Types of Gradient Masking

1. **Shattered gradients**: Non-differentiable operations (JPEG compression, quantization, input thresholding) break the computation graph
2. **Stochastic gradients**: Randomized defenses (dropout at test time, random resizing) create noisy, unreliable gradients
3. **Vanishing/exploding gradients**: Deep defensive layers or unusual architectures cause numerical gradient issues
4. **Obfuscated gradients**: Defenses specifically designed to confuse gradient computation

### Detection Heuristics

Several red flags indicate gradient masking:

1. **Single-step attacks outperform iterative**: If FGSM achieves higher attack success than PGD, gradients are unreliable
2. **Black-box attacks outperform white-box**: If transfer attacks work better than direct gradient attacks, the defense is masking gradients
3. **Unbounded attacks fail**: If increasing $\varepsilon$ doesn't increase attack success, the optimization is stuck
4. **Random noise is competitive**: If random perturbations are nearly as effective as gradient-based ones, gradients provide no useful signal

```python
import torch

def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    Diagnostic check for gradient masking.
    
    Returns dict with diagnostic indicators.
    """
    results = {}
    
    # Test 1: FGSM vs PGD
    from attacks import FGSM, PGD
    fgsm = FGSM(model, epsilon=epsilon)
    pgd = PGD(model, epsilon=epsilon, num_iter=40)
    
    x_fgsm = fgsm.generate(x, y)
    x_pgd = pgd.generate(x, y)
    
    fgsm_success = evaluate_success(model, x_fgsm, y)
    pgd_success = evaluate_success(model, x_pgd, y)
    
    results['fgsm_success'] = fgsm_success
    results['pgd_success'] = pgd_success
    results['fgsm_stronger'] = fgsm_success > pgd_success + 0.05
    
    # Test 2: Monotonic increase with epsilon
    success_curve = []
    for eps in [0.01, 0.02, 0.04, 0.08, 0.16]:
        attack = PGD(model, epsilon=eps, num_iter=20)
        x_adv = attack.generate(x, y)
        success_curve.append(evaluate_success(model, x_adv, y))
    
    results['monotonic'] = all(
        s1 <= s2 + 0.02 for s1, s2 in zip(success_curve[:-1], success_curve[1:])
    )
    
    # Test 3: Random noise comparison
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_random = torch.clamp(x + noise, 0, 1)
    random_success = evaluate_success(model, x_random, y)
    
    results['random_success'] = random_success
    results['random_competitive'] = random_success > 0.5 * pgd_success
    
    # Overall verdict
    results['gradient_masking_suspected'] = (
        results['fgsm_stronger'] or
        not results['monotonic'] or
        results['random_competitive']
    )
    
    return results
```

## Designing Adaptive Attacks

### Backward Pass Differentiable Approximation (BPDA)

For defenses with non-differentiable components $g$, replace $g$ with a differentiable approximation $\hat{g}$ during the backward pass:

$$
\text{Forward: } f(g(\mathbf{x})), \quad \text{Backward: } \nabla f(\hat{g}(\mathbf{x}))
$$

Common choices for $\hat{g}$: identity function (if $g$ is approximately identity-preserving), or a trained neural network approximation.

### Expectation over Transformation (EOT)

For stochastic defenses, compute gradients as expectations over the randomness:

$$
\nabla_\mathbf{x} \mathbb{E}_{t \sim \mathcal{T}} [\mathcal{L}(f(t(\mathbf{x})), y)] \approx \frac{1}{K} \sum_{k=1}^K \nabla_\mathbf{x} \mathcal{L}(f(t_k(\mathbf{x})), y)
$$

### Attack Recommendations by Defense Type

| Defense Type | Adaptive Attack Strategy |
|-------------|------------------------|
| Non-differentiable preprocessing | BPDA (identity or learned approximation) |
| Stochastic defense | EOT (average gradients over randomness) |
| Ensemble/voting | Attack individual members + majority vote |
| Detection + rejection | Joint attack on classifier + detector |
| Input transformation | Optimization through the transformation |

## Evaluation Checklist

For any new defense, verify robustness by checking:

- [ ] PGD-100+ with multiple random restarts
- [ ] AutoAttack (parameter-free ensemble)
- [ ] Gradient masking diagnostics (4 tests above)
- [ ] BPDA if defense has non-differentiable components
- [ ] EOT if defense is stochastic
- [ ] Transfer attacks from undefended surrogate models
- [ ] C&W attack with sufficient iterations
- [ ] Multiple norm threats ($\ell_\infty$, $\ell_2$)

## Summary

Gradient masking is the most common failure mode of adversarial defenses. Honest evaluation requires adaptive attacks specifically designed for the defense mechanism. AutoAttack helps but may not be sufficient for novel defense architectures—always complement with defense-specific adaptive attacks.

## References

1. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
2. Tramer, F., et al. (2020). "On Adaptive Attacks to Adversarial Example Defenses." NeurIPS.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv preprint arXiv:1902.06705.

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
