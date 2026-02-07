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
