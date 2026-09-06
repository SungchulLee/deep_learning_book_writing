# Evaluation and Benchmarking
## Introduction

Proper evaluation of adversarial robustness is critical—many "robust" defenses have been broken due to improper evaluation. This section covers best practices, common pitfalls, and standardized evaluation protocols including **AutoAttack** and **RobustBench**.

## The Problem: Gradient Masking

### What is Gradient Masking?

**Gradient masking** occurs when a defense hides or obfuscates gradients without providing true robustness. The model appears robust against gradient-based attacks but remains vulnerable to other attacks.

### Types of Gradient Masking

1. **Shattered Gradients**: Non-differentiable operations (JPEG compression, quantization)
2. **Stochastic Gradients**: Randomized defenses that create noisy gradients
3. **Vanishing/Exploding Gradients**: Numerical issues that prevent effective optimization
4. **Obfuscated Gradients**: Defenses specifically designed to confuse gradient-based attacks

### How to Detect Gradient Masking

**Red flags:**

1. **One-step attacks outperform iterative attacks**: FGSM > PGD is suspicious
2. **Black-box attacks outperform white-box**: Transfer attacks > direct attacks
3. **Unbounded attacks fail**: Increasing $\varepsilon$ doesn't increase success
4. **Random noise competitive with attacks**: Gradients not providing useful signal

### Example: Detecting Gradient Masking

```python
def check_gradient_masking(model, x, y, epsilon=8/255):
    """
    Check for signs of gradient masking.
    
    Returns dict with diagnostic results.
    """
    results = {}
    
    # Test 1: FGSM vs PGD
    fgsm = FGSM(model, epsilon=epsilon)
    pgd = PGD(model, epsilon=epsilon, num_iter=40)
    
    x_fgsm = fgsm.generate(x, y)
    x_pgd = pgd.generate(x, y)
    
    fgsm_success = evaluate_attack_success(model, x_fgsm, y)
    pgd_success = evaluate_attack_success(model, x_pgd, y)
    
    results['fgsm_success'] = fgsm_success
    results['pgd_success'] = pgd_success
    results['fgsm_stronger'] = fgsm_success > pgd_success + 0.05  # Suspicious
    
    # Test 2: Increasing epsilon
    success_by_eps = []
    for eps in [0.01, 0.02, 0.04, 0.08, 0.16]:
        attack = PGD(model, epsilon=eps, num_iter=20)
        x_adv = attack.generate(x, y)
        success = evaluate_attack_success(model, x_adv, y)
        success_by_eps.append(success)
    
    results['success_increases'] = all(s1 <= s2 + 0.02 
                                        for s1, s2 in zip(success_by_eps[:-1], success_by_eps[1:]))
    
    # Test 3: Random noise comparison
    noise = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_random = torch.clamp(x + noise, 0, 1)
    random_success = evaluate_attack_success(model, x_random, y)
    
    results['random_success'] = random_success
    results['random_competitive'] = random_success > 0.5 * pgd_success  # Suspicious
    
    # Summary
    results['gradient_masking_suspected'] = (
        results['fgsm_stronger'] or 
        not results['success_increases'] or 
        results['random_competitive']
    )
    
    return results
```

## AutoAttack

### Overview

**AutoAttack** (Croce & Hein, 2020) is a parameter-free, ensemble attack designed for reliable robustness evaluation. It combines multiple complementary attacks to minimize false robustness claims.

### Components

AutoAttack consists of four attacks:

1. **APGD-CE**: Auto-PGD with cross-entropy loss
2. **APGD-DLR**: Auto-PGD with Difference of Logits Ratio loss
3. **FAB**: Fast Adaptive Boundary attack
4. **Square**: Black-box score-based attack

### Auto-PGD (APGD)

APGD improves upon standard PGD with:

**Adaptive step size:**

$$
\alpha^{(t+1)} = \begin{cases}
\alpha^{(t)} / 2 & \text{if loss hasn't improved in } w \text{ steps} \\
\alpha^{(t)} & \text{otherwise}
\end{cases}
$$

**Momentum:**

$$
\mathbf{z}^{(t+1)} = \rho \cdot \mathbf{z}^{(t)} + \nabla_\mathbf{x} \mathcal{L}
$$

**Checkpoint restarts:** Reset to best point when step size decreases.

### DLR Loss

The **Difference of Logits Ratio** loss is more effective than cross-entropy for attacks:

$$
\mathcal{L}_{\text{DLR}} = -\frac{z_y - \max_{i \neq y} z_i}{z_{\pi_1} - z_{\pi_3}}
$$

where $\pi$ is the sorted order of logits.

### Using AutoAttack

```python
# Install: pip install autoattack
from autoattack import AutoAttack

# Standard evaluation
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
x_adv = adversary.run_standard_evaluation(x, y, bs=100)

# Faster evaluation (2 attacks instead of 4)
adversary_fast = AutoAttack(model, norm='Linf', eps=8/255, version='plus')

# Custom attack set
adversary_custom = AutoAttack(model, norm='Linf', eps=8/255)
adversary_custom.attacks_to_run = ['apgd-ce', 'apgd-t']  # Only APGD attacks
```

### AutoAttack Results

Typical evaluation output:

```
initial accuracy: 87.20%
apgd-ce:         robust accuracy: 53.40% (- 33.80%)
apgd-t:          robust accuracy: 51.20% (- 2.20%)
fab-t:           robust accuracy: 50.80% (- 0.40%)
square:          robust accuracy: 50.60% (- 0.20%)
```

The final robust accuracy (50.60%) is the fraction surviving all attacks.

## RobustBench

### Overview

**RobustBench** is a standardized benchmark for adversarial robustness with:

- Curated leaderboards for CIFAR-10, CIFAR-100, ImageNet
- Pre-evaluated models with AutoAttack
- Easy model loading API

### Using RobustBench

```python
# Install: pip install robustbench
from robustbench import load_model
from robustbench.eval import benchmark

# Load pre-trained robust model
model = load_model(
    model_name='Carmon2019Unlabeled',  # Model name from leaderboard
    dataset='cifar10',
    threat_model='Linf'
)

# Benchmark a custom model
clean_acc, robust_acc = benchmark(
    model,
    dataset='cifar10',
    threat_model='Linf',
    eps=8/255
)

print(f"Clean: {clean_acc:.2%}, Robust: {robust_acc:.2%}")
```

### Leaderboard (CIFAR-10, L∞, ε=8/255)

As of 2024:

| Rank | Model | Clean Acc | Robust Acc |
|------|-------|-----------|------------|
| 1 | Wang2023Better | 93.25% | 70.69% |
| 2 | Cui2023... | 92.16% | 67.73% |
| 3 | Peng2023... | 93.27% | 67.31% |
| ... | ... | ... | ... |
| Baseline | Madry (2018) | 87.14% | 44.04% |

## Comprehensive Evaluation Protocol

### Step 1: Basic Checks

```python
def basic_evaluation(model, test_loader, epsilon=8/255):
    """Basic evaluation with multiple attacks."""
    results = {}
    
    # Clean accuracy
    results['clean'] = compute_accuracy(model, test_loader)
    
    # FGSM (fast, weak)
    fgsm = FGSM(model, epsilon=epsilon)
    results['fgsm'] = evaluate_robust_accuracy(model, test_loader, fgsm)
    
    # PGD-20 (standard)
    pgd20 = PGD(model, epsilon=epsilon, num_iter=20)
    results['pgd20'] = evaluate_robust_accuracy(model, test_loader, pgd20)
    
    # PGD-100 (strong)
    pgd100 = PGD(model, epsilon=epsilon, num_iter=100)
    results['pgd100'] = evaluate_robust_accuracy(model, test_loader, pgd100)
    
    return results
```

### Step 2: Gradient Masking Check

```python
def gradient_masking_check(model, x, y, epsilon=8/255):
    """Check for gradient masking indicators."""
    # Already implemented above
    return check_gradient_masking(model, x, y, epsilon)
```

### Step 3: AutoAttack Evaluation

```python
def autoattack_evaluation(model, x, y, epsilon=8/255):
    """Standard AutoAttack evaluation."""
    from autoattack import AutoAttack
    
    adversary = AutoAttack(model, norm='Linf', eps=epsilon)
    x_adv = adversary.run_standard_evaluation(x, y, bs=100)
    
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1)
        robust_acc = (pred == y).float().mean().item()
    
    return robust_acc
```

### Step 4: Multiple Threat Models

```python
def multi_threat_evaluation(model, test_loader):
    """Evaluate across multiple threat models."""
    results = {}
    
    # L∞ threat model
    for eps in [4/255, 8/255, 16/255]:
        pgd = PGD(model, epsilon=eps, norm='linf')
        results[f'linf_eps={eps:.4f}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    # L2 threat model
    for eps in [0.25, 0.5, 1.0]:
        pgd = PGD(model, epsilon=eps, norm='l2')
        results[f'l2_eps={eps}'] = evaluate_robust_accuracy(model, test_loader, pgd)
    
    return results
```

## Common Evaluation Mistakes

### 1. Weak Attacks Only

❌ **Wrong:** Only testing with FGSM
✅ **Correct:** Use PGD-100+ or AutoAttack

### 2. Fixed Hyperparameters

❌ **Wrong:** Using default PGD settings for all defenses
✅ **Correct:** Tune attack hyperparameters against defense

### 3. Ignoring Black-Box

❌ **Wrong:** Only white-box evaluation
✅ **Correct:** Include transfer and query-based attacks

### 4. Small Test Set

❌ **Wrong:** Evaluating on 100 examples
✅ **Correct:** Full test set (10,000 for CIFAR-10)

### 5. Adaptive Attack Missing

❌ **Wrong:** Not adapting attacks to defense mechanism
✅ **Correct:** Design attacks that account for defense properties

## Best Practices Summary

### For Defenders

1. **Use AutoAttack**: Standard, reliable evaluation
2. **Check for gradient masking**: Run diagnostic tests
3. **Multiple threat models**: Test L∞, L2, varying ε
4. **Compare to baselines**: Check against RobustBench models
5. **Report confidence intervals**: Especially for small test sets

### For Attackers

1. **Adaptive attacks**: Design attacks for specific defenses
2. **Multiple restarts**: Use random initialization
3. **Step size tuning**: Adapt to loss landscape
4. **Loss function selection**: CE, DLR, margin loss
5. **Gradient approximation**: For non-differentiable defenses

## Complete Evaluation Pipeline

```python
def full_robustness_evaluation(
    model,
    test_loader,
    epsilon=8/255,
    device='cuda'
):
    """
    Complete robustness evaluation pipeline.
    """
    model.eval()
    model.to(device)
    
    results = {
        'epsilon': epsilon,
        'model': str(model.__class__.__name__)
    }
    
    # Get test data
    x_test, y_test = [], []
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    x_test = torch.cat(x_test).to(device)
    y_test = torch.cat(y_test).to(device)
    
    print("=" * 60)
    print("ROBUSTNESS EVALUATION")
    print("=" * 60)
    
    # 1. Clean accuracy
    with torch.no_grad():
        clean_pred = model(x_test).argmax(dim=1)
        clean_acc = (clean_pred == y_test).float().mean().item()
    results['clean_accuracy'] = clean_acc
    print(f"Clean Accuracy: {clean_acc:.2%}")
    
    # 2. Gradient masking check
    print("\nGradient Masking Check...")
    gm_results = check_gradient_masking(model, x_test[:1000], y_test[:1000], epsilon)
    results['gradient_masking'] = gm_results
    if gm_results['gradient_masking_suspected']:
        print("⚠️  WARNING: Gradient masking suspected!")
    else:
        print("✓ No gradient masking detected")
    
    # 3. PGD evaluation
    print("\nPGD Evaluation...")
    for num_iter in [20, 100]:
        pgd = PGD(model, epsilon=epsilon, num_iter=num_iter, device=device)
        x_adv = pgd.generate(x_test, y_test)
        with torch.no_grad():
            adv_pred = model(x_adv).argmax(dim=1)
            robust_acc = (adv_pred == y_test).float().mean().item()
        results[f'pgd{num_iter}'] = robust_acc
        print(f"  PGD-{num_iter}: {robust_acc:.2%}")
    
    # 4. AutoAttack (subset for speed)
    print("\nAutoAttack Evaluation (1000 samples)...")
    try:
        from autoattack import AutoAttack
        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
        x_adv_aa = adversary.run_standard_evaluation(
            x_test[:1000], y_test[:1000], bs=100
        )
        with torch.no_grad():
            aa_pred = model(x_adv_aa).argmax(dim=1)
            aa_acc = (aa_pred == y_test[:1000]).float().mean().item()
        results['autoattack'] = aa_acc
        print(f"  AutoAttack: {aa_acc:.2%}")
    except ImportError:
        print("  AutoAttack not installed, skipping...")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Clean Accuracy:     {results['clean_accuracy']:.2%}")
    print(f"PGD-20 Accuracy:    {results.get('pgd20', 'N/A'):.2%}")
    print(f"PGD-100 Accuracy:   {results.get('pgd100', 'N/A'):.2%}")
    print(f"AutoAttack:         {results.get('autoattack', 'N/A'):.2%}")
    print("=" * 60)
    
    return results
```

## References

1. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-Free Attacks." ICML.
2. Athalye, A., Carlini, N., & Wagner, D. (2018). "Obfuscated Gradients Give a False Sense of Security." ICML.
3. Carlini, N., et al. (2019). "On Evaluating Adversarial Robustness." arXiv.
4. RobustBench: https://robustbench.github.io/

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
