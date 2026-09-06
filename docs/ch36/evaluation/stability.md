# Stability Evaluation
## Introduction

An explanation method is **stable** if similar inputs produce similar explanations. If adding imperceptible noise completely changes the explanation, practitioners cannot trust it. Stability is essential for regulated environments where explanations must be reproducible.

## Metrics

### Relative Input Stability (RIS)

$$
\text{RIS}(x, x') = \frac{\|E(x) - E(x')\|_2}{\|E(x)\|_2 \cdot \|x - x'\|_2}
$$

Lower RIS = more stable.

### Max-Sensitivity

$$
\text{MaxSens}(x) = \max_{\|\epsilon\| \leq r} \|E(x) - E(x + \epsilon)\|_2
$$

### Implementation

```python
import torch
import numpy as np

def compute_stability(
    explanation_fn, input_tensor, n_perturbations=50, noise_level=0.01
):
    """Measure explanation stability under input perturbations."""
    base_explanation = explanation_fn(input_tensor)
    base_norm = np.linalg.norm(base_explanation)
    
    relative_changes = []
    for _ in range(n_perturbations):
        noise = torch.randn_like(input_tensor) * noise_level
        perturbed_explanation = explanation_fn(input_tensor + noise)
        
        explanation_change = np.linalg.norm(base_explanation - perturbed_explanation)
        input_change = noise.norm().item()
        
        if base_norm > 0 and input_change > 0:
            relative_changes.append(explanation_change / (base_norm * input_change))
    
    return {
        'mean_ris': np.mean(relative_changes),
        'max_ris': np.max(relative_changes),
        'std_ris': np.std(relative_changes)
    }
```

## Method Stability Comparison

| Method | Typical Stability | Why |
|--------|------------------|-----|
| Vanilla Gradients | Low | Noisy by construction |
| SmoothGrad | High | Averaging reduces sensitivity |
| Integrated Gradients | Medium | Path-dependent, baseline matters |
| SHAP | Medium-High | Sampling introduces some variance |
| LIME | Low-Medium | Random sampling, kernel width sensitive |
| Grad-CAM | High | Spatial averaging smooths output |

## Summary

Stability evaluation ensures explanations are robust to minor input variations. Methods like SmoothGrad and Grad-CAM inherently improve stability through averaging.

## References

1. Alvarez-Melis, D., & Jaakkola, T. S. (2018). "On the Robustness of Interpretability Methods." *ICML Workshop*.
2. Yeh, C. K., et al. (2019). "On the (In)fidelity and Sensitivity of Explanations." *NeurIPS*.

## Exercises

**Exercise 1.**
Apply the interpretability method described in this section to a 2-layer neural network with ReLU activations classifying XOR inputs. Compute the explanation for the input $x = [1, 1]$.

??? success "Solution to Exercise 1"
    For a trained XOR network with weights $W_1, b_1, W_2, b_2$, the output is $f(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$. The explanation method produces attributions for each input feature. For $x = [1, 1]$ (class 0), both features contribute to the negative classification. The specific attribution values depend on the method: gradient-based methods compute $\partial f / \partial x_i$; perturbation-based methods measure output change when features are masked. The XOR problem demonstrates that linear explanation methods can mislead because the decision boundary is non-linear. $\square$

---

**Exercise 2.**
Prove or disprove that the explanation method in this section satisfies the completeness axiom: the sum of all feature attributions equals $f(x) - f(x_0)$ for some baseline $x_0$.

??? success "Solution to Exercise 2"
    The completeness axiom (also called efficiency in Shapley value theory) states that attributions sum to the difference between the model output at the input and at the baseline. Whether this method satisfies completeness depends on its formulation. Gradient methods do not satisfy completeness (gradients are local, not path-integrated). Integrated Gradients satisfies completeness by construction (fundamental theorem of calculus along the path). SHAP values satisfy efficiency by the Shapley axiom. Methods that violate completeness may over- or under-attribute, making the total attribution unreliable as a global explanation. $\square$

---

**Exercise 3.**
Design an experiment to evaluate the faithfulness of the explanations produced by this method. Use insertion and deletion curves to measure whether highlighted features are truly important to the model.

??? success "Solution to Exercise 3"
    Protocol: (1) Compute feature attributions for each test image. (2) Deletion: progressively mask features in order of decreasing attribution, recording the model confidence drop. Faithful explanations cause rapid confidence decrease. (3) Insertion: progressively reveal features in order of decreasing attribution from a blank baseline, recording confidence increase. Faithful explanations cause rapid confidence increase. (4) Compute AUC for both curves. (5) Compare against random ordering (baseline) and other methods. A faithful method should have low deletion AUC and high insertion AUC. Repeat over 1000+ test samples for statistical reliability. $\square$

---

**Exercise 4.**
Discuss how this interpretability method could be applied to a financial model predicting credit default. What regulatory requirements must the explanations satisfy?

??? success "Solution to Exercise 4"
    For credit models, regulations (ECOA, GDPR Article 22) require individualized explanations for adverse decisions. The method must produce: (1) the top factors contributing to the denial (adverse action reasons); (2) explanations that are consistent (similar applicants get similar explanations); (3) explanations that are actionable (the applicant understands what to change). The interpretability method from this section can identify feature importances, but must be validated for stability (small input changes should not drastically alter the explanation) and correctness (removing important features should change the prediction). Protected attributes must be handled carefully to avoid revealing proxy discrimination. $\square$
