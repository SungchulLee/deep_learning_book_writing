# Comprehensiveness Evaluation
## Introduction

**Comprehensiveness** measures whether an explanation captures all the important features, not just some. Its complement, **sufficiency**, tests whether the identified features alone reproduce the prediction.

## Metrics

### Comprehensiveness Score

Remove top-$k$ features and measure prediction change:

$$
\text{Comprehensiveness}(E, x) = f(x) - f(x_{\setminus E})
$$

Higher = the explanation captures features that matter.

### Sufficiency Score

Keep only top-$k$ features and measure prediction preservation:

$$
\text{Sufficiency}(E, x) = f(x) - f(x_E)
$$

Lower = the identified features are sufficient.

### Implementation

```python
import torch
import numpy as np

def comprehensiveness_sufficiency(
    model, input_tensor, attribution, target_class,
    k_values=[0.1, 0.2, 0.3, 0.5]
):
    """Compute comprehensiveness and sufficiency at various thresholds."""
    model.eval()
    
    with torch.no_grad():
        base_score = torch.softmax(model(input_tensor), dim=1)[0, target_class].item()
    
    attr_flat = attribution.flatten()
    sorted_idx = np.argsort(np.abs(attr_flat))[::-1]
    n_features = len(attr_flat)
    
    results = {}
    for k in k_values:
        n_top = int(k * n_features)
        top_indices = sorted_idx[:n_top]
        
        # Comprehensiveness: remove top-k
        removed = input_tensor.clone().flatten()
        removed[top_indices] = 0
        with torch.no_grad():
            removed_score = torch.softmax(model(removed.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
        
        # Sufficiency: keep only top-k
        kept = torch.zeros_like(input_tensor).flatten()
        kept[top_indices] = input_tensor.flatten()[top_indices]
        with torch.no_grad():
            kept_score = torch.softmax(model(kept.reshape(input_tensor.shape)), dim=1)[0, target_class].item()
        
        results[k] = {
            'comprehensiveness': base_score - removed_score,
            'sufficiency': base_score - kept_score
        }
    
    return results
```

## Interpretation

A good explanation should be both **comprehensive** (high comprehensiveness score) and **sufficient** (low sufficiency score). These metrics are complementary: comprehensiveness alone can be gamed by always selecting all features.

## Summary

Comprehensiveness and sufficiency quantify whether explanations capture all relevant features and whether those features alone reproduce the prediction.

## References

1. DeYoung, J., et al. (2020). "ERASER: A Benchmark to Evaluate Rationalized NLP Models." *ACL*.
2. Carton, S., Rathore, A., & Tan, C. (2020). "Evaluating and Characterizing Human Rationales." *EMNLP*.

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
