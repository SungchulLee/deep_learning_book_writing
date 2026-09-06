# Concept Activation Vectors (CAV)
## Introduction

**Concept Activation Vectors (CAVs)** provide a way to test whether a neural network has learned a specific human-understandable concept. Rather than explaining predictions in terms of raw input features (pixels, token embeddings), CAVs explain in terms of high-level concepts like "striped," "furry," "high volatility," or "mean-reverting."

Introduced by Kim et al. (2018), CAVs bridge the gap between what neural networks compute (activations in high-dimensional spaces) and what humans understand (concepts).

## Mathematical Foundation

### Concept Direction in Activation Space

Given a trained neural network, consider the activations at layer $l$ for a set of inputs. A CAV for concept $k$ at layer $l$ is a vector $v_l^k$ in the activation space that points in the direction of the concept.

To find this direction:

1. Collect a set of **positive examples** $P_k$ that exhibit concept $k$
2. Collect a set of **negative examples** $N_k$ (random or concept-absent)
3. Train a linear classifier on the activations $h_l(x)$ at layer $l$:

$$
v_l^k = \arg\min_v \sum_{x \in P_k} \ell(\sigma(v^\top h_l(x)), 1) + \sum_{x \in N_k} \ell(\sigma(v^\top h_l(x)), 0)
$$

The normal vector to the decision boundary is the CAV $v_l^k$.

### Conceptual Sensitivity

The **conceptual sensitivity** of class $c$ to concept $k$ at layer $l$ is:

$$
S_{c,k,l}(x) = \nabla_{h_l(x)} f_c(x) \cdot v_l^k
$$

This measures how much the class score changes when moving the activation in the concept direction.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

class ConceptActivationVector:
    """Compute and use Concept Activation Vectors."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activation)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def get_activations(self, inputs: torch.Tensor) -> np.ndarray:
        """Extract activations at target layer."""
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)
        
        act = self.activations
        if act.dim() > 2:
            act = act.mean(dim=tuple(range(2, act.dim())))
        return act.cpu().numpy()
    
    def train_cav(
        self,
        concept_examples: torch.Tensor,
        random_examples: torch.Tensor
    ) -> np.ndarray:
        """
        Train a CAV by fitting a linear classifier.
        
        Returns:
            cav_vector: Normal to the decision boundary
        """
        pos_act = self.get_activations(concept_examples)
        neg_act = self.get_activations(random_examples)
        
        X = np.vstack([pos_act, neg_act])
        y = np.array([1] * len(pos_act) + [0] * len(neg_act))
        
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        
        cav = clf.coef_[0]
        cav = cav / np.linalg.norm(cav)
        
        accuracy = clf.score(X, y)
        print(f"CAV classifier accuracy: {accuracy:.3f}")
        
        return cav
    
    def conceptual_sensitivity(
        self,
        input_tensor: torch.Tensor,
        cav: np.ndarray,
        target_class: int
    ) -> float:
        """Compute sensitivity of target class to concept direction."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(False)
        
        self.model(input_tensor)
        act = self.activations
        act.requires_grad_(True)
        
        output = self.model.fc(act.mean(dim=(2, 3)) if act.dim() > 2 else act)
        output[0, target_class].backward()
        
        grad = act.grad
        if grad.dim() > 2:
            grad = grad.mean(dim=tuple(range(2, grad.dim())))
        
        grad_np = grad[0].cpu().numpy()
        cav_tensor = cav
        
        sensitivity = np.dot(grad_np, cav_tensor)
        return sensitivity
```

## Applications in Quantitative Finance

CAVs can test whether financial models have learned meaningful economic concepts:

| Concept | Positive Examples | Use Case |
|---------|-------------------|----------|
| "High volatility regime" | VIX > 25 periods | Risk model validation |
| "Mean reversion" | Assets returning to moving average | Strategy interpretability |
| "Momentum" | Assets with strong recent returns | Factor model analysis |
| "Credit stress" | Spread widening episodes | Credit risk model |

## Summary

CAVs provide concept-level explanations by finding directions in activation space that correspond to human-understandable concepts. They bridge the gap between neural network internals and domain expertise.

## References

1. Kim, B., et al. (2018). "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors." *ICML*.

2. Ghorbani, A., et al. (2019). "Towards Automatic Concept-based Explanations." *NeurIPS*.

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
