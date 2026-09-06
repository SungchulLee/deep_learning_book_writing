# Feature Inversion
## Introduction

**Feature inversion** reconstructs inputs from intermediate representations, revealing what information the network preserves at each layer. This complementary approach to attribution shows not just *which* features matter, but *what the model actually sees* at different stages of processing.

## Mathematical Foundation

Given a feature representation $\Phi_l(\mathbf{x})$ at layer $l$, feature inversion finds:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\Phi_l(\mathbf{x}) - \Phi_l(\mathbf{x}_0)\|^2 + \lambda R(\mathbf{x})
$$

where $R(\mathbf{x})$ is a regularizer (total variation, $L^2$ norm) that encourages natural-looking images.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureInversion:
    """Reconstruct input from intermediate features."""
    
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.target_features = None
        
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'current_features', o)
        )
    
    def total_variation(self, x):
        """Total variation regularizer for spatial smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def invert(
        self, target_input, n_steps=500, lr=0.05,
        tv_weight=1e-3, l2_weight=1e-5
    ):
        """Reconstruct input from layer features."""
        self.model.eval()
        
        with torch.no_grad():
            self.model(target_input.to(self.device))
            target_features = self.current_features.clone()
        
        # Start from noise
        x = torch.randn_like(target_input, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            self.model(x)
            
            # Feature matching loss
            feat_loss = F.mse_loss(self.current_features, target_features)
            
            # Regularization
            tv_loss = tv_weight * self.total_variation(x)
            l2_loss = l2_weight * x.pow(2).mean()
            
            loss = feat_loss + tv_loss + l2_loss
            loss.backward()
            optimizer.step()
        
        return x.detach()
```

## Interpretation

Feature inversion reveals a fundamental insight: **early layers preserve spatial detail but lose semantic content, while later layers preserve semantic content but lose spatial detail.** This progressive abstraction is why Grad-CAM (targeting late layers) produces coarse but class-discriminative heatmaps.

## Summary

Feature inversion complements attribution methods by showing what information is preserved versus discarded at each network layer, providing a holistic understanding of the model's internal representations.

## References

1. Mahendran, A., & Vedaldi, A. (2015). "Understanding Deep Image Representations by Inverting Them." *CVPR*.

2. Dosovitskiy, A., & Brox, T. (2016). "Inverting Visual Representations with Convolutional Networks." *CVPR*.\n

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
