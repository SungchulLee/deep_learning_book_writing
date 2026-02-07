# DeepFool

## Introduction

**DeepFool** (Moosavi-Dezfooli et al., 2016) is a geometric adversarial attack that finds the **minimal perturbation** to move an input across the nearest decision boundary. Unlike fixed-budget attacks (FGSM, PGD) that use a predetermined $\varepsilon$, DeepFool iteratively computes the smallest perturbation needed for misclassification, providing insight into the model's local decision geometry.

## Mathematical Foundation

### Binary Classifier Case

Consider a binary affine classifier $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$. The decision boundary is the hyperplane $\{\mathbf{x} : f(\mathbf{x}) = 0\}$. The minimum perturbation to cross this boundary is the orthogonal projection onto the hyperplane:

$$
\boldsymbol{\delta}^* = -\frac{f(\mathbf{x})}{\|\mathbf{w}\|_2^2} \mathbf{w}
$$

with magnitude:

$$
\|\boldsymbol{\delta}^*\|_2 = \frac{|f(\mathbf{x})|}{\|\mathbf{w}\|_2}
$$

This is simply the distance from $\mathbf{x}$ to the decision hyperplane.

### Extension to Neural Networks

For a nonlinear classifier, DeepFool linearizes the decision boundary at the current point and iteratively projects onto the linearized boundary. At each iteration $t$:

1. Approximate the classifier as locally affine around $\mathbf{x}^{(t)}$
2. Compute the minimal perturbation to cross the approximate boundary
3. Update: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \boldsymbol{\delta}^{(t)}$
4. Repeat until misclassification

### Multi-Class Extension

For a multi-class classifier with logits $Z_k(\mathbf{x})$, the decision boundary between class $y$ (true) and class $k$ is:

$$
\{\mathbf{x} : Z_y(\mathbf{x}) = Z_k(\mathbf{x})\}
$$

At each iteration, DeepFool:

1. Computes the linearized distance to each class boundary:

$$
d_k = \frac{|Z_k(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})|}{\|\nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)}) - \nabla_\mathbf{x} Z_y(\mathbf{x}^{(t)})\|_2}
$$

2. Selects the nearest boundary: $\hat{k} = \arg\min_{k \neq y} d_k$

3. Computes the perturbation toward that boundary:

$$
\boldsymbol{\delta}^{(t)} = \frac{Z_{\hat{k}}(\mathbf{x}^{(t)}) - Z_y(\mathbf{x}^{(t)})}{\|\mathbf{w}_{\hat{k}} - \mathbf{w}_y\|_2^2} (\mathbf{w}_{\hat{k}} - \mathbf{w}_y)
$$

where $\mathbf{w}_k = \nabla_\mathbf{x} Z_k(\mathbf{x}^{(t)})$.

## Algorithm

**Algorithm: DeepFool**

**Input:** Input $\mathbf{x}$, classifier $f$, max iterations $T$, overshoot $\eta$

**Output:** Minimal adversarial perturbation $\hat{\boldsymbol{\delta}}$

1. Initialize: $\mathbf{x}^{(0)} = \mathbf{x}$, $\hat{\boldsymbol{\delta}} = \mathbf{0}$
2. While $f(\mathbf{x}^{(t)}) = f(\mathbf{x})$ and $t < T$:
    - For each class $k \neq y$: compute $\mathbf{w}_k' = \nabla_\mathbf{x} Z_k - \nabla_\mathbf{x} Z_y$ and $f_k' = Z_k - Z_y$
    - Find nearest boundary: $\hat{k} = \arg\min_{k \neq y} \frac{|f_k'|}{\|\mathbf{w}_k'\|_2}$
    - Compute step: $\boldsymbol{\delta}^{(t)} = \frac{|f_{\hat{k}}'|}{\|\mathbf{w}_{\hat{k}}'\|_2^2} \mathbf{w}_{\hat{k}}'$
    - Update: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
    - Accumulate: $\hat{\boldsymbol{\delta}} \leftarrow \hat{\boldsymbol{\delta}} + (1 + \eta) \boldsymbol{\delta}^{(t)}$
3. Return $\hat{\boldsymbol{\delta}}$

The overshoot parameter $\eta > 0$ (typically 0.02) ensures the perturbation crosses the boundary rather than landing exactly on it.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, Dict

class DeepFool:
    """
    DeepFool Attack: finds minimal L2 perturbation for misclassification.
    
    Iteratively linearizes the decision boundary and projects
    onto the nearest class boundary.
    
    Parameters
    ----------
    model : nn.Module
        Neural network to attack
    num_classes : int
        Number of output classes
    max_iter : int
        Maximum iterations
    overshoot : float
        Overshoot parameter to ensure boundary crossing
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        max_iter: int = 50,
        overshoot: float = 0.02,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.device = device or next(model.parameters()).device
        
        self.model.eval()
        self.model.to(self.device)
    
    def _attack_single(self, x: torch.Tensor) -> tuple:
        """
        Attack a single input.
        
        Returns (perturbation, iterations, original_label, final_label)
        """
        x = x.unsqueeze(0).to(self.device).clone().detach()
        x_orig = x.clone()
        
        # Get original prediction
        with torch.no_grad():
            logits = self.model(x)
            orig_label = logits.argmax(dim=1).item()
        
        x_pert = x.clone()
        total_pert = torch.zeros_like(x)
        
        for iteration in range(self.max_iter):
            x_pert.requires_grad_(True)
            logits = self.model(x_pert)
            pred = logits.argmax(dim=1).item()
            
            if pred != orig_label:
                break
            
            # Compute gradients for each class
            grads = []
            for k in range(self.num_classes):
                if k == orig_label:
                    grads.append(None)
                    continue
                
                self.model.zero_grad()
                if x_pert.grad is not None:
                    x_pert.grad.zero_()
                
                logits[0, k].backward(retain_graph=True)
                grad_k = x_pert.grad.data.clone()
                
                # Also need gradient of original class
                self.model.zero_grad()
                x_pert.grad.zero_()
                logits[0, orig_label].backward(retain_graph=True)
                grad_orig = x_pert.grad.data.clone()
                
                grads.append(grad_k - grad_orig)
            
            # Find nearest boundary
            min_dist = float('inf')
            best_delta = None
            
            for k in range(self.num_classes):
                if k == orig_label:
                    continue
                
                w_k = grads[k]
                f_k = (logits[0, k] - logits[0, orig_label]).item()
                
                w_norm = w_k.view(-1).norm(p=2).item()
                if w_norm < 1e-8:
                    continue
                
                dist = abs(f_k) / w_norm
                
                if dist < min_dist:
                    min_dist = dist
                    best_delta = (abs(f_k) / (w_norm ** 2)) * w_k
            
            if best_delta is None:
                break
            
            # Update with overshoot
            step = (1 + self.overshoot) * best_delta
            total_pert += step
            x_pert = (x_orig + total_pert).detach()
            x_pert = torch.clamp(x_pert, 0, 1)
        
        final_pert = (x_pert - x_orig).squeeze(0)
        with torch.no_grad():
            final_label = self.model(x_pert).argmax(dim=1).item()
        
        return final_pert, iteration + 1, orig_label, final_label
    
    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate adversarial examples for a batch.
        
        Parameters
        ----------
        x : torch.Tensor
            Clean inputs, shape (N, C, H, W)
        y : torch.Tensor
            True labels (used only for evaluation)
            
        Returns
        -------
        x_adv : torch.Tensor
            Adversarial examples
        """
        x_adv = x.clone()
        
        for i in range(len(x)):
            pert, iters, orig, final = self._attack_single(x[i])
            x_adv[i] = torch.clamp(x[i].to(self.device) + pert, 0, 1)
        
        return x_adv.detach()
    
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate attack with perturbation statistics."""
        with torch.no_grad():
            x_dev = x.to(self.device)
            y_dev = y.to(self.device)
            x_adv_dev = x_adv.to(self.device)
            
            clean_pred = self.model(x_dev).argmax(1)
            adv_pred = self.model(x_adv_dev).argmax(1)
            
            clean_acc = (clean_pred == y_dev).float().mean().item()
            robust_acc = (adv_pred == y_dev).float().mean().item()
            
            delta = (x_adv_dev - x_dev).view(len(x), -1)
            l2_norms = delta.norm(p=2, dim=1)
        
        return {
            'clean_accuracy': clean_acc,
            'robust_accuracy': robust_acc,
            'attack_success_rate': 1 - robust_acc,
            'avg_l2': l2_norms.mean().item(),
            'median_l2': l2_norms.median().item(),
            'min_l2': l2_norms.min().item(),
            'max_l2': l2_norms.max().item()
        }
```

## Properties and Comparison

### Strengths

- **Finds minimal perturbations**: Unlike fixed-$\varepsilon$ attacks, reveals the true distance to the decision boundary
- **Geometric insight**: Directly measures decision boundary proximity
- **No hyperparameter tuning**: No $\varepsilon$ to choose (except overshoot and max iterations)
- **Theoretically grounded**: Optimal for affine classifiers

### Limitations

- **Slow**: Requires per-class gradient computation at each iteration
- **L2 only**: Natural formulation is for $\ell_2$ norm; $\ell_\infty$ variant (Universal DeepFool) exists but is less elegant
- **Not strongest attack**: For fixed-budget evaluation, PGD and C&W are preferred
- **Sequential**: Each example must be processed individually

### Comparison with Other Attacks

| Aspect | DeepFool | PGD | C&W |
|--------|----------|-----|-----|
| Finds minimal $\|\boldsymbol{\delta}\|$ | Yes | No | Yes |
| Fixed-$\varepsilon$ evaluation | No | Yes | No |
| Speed | Slow | Moderate | Very slow |
| Primary norm | $\ell_2$ | $\ell_\infty$ or $\ell_2$ | $\ell_2$ |
| Use case | Geometry analysis | Robustness evaluation | Defense breaking |

## Applications

DeepFool is particularly valuable for:

- **Measuring robustness margins**: The average minimum perturbation across a dataset quantifies how close typical inputs are to decision boundaries
- **Comparing architectures**: Reveals which models have wider margins
- **Understanding decision geometry**: The direction of minimal perturbation indicates the most vulnerable input dimensions
- **Financial applications**: Measuring how much a model's credit decision or trading signal would change under minimal input perturbation

## References

1. Moosavi-Dezfooli, S. M., Fawzi, A., & Frossard, P. (2016). "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks." CVPR.
2. Moosavi-Dezfooli, S. M., et al. (2017). "Universal Adversarial Perturbations." CVPR.
