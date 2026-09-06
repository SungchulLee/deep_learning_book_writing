# Lipschitz-Constrained Networks
## Introduction

**Lipschitz-constrained networks** achieve certified robustness by controlling the network's sensitivity to input perturbations through explicit Lipschitz constant constraints. If a network has Lipschitz constant $L$, then the change in output is bounded by $L$ times the change in input, providing an inherent robustness certificate.

## Mathematical Foundation

### Lipschitz Continuity

A function $f: \mathbb{R}^d \to \mathbb{R}^m$ is **$L$-Lipschitz** under norm $\|\cdot\|$ if:

$$
\|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L \cdot \|\mathbf{x}_1 - \mathbf{x}_2\|
$$

for all $\mathbf{x}_1, \mathbf{x}_2$ in the domain.

### Robustness Certificate

For a classifier with margin $m(\mathbf{x}) = f(\mathbf{x})_y - \max_{k \neq y} f(\mathbf{x})_k$ and Lipschitz constant $L$, the certified radius is:

$$
R(\mathbf{x}) = \frac{m(\mathbf{x})}{\sqrt{2} \cdot L}
$$

Any perturbation $\|\boldsymbol{\delta}\|_2 \leq R(\mathbf{x})$ is guaranteed not to change the prediction.

### Compositional Lipschitz Bounds

For a network $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$, the overall Lipschitz constant is bounded by the product:

$$
L_f \leq \prod_{\ell=1}^L L_{f_\ell}
$$

For a linear layer with weight matrix $\mathbf{W}$, $L = \|\mathbf{W}\|_\sigma$ (spectral norm). For ReLU, $L = 1$.

## Spectral Normalization

The most common approach constrains each layer's spectral norm:

$$
\hat{\mathbf{W}} = \frac{\mathbf{W}}{\|\mathbf{W}\|_\sigma}
$$

This ensures each linear layer has Lipschitz constant at most 1.

```python
import torch
import torch.nn as nn

class LipschitzLinear(nn.Module):
    """Linear layer with spectral normalization for Lipschitz bound."""
    
    def __init__(self, in_features, out_features, lip_const=1.0):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features)
        )
        self.lip_const = lip_const
    
    def forward(self, x):
        return self.lip_const * self.linear(x)

class LipschitzNetwork(nn.Module):
    """
    Network with controlled Lipschitz constant.
    
    Uses spectral normalization on all layers to bound
    the overall Lipschitz constant.
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h in hidden_dims:
            layers.append(LipschitzLinear(prev_dim, h))
            layers.append(nn.ReLU())  # Lipschitz constant = 1
            prev_dim = h
        
        layers.append(LipschitzLinear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.flatten(1))
    
    def certify(self, x, y):
        """Compute certified radius for each input."""
        logits = self.forward(x)
        
        # Classification margin
        true_logit = logits.gather(1, y.view(-1, 1)).squeeze()
        mask = torch.ones_like(logits).scatter_(1, y.view(-1, 1), 0)
        max_other = (logits * mask - (1 - mask) * 1e9).max(dim=1)[0]
        margin = true_logit - max_other
        
        # Lipschitz constant (product of spectral norms, all ≤ 1)
        L = 1.0  # With spectral norm, L ≤ 1 per layer
        
        # Certified L2 radius
        radius = margin / (2**0.5 * L)
        radius = torch.clamp(radius, min=0)
        
        return radius
```

## Advanced Approaches

### Orthogonal Layers

Using orthogonal weight matrices ($\mathbf{W}^\top \mathbf{W} = \mathbf{I}$) gives exactly $L = 1$ per layer, with no bound looseness:

- **Cayley transform**: Parameterizes orthogonal matrices via $\mathbf{W} = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})^{-1}$ for skew-symmetric $\mathbf{A}$
- **Householder reflections**: Composes orthogonal matrices from Householder reflections

### GroupSort Activation

The **GroupSort** activation (Anil et al., 2019) is a 1-Lipschitz alternative to ReLU that preserves more information:

$$
\text{GroupSort}_{k}(\mathbf{x}) = \text{sort groups of } k \text{ elements}
$$

For $k=2$: sorts pairs, equivalent to $(\min(x_1, x_2), \max(x_1, x_2))$.

## Trade-offs

| Aspect | Lipschitz Networks | IBP/CROWN | Randomized Smoothing |
|--------|-------------------|-----------|---------------------|
| Certificate type | Deterministic | Deterministic | Probabilistic |
| Norm | $\ell_2$ (natural) | $\ell_\infty$ (natural) | $\ell_2$ |
| Architecture constraints | Strong | Moderate | None |
| Certified radius | Small-moderate | Small | Large |
| Clean accuracy | Limited | Moderate | Higher |

## Summary

Lipschitz-constrained networks provide elegant robustness certificates directly from the network architecture. While current certified radii remain smaller than those achievable by randomized smoothing, the deterministic guarantees and the clean theoretical framework make this an active and promising research direction.

## References

1. Anil, C., Lucas, J., & Grosse, R. (2019). "Sorting Out Lipschitz Function Approximation." ICML.
2. Li, Q., et al. (2019). "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks." NeurIPS.
3. Trockman, A., & Kolter, J. Z. (2021). "Orthogonalizing Convolutional Layers with the Cayley Transform." ICLR.

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
