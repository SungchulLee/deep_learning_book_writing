# CROWN: Convex Relaxation-Based Certification
## Introduction

**CROWN** (Zhang et al., 2018) and its variants provide tighter certified bounds than IBP by using **linear relaxations** of nonlinear activation functions. While IBP propagates independent lower and upper bounds, CROWN computes bounds that depend on the input, yielding significantly tighter certificates at moderate computational cost.

## Mathematical Foundation

### Linear Relaxation of ReLU

The key challenge in bound propagation is handling nonlinear activations. For ReLU $z = \max(0, x)$ with bounds $x \in [\underline{x}, \overline{x}]$, CROWN uses a **linear relaxation**:

**Case 1:** $\underline{x} \geq 0$ (always active): $z = x$ exactly

**Case 2:** $\overline{x} \leq 0$ (always inactive): $z = 0$ exactly

**Case 3:** $\underline{x} < 0 < \overline{x}$ (unstable neuron):

$$
\alpha x \leq z \leq \frac{\overline{x}}{\overline{x} - \underline{x}}(x - \underline{x})
$$

where $\alpha \in [0, 1]$ is a learnable or heuristic slope parameter for the lower bound.

### Bound Propagation

CROWN expresses final-layer bounds as **linear functions** of the input:

$$
\underline{z}_L = \mathbf{A}^L \mathbf{x} + \mathbf{b}^L_\text{lower}, \quad \overline{z}_L = \mathbf{A}^U \mathbf{x} + \mathbf{b}^U_\text{upper}
$$

By back-substituting through layers, the final bounds depend linearly on the input, enabling efficient optimization over the input perturbation set.

### CROWN-IBP

**CROWN-IBP** combines the tightness of CROWN with the efficiency of IBP:

$$
\mathcal{L} = \beta \cdot \mathcal{L}_{\text{CROWN}} + (1 - \beta) \cdot \mathcal{L}_{\text{IBP}}
$$

During training, $\beta$ is annealed from 1 (CROWN, tighter but slower) to 0 (IBP, faster), combining the benefits of both approaches.

## Auto-LiRPA

**Auto-LiRPA** (Xu et al., 2020) is a general framework that automates linear relaxation-based perturbation analysis for arbitrary computational graphs, extending CROWN beyond simple feedforward networks.

```python
# Using auto_LiRPA library
# pip install auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# Wrap model
bounded_model = BoundedModule(model, torch.zeros(1, 3, 32, 32))

# Define perturbation
ptb = PerturbationLpNorm(norm=float('inf'), eps=8/255)
bounded_input = BoundedTensor(x, ptb)

# Compute bounds
lb, ub = bounded_model.compute_bounds(
    x=(bounded_input,), method='CROWN'
)

# Certification: check if lb[y] > max(ub[k]) for k ≠ y
```

## Comparison of Certification Methods

| Method | Bound Tightness | Computational Cost | Scalability |
|--------|----------------|-------------------|-------------|
| IBP | Loose | Low | Good |
| CROWN | Tight | High | Moderate |
| CROWN-IBP | Moderate | Moderate | Good |
| α-CROWN | Tightest | Highest | Limited |
| SDP relaxation | Very tight | Very high | Small nets only |

## Summary

CROWN and its variants represent the state of the art in deterministic certified robustness, providing tighter bounds than IBP at increased computational cost. The CROWN-IBP combination offers the best practical trade-off for training certifiably robust networks, while Auto-LiRPA extends these ideas to general architectures.

## References

1. Zhang, H., et al. (2018). "Efficient Neural Network Robustness Certification with General Activation Functions." NeurIPS.
2. Xu, K., et al. (2020). "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond." NeurIPS.
3. Zhang, H., et al. (2020). "General Cutting Planes for Bound-Propagation-Based Neural Network Verification." NeurIPS.

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
