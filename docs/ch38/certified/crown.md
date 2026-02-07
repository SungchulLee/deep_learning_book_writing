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
