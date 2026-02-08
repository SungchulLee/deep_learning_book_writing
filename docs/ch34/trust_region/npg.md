# 34.3.2 Natural Policy Gradient

## Introduction

The Natural Policy Gradient (NPG) (Kakade, 2001) replaces standard gradient ascent with steps in the direction of the natural gradient, which accounts for the geometry of the policy's probability distribution. While standard gradients treat parameter space as Euclidean, the natural gradient uses the Fisher information metric, ensuring that equal-sized steps in parameter space produce equal-sized changes in the policy distribution.

## Motivation

Consider two policies that differ by a small parameter change $\Delta\theta$. The actual change in policy behavior depends on the local curvature of the parameter-to-distribution mapping:

- A large parameter change in a "flat" direction may barely change the policy
- A small parameter change in a "curved" direction may drastically change the policy

The natural gradient normalizes for this curvature, producing updates that are invariant to the policy's parameterization.

## Fisher Information Matrix

The Fisher information matrix $F(\theta)$ captures the local curvature of the KL divergence:

$$F(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top\right]$$

Equivalently, it is the Hessian of the KL divergence at zero displacement:

$$F(\theta) = \nabla_\theta^2 D_{\text{KL}}(\pi_\theta \| \pi_{\theta'})\big|_{\theta' = \theta}$$

## Natural Gradient Update

The natural gradient replaces the standard gradient with:

$$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)$$

The natural policy gradient update:

$$\theta_{k+1} = \theta_k + \alpha F(\theta_k)^{-1} \nabla_\theta J(\theta_k)$$

### Interpretation

The natural gradient solves the constrained optimization:

$$\tilde{\nabla}_\theta J = \arg\max_{\Delta\theta} \left\{\nabla_\theta J^\top \Delta\theta \quad \text{s.t.} \quad \Delta\theta^\top F \Delta\theta \leq \epsilon\right\}$$

This finds the direction of steepest ascent in $J$ subject to a KL divergence constraint on the policy change.

## Connection to TRPO

TRPO can be viewed as a practical implementation of the natural policy gradient:
1. NPG computes $F^{-1}g$ and takes a fixed step
2. TRPO computes $F^{-1}g$ and uses line search to find the best step within the trust region
3. Both use conjugate gradient to avoid forming $F$ explicitly

The key difference is that TRPO adjusts the step size to strictly satisfy the KL constraint, while NPG uses a fixed learning rate.

## Compatible Function Approximation

Kakade showed that using a compatible value function:

$$Q_w(s, a) = \nabla_\theta \log \pi_\theta(a|s)^\top w + V(s)$$

The natural gradient direction is exactly the compatible value function parameters:

$$F(\theta)^{-1} \nabla_\theta J(\theta) = w^*$$

where $w^*$ minimizes the compatible function approximation error.

## Practical Computation

For neural network policies, the Fisher matrix is too large to form or invert directly. Practical approaches:

1. **Conjugate gradient**: Solve $Fx = g$ iteratively using only Fisher-vector products
2. **Kronecker-factored approximation (K-FAC)**: Approximate $F$ using Kronecker products
3. **Diagonal approximation**: Use only the diagonal of $F$
4. **Empirical Fisher**: Use the outer product of individual sample gradients

## Advantages Over Standard Gradient

| Property | Standard Gradient | Natural Gradient |
|----------|------------------|------------------|
| Parameterization invariance | No | Yes |
| Convergence rate | $O(1/\sqrt{t})$ | $O(1/t)$ (in some settings) |
| Step interpretation | Parameter space | Distribution space |
| Plateau handling | Slow | Faster |

## Summary

The natural policy gradient provides a principled foundation for trust region methods. By accounting for the Fisher information geometry of the policy space, it produces updates that are invariant to reparameterization and typically more efficient than standard gradient ascent. TRPO and PPO can be understood as practical approximations to the natural gradient update.
