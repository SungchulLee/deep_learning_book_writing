# L-BFGS

## Overview

L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton optimization method that approximates the inverse Hessian using a limited history of gradient differences. Unlike first-order methods, L-BFGS uses curvature information to make more informed updates.

## Background

Newton's method updates parameters using the inverse Hessian:

$$\theta_{t+1} = \theta_t - H_t^{-1} g_t$$

Computing and storing the full Hessian $H \in \mathbb{R}^{d \times d}$ is infeasible for deep networks with millions of parameters. L-BFGS approximates $H^{-1}$ using only the last $m$ gradient and parameter differences, requiring $O(md)$ storage instead of $O(d^2)$.

## PyTorch Implementation

L-BFGS in PyTorch requires a closure that recomputes the loss:

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0,
                               max_iter=20, history_size=10)

def closure():
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

The closure pattern is necessary because L-BFGS may evaluate the function multiple times per step (line search).

## When to Use

L-BFGS is effective for:

- **Small-to-medium models**: Where the full dataset fits in memory and the overhead of multiple forward/backward passes per step is acceptable.
- **Physics-informed neural networks (PINNs)**: Where high precision is needed and datasets are small.
- **Neural ODE calibration**: Fitting neural ODEs to financial data with smooth loss landscapes.

L-BFGS is generally not suitable for large-scale deep learning with mini-batch stochastic gradients, as the noisy gradient estimates corrupt the Hessian approximation.

## Key Takeaways

- L-BFGS uses curvature information for faster convergence on smooth problems.
- Requires a closure for function re-evaluation and works best with full-batch gradients.
- Excellent for small models and optimization problems requiring high precision; impractical for large-scale stochastic training.
