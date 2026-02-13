# Unadjusted Langevin Algorithm (ULA)

The Unadjusted Langevin Algorithm discretizes the Langevin SDE without a Metropolis-Hastings correction. This makes it simple and compatible with stochastic gradients, but introduces a persistent bias that depends on the step size.

---

## Algorithm

```
Algorithm: Unadjusted Langevin Algorithm (ULA)
──────────────────────────────────────────────
Input: ∇log π̃(θ), step size ε, n_samples T
Initialize: θ₀

For t = 0, 1, ..., T-1:
    η ~ N(0, I)
    θₜ₊₁ = θₜ + (ε/2) ∇log π̃(θₜ) + √ε η
```

This is simply the Euler-Maruyama discretization of the Langevin SDE:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon}{2} \nabla \log \pi(\theta_t) + \sqrt{\epsilon} \, \boldsymbol{\eta}_t
$$

---

## Bias Analysis

ULA's stationary distribution $\pi_\epsilon$ differs from the true target $\pi$. For a step size $\epsilon$, the bias in total variation distance is:

$$
\text{TV}(\pi_\epsilon, \pi) = O(\epsilon)
$$

For strongly log-concave targets with Lipschitz gradients (smoothness constant $L$, strong convexity $m$):

$$
\text{TV}(\pi_\epsilon, \pi) \leq C \cdot \epsilon \cdot \frac{L^2 d}{m}
$$

**Implication**: To achieve bias $\leq \delta$, we need step size $\epsilon = O(\delta / (L^2 d))$, and convergence requires $O(d/\delta^2)$ iterations — polynomial in dimension but inversely quadratic in desired accuracy.

---

## Stochastic Gradient Langevin Dynamics (SGLD)

The key practical extension: replace the full gradient with a **stochastic gradient** from a minibatch:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\left(\nabla \log p(\theta_t) + \frac{N}{n}\sum_{i \in \text{batch}} \nabla \log p(x_i \mid \theta_t)\right) + \sqrt{\epsilon_t} \, \boldsymbol{\eta}_t
$$

where $N$ is the dataset size and $n$ is the minibatch size.

### Decreasing Step Size Schedule

With a decreasing step size $\epsilon_t$ satisfying:

$$
\sum_{t=1}^{\infty} \epsilon_t = \infty, \quad \sum_{t=1}^{\infty} \epsilon_t^2 < \infty
$$

(e.g., $\epsilon_t = a / (b + t)$), the stochastic gradient noise and discretization bias both vanish asymptotically, and the samples converge to the true posterior.

### PyTorch Implementation

```python
import torch


class SGLD:
    """
    Stochastic Gradient Langevin Dynamics.
    
    Combines minibatch gradients with Langevin noise for
    scalable approximate posterior sampling.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 noise_scale=1.0, lr_decay=0.0):
        self.params = list(params)
        self.lr_init = lr
        self.weight_decay = weight_decay
        self.noise_scale = noise_scale
        self.lr_decay = lr_decay
        self.step_count = 0
    
    @property
    def lr(self):
        if self.lr_decay > 0:
            return self.lr_init / (1 + self.lr_decay * self.step_count)
        return self.lr_init
    
    def step(self):
        """Perform one SGLD update."""
        eps = self.lr
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # Gradient step (includes data likelihood + prior via weight_decay)
            d_p = p.grad.data
            if self.weight_decay > 0:
                d_p = d_p + self.weight_decay * p.data
            
            # Langevin noise
            noise = torch.randn_like(p.data) * (self.noise_scale * (2 * eps) ** 0.5)
            
            # Update: θ ← θ - ε∇U(θ) + √(2ε)η
            p.data.add_(-eps * d_p + noise)
        
        self.step_count += 1
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def sgld_sample(model, dataloader, n_samples=100, burnin=1000,
                thin=10, lr=1e-4):
    """
    Collect posterior samples from a neural network using SGLD.
    """
    optimizer = SGLD(model.parameters(), lr=lr, lr_decay=1e-5)
    
    samples = []
    total_steps = burnin + n_samples * thin
    
    for step in range(total_steps):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            break  # one batch per step
        
        if step >= burnin and (step - burnin) % thin == 0:
            samples.append({
                name: p.data.clone() 
                for name, p in model.named_parameters()
            })
    
    return samples
```

---

## ULA vs MALA vs HMC

| Aspect | ULA/SGLD | MALA | HMC |
|--------|----------|------|-----|
| MH correction | No | Yes | Yes |
| Bias | $O(\epsilon)$ | $O(\epsilon^3)$ | $O(\epsilon^{L})$ |
| Stochastic gradients | Natural | Difficult | Difficult |
| Scalability | Excellent | Good | Moderate |
| Accuracy | Approximate | Asymptotically exact | Asymptotically exact |
| Tuning | Step size only | Step size | Step size, trajectory, mass |

---

## When to Use ULA/SGLD

**Use SGLD when:**
- Dataset is very large (millions of observations)
- Approximate posterior is acceptable
- Integrating with SGD-based training pipelines
- Need simplicity and minimal tuning

**Prefer MALA/HMC when:**
- Exact posterior samples are required
- Dataset is small to moderate
- High accuracy for uncertainty quantification is critical

---

## References

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Dalalyan, A. S. (2017). Theoretical guarantees for approximate sampling from a smooth and log-concave density. *JRSS-B*, 79(3), 651-676.
- Chen, T., Fox, E., & Guestrin, C. (2014). Stochastic gradient Hamiltonian Monte Carlo. *ICML*.

---

## Detailed Bias Example: 1-D Gaussian

For $\pi(x) = \mathcal{N}(0, 1)$ the score is $s(x) = -x$. The ULA update is:

$$
x_{t+1} = x_t - \epsilon x_t + \sqrt{2\epsilon} \, \eta_t = (1 - \epsilon) x_t + \sqrt{2\epsilon} \, \eta_t
$$

This is an AR(1) process with stationary variance:

$$
\sigma^2_\epsilon = \frac{2\epsilon}{1 - (1-\epsilon)^2} = \frac{1}{1 - \epsilon/2}
$$

For $\epsilon = 0.1$ the stationary variance is $\approx 1.053$, not 1. The bias is $O(\epsilon)$.

---

## Basic ULA Implementation

```python
import torch

def ula(score_fn, x0, n_steps, epsilon):
    """Unadjusted Langevin Algorithm.

    Args:
        score_fn: Maps x [batch, dim] → score [batch, dim].
        x0: Initial state [batch, dim].
        n_steps: Number of iterations.
        epsilon: Step size.

    Returns:
        Samples [batch, dim] after n_steps iterations.
    """
    x = x0.clone()
    sqrt_2eps = (2 * epsilon) ** 0.5
    for _ in range(n_steps):
        x = x + epsilon * score_fn(x) + sqrt_2eps * torch.randn_like(x)
    return x
```

---

## When to Use ULA

ULA is appropriate when:

- Exact samples are not required (e.g., approximate inference, optimisation warm-starts)
- The step size can be made small enough that bias is negligible
- Speed matters more than exactness

### Convergence Diagnostics for ULA

Standard MCMC diagnostics apply with caveats:

- **Trace plots** should show stationarity and good mixing
- **ESS** measures effective independence of samples
- **Running mean** should stabilize—but to a biased target
- Multiple chains should converge to the same (biased) distribution
