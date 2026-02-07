# Metropolis-Adjusted Langevin Algorithm (MALA)

MALA combines the gradient-informed proposals of Langevin dynamics with a Metropolis-Hastings accept-reject correction, eliminating the discretization bias of ULA while retaining the benefits of gradient-guided exploration.

---

## Algorithm

```
Algorithm: MALA
───────────────
Input: log π̃(θ), ∇log π̃(θ), step size ε
Initialize: θ₀

For t = 0, 1, ..., T-1:
    1. Propose:
       θ' = θₜ + (ε/2) ∇log π̃(θₜ) + √ε η,    η ~ N(0, I)
    
    2. Compute acceptance probability:
       α = min(1, [π̃(θ') q(θₜ|θ')] / [π̃(θₜ) q(θ'|θₜ)])
    
       where q(θ'|θ) = N(θ' | θ + (ε/2)∇log π̃(θ), εI)
    
    3. Accept/reject:
       u ~ Uniform(0,1)
       If u < α:  θₜ₊₁ = θ'    (accept)
       Else:      θₜ₊₁ = θₜ    (reject)
```

### The Proposal Distribution

The MALA proposal is a Gaussian centered at the Langevin-guided position:

$$
q(\theta' \mid \theta) = \mathcal{N}\left(\theta' \;\Big|\; \theta + \frac{\epsilon}{2}\nabla \log \tilde{\pi}(\theta), \; \epsilon \mathbf{I}\right)
$$

This is **asymmetric**: $q(\theta' \mid \theta) \neq q(\theta \mid \theta')$ because the gradient term differs at $\theta$ and $\theta'$. The MH ratio must therefore include the proposal densities.

### Log Acceptance Ratio

$$
\log \alpha = \log \tilde{\pi}(\theta') - \log \tilde{\pi}(\theta) + \log q(\theta \mid \theta') - \log q(\theta' \mid \theta)
$$

where:

$$
\log q(\theta' \mid \theta) = -\frac{1}{2\epsilon}\left\|\theta' - \theta - \frac{\epsilon}{2}\nabla \log \tilde{\pi}(\theta)\right\|^2 + \text{const}
$$

---

## Bias Correction

The MH step corrects the $O(\epsilon)$ discretization bias of ULA:

| Method | Stationary Distribution | Bias |
|--------|------------------------|------|
| ULA | $\pi_\epsilon \neq \pi$ | $O(\epsilon)$ |
| **MALA** | $\pi$ (exact) | **None** (asymptotically exact) |
| HMC | $\pi$ (exact) | None (asymptotically exact) |

However, MALA's acceptance rate depends on $\epsilon$ and dimension $d$:

$$
\text{Optimal } \epsilon = O(d^{-1/3})
$$

with an optimal acceptance rate of approximately **0.574** (Roberts & Rosenthal, 1998).

---

## PyTorch Implementation

```python
import torch


class MALA:
    """
    Metropolis-Adjusted Langevin Algorithm.
    
    Parameters
    ----------
    log_prob : callable
        Computes log π̃(θ). Must support autograd.
    step_size : float
        Langevin step size ε
    """
    
    def __init__(self, log_prob, step_size=0.1):
        self.log_prob = log_prob
        self.step_size = step_size
    
    def _grad_log_prob(self, theta):
        """Compute ∇log π̃(θ) via autograd."""
        theta = theta.detach().requires_grad_(True)
        lp = self.log_prob(theta)
        lp.backward()
        return theta.grad.detach()
    
    def _log_proposal(self, theta_to, theta_from):
        """Compute log q(theta_to | theta_from)."""
        grad = self._grad_log_prob(theta_from)
        mean = theta_from + 0.5 * self.step_size * grad
        diff = theta_to - mean
        return -0.5 / self.step_size * (diff ** 2).sum()
    
    def sample(self, theta_init, n_samples, warmup=1000):
        """Run MALA chain."""
        d = theta_init.shape[0]
        theta = theta_init.clone().float()
        
        samples = torch.zeros(n_samples, d)
        n_accept = 0
        
        for t in range(n_samples + warmup):
            # Langevin proposal
            grad = self._grad_log_prob(theta)
            mean = theta + 0.5 * self.step_size * grad
            theta_prop = mean + (self.step_size ** 0.5) * torch.randn(d)
            
            # Log acceptance ratio
            log_alpha = (
                self.log_prob(theta_prop) - self.log_prob(theta)
                + self._log_proposal(theta, theta_prop)
                - self._log_proposal(theta_prop, theta)
            )
            
            # Accept/reject
            if torch.log(torch.rand(1)) < log_alpha:
                theta = theta_prop
                if t >= warmup:
                    n_accept += 1
            
            if t >= warmup:
                samples[t - warmup] = theta.detach()
        
        accept_rate = n_accept / n_samples
        return samples, accept_rate
```

---

## Preconditioned MALA

Using a preconditioning matrix $\mathbf{M}$ (analogous to the HMC mass matrix):

$$
\theta' = \theta + \frac{\epsilon}{2}\mathbf{M}\nabla \log \tilde{\pi}(\theta) + \sqrt{\epsilon} \, \mathbf{M}^{1/2} \boldsymbol{\eta}
$$

Setting $\mathbf{M}$ to an approximation of the posterior covariance equalizes step sizes across directions, improving mixing for anisotropic posteriors.

The Riemannian manifold variant uses the **Fisher information matrix** $\mathbf{G}(\theta)$ as a position-dependent preconditioner, adapting to local curvature.

---

## MALA in the Sampling Hierarchy

| Method | Proposal | Correction | Scaling | Best For |
|--------|----------|------------|---------|----------|
| Random Walk MH | $\mathcal{N}(\theta, \sigma^2 I)$ | MH | $O(d)$ | Low-$d$ |
| ULA/SGLD | Langevin step | None | — | Large data |
| **MALA** | Langevin step | **MH** | $O(d^{1/3})$ | **Moderate-$d$** |
| HMC | Hamiltonian trajectory | MH | $O(d^{1/4})$ | High-$d$ |

MALA occupies a sweet spot: simpler than HMC, more accurate than ULA, and better scaling than random walk MH.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **MALA = ULA + MH correction** | Eliminates discretization bias |
| **Asymmetric proposal** | Gradient makes $q(\theta' \mid \theta) \neq q(\theta \mid \theta')$ |
| **Optimal acceptance** | ~57.4% (compare: RW-MH ~23.4%, HMC ~65%) |
| **Step size scaling** | $\epsilon = O(d^{-1/3})$ — better than RW-MH's $O(d^{-1})$ |
| **Preconditioning** | Match posterior geometry for better mixing |

---

## References

- Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
- Roberts, G. O., & Rosenthal, J. S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. *JRSS-B*, 60(1), 255-268.
- Girolami, M., & Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *JRSS-B*, 73(2), 123-214.

---

## Why the Hastings Correction Is Needed

The proposal is **asymmetric**: $q(x' | x) \neq q(x | x')$ because the drift $\epsilon s(x)$ depends on the current position. Without the Hastings correction, detailed balance would be violated and the chain would not have $\pi$ as its stationary distribution.

### Computing the Log-Acceptance Ratio in Detail

The proposal log-densities are:

$$
\log q(x' | x) = -\frac{\|x' - x - \frac{\epsilon}{2} s(x)\|^2}{2\epsilon} + \text{const}
$$

$$
\log q(x | x') = -\frac{\|x - x' - \frac{\epsilon}{2} s(x')\|^2}{2\epsilon} + \text{const}
$$

So the proposal log-ratio is:

$$
\log q(x | x') - \log q(x' | x) = \frac{1}{2\epsilon}\left[\|x' - x - \tfrac{\epsilon}{2} s(x)\|^2 - \|x - x' - \tfrac{\epsilon}{2} s(x')\|^2\right]
$$

---

## Batch MALA Implementation

```python
import torch

def mala_batch(log_prob_fn, score_fn, x0, n_steps, epsilon):
    """Batch MALA for multiple parallel chains.

    Args:
        log_prob_fn: Maps x [batch, dim] → log π(x) [batch].
        score_fn: Maps x [batch, dim] → ∇ log π(x) [batch, dim].
        x0: Initial state [batch, dim].
        n_steps: Number of iterations.
        epsilon: Step size.

    Returns:
        samples: Final samples [batch, dim].
        accept_rate: Fraction of accepted proposals.
    """
    x = x0.clone()
    n_accept = 0
    sqrt_eps = epsilon ** 0.5

    for _ in range(n_steps):
        s_x = score_fn(x)
        noise = torch.randn_like(x)
        x_prop = x + 0.5 * epsilon * s_x + sqrt_eps * noise

        s_xp = score_fn(x_prop)

        # Log acceptance ratio
        log_pi_diff = log_prob_fn(x_prop) - log_prob_fn(x)

        # Proposal log-ratio: log q(x|x') - log q(x'|x)
        fwd = x_prop - x - 0.5 * epsilon * s_x
        bwd = x - x_prop - 0.5 * epsilon * s_xp
        log_q_diff = (fwd.pow(2).sum(dim=-1) - bwd.pow(2).sum(dim=-1)) / (2 * epsilon)

        log_alpha = log_pi_diff + log_q_diff
        accept = torch.log(torch.rand(x.shape[0], device=x.device)) < log_alpha

        x = torch.where(accept.unsqueeze(-1), x_prop, x)
        n_accept += accept.float().mean().item()

    return x, n_accept / n_steps
```

---

## Optimal Tuning

### Step Size Scaling

Optimal step sizes depend on dimension $d$:

| Algorithm | Optimal $\epsilon$ | Scaling |
|-----------|-------------------|---------| 
| ULA | $\epsilon \propto d^{-1/3}$ | Bias control |
| MALA | $\epsilon \propto d^{-1/6}$ | Acceptance rate |

MALA tolerates larger steps because the acceptance step catches overly aggressive proposals.

### Practical Tuning

**Target acceptance rate**: approximately **57%** (Roberts & Rosenthal, 1998). If acceptance is too high, the step size is too small and mixing is slow. If acceptance is too low, most proposals are rejected and the chain barely moves.

- Acceptance > 70%: increase $\epsilon$
- Acceptance < 45%: decrease $\epsilon$

### Example: Verifying MALA Tuning

```python
# Target: N(0, 1) in 10 dimensions
dim = 10
n_samples = 1000

def log_prob(x):
    return -0.5 * x.pow(2).sum(dim=-1)

def score(x):
    return -x

x0 = torch.randn(n_samples, dim) * 3  # overdispersed start

x_mala, acc = mala_batch(log_prob, score, x0.clone(), n_steps=500, epsilon=0.5)
print(f"MALA variance: {x_mala.var(dim=0).mean():.4f}, acceptance: {acc:.2%}")
```
