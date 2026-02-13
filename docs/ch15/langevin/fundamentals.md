# Langevin Dynamics Fundamentals

Langevin dynamics provides a continuous-time framework that connects MCMC sampling with gradient-based optimization. The Langevin stochastic differential equation (SDE) describes a particle undergoing gradient-driven drift plus Brownian noise, and its stationary distribution is the target posterior. This connection underpins both practical sampling algorithms and modern score-based generative models.

---

## Physical Origins: Brownian Motion

In 1908, Paul Langevin proposed an equation to describe **Brownian motion**—the random motion of a particle suspended in a fluid, buffeted by collisions with surrounding molecules. This equation unified deterministic forces (from potential energy) with random fluctuations (from thermal noise).

The remarkable insight: the same mathematics that describes physical particles can be used to **sample from probability distributions**.

### The Classical Langevin Equation

Consider a particle with position $x$ and mass $m$ moving in a potential $U(x)$:

$$
m\frac{d^2 x}{dt^2} = -\gamma \frac{dx}{dt} - \nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

where:

- $\gamma$ is the **friction coefficient** (viscous damping)
- $k_B T$ is the **thermal energy** (Boltzmann constant times temperature)
- $\xi(t)$ is **white noise** satisfying $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$

The three forces are:

| Force | Expression | Physical Meaning |
|-------|------------|------------------|
| Friction | $-\gamma \dot{x}$ | Opposes motion, dissipates energy |
| Potential | $-\nabla U(x)$ | Pushes toward energy minima |
| Thermal | $\sqrt{2\gamma k_B T} \, \xi(t)$ | Random kicks from molecular collisions |

### The Overdamped Limit

In the **high-friction regime** where inertia is negligible ($m \to 0$), the acceleration term vanishes. This gives the **overdamped Langevin equation**:

$$
\gamma \frac{dx}{dt} = -\nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

Rescaling time by $\gamma$ and setting $k_B T = 1$:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

where $W_t$ is standard **Brownian motion** (Wiener process).

!!! info "From Physics to Sampling"
    The Boltzmann distribution at temperature $T=1$ is $\pi(x) \propto \exp(-U(x))$. The Langevin equation converges to this distribution—particles spend more time where energy is low!

---

## The Langevin SDE for Sampling

### Formulation

To sample from a target distribution $\pi(x) \propto \exp(-U(x))$, we simulate the SDE:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

Equivalently, using the score function $s(x) = \nabla_x \log \pi(x) = -\nabla U(x)$:

$$
\boxed{dx_t = s(x_t) \, dt + \sqrt{2} \, dW_t}
$$

### Interpreting the Two Terms

**Drift term** $s(x_t) \, dt$:

- Deterministic flow toward high-probability regions
- Points "uphill" in the probability landscape
- Drives exploitation of modes

**Diffusion term** $\sqrt{2} \, dW_t$:

- Random exploration via Brownian motion
- Prevents getting stuck at modes
- Enables exploration of the full distribution

The constant $\sqrt{2}$ in front of $dW_t$ is not arbitrary—it is precisely calibrated so that the stationary distribution is $\pi(x)$.

---

## The Score Function

The **score function** $\mathbf{s}(x) = \nabla_x \log \pi(x)$ is the central quantity in Langevin methods:

$$
\nabla_x \log \pi(x) = \nabla_x \log p(\mathcal{D} \mid x) + \nabla_x \log p(x)
$$

Properties:

- Points toward increasing posterior density
- Magnitude reflects the local gradient steepness
- At a mode: $\mathbf{s}(x^*) = \mathbf{0}$
- Does not require the normalizing constant: $\nabla_x \log(\tilde{\pi}(x)/Z) = \nabla_x \log \tilde{\pi}(x)$

---

## General Form with Temperature

At temperature $T$, the target distribution is $\pi_T(x) \propto \exp(-U(x)/T)$ and the SDE becomes:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T} \, dW_t
$$

| Temperature | Noise Level | Behaviour |
|-------------|-------------|-----------|
| High $T$ | Large $\sqrt{2T}$ | Wide exploration, modes blurred |
| $T = 1$ | $\sqrt{2}$ | Standard sampling from $\pi$ |
| Low $T$ | Small $\sqrt{2T}$ | Concentrates near modes |
| $T \to 0$ | No noise | Deterministic gradient descent to minimum |

---

## The Fokker-Planck Equation

### Density Evolution

The **Fokker-Planck equation** (forward Kolmogorov equation) describes how the probability density $\rho(x, t)$ of $x_t$ evolves:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \, \nabla \log \pi) + \Delta \rho
$$

where $\Delta = \nabla \cdot \nabla$ is the Laplacian.

### Stationary Solution

At stationarity ($\partial \rho / \partial t = 0$), we claim $\rho = \pi$ is the stationary solution.

**Proof**: Substitute $\rho = \pi$:

$$
-\nabla \cdot (\pi \nabla \log \pi) + \Delta \pi = -\nabla \cdot \left(\pi \cdot \frac{\nabla \pi}{\pi}\right) + \Delta \pi = -\Delta \pi + \Delta \pi = 0 \quad \checkmark
$$

### An Elegant Reformulation

The Fokker-Planck equation can be rewritten as:

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot \left( \pi \nabla \left( \frac{\rho}{\pi} \right) \right)
$$

This form makes it obvious that $\rho = \pi$ is stationary (the gradient of a constant vanishes). Moreover, this shows that Langevin dynamics performs **gradient flow** in the space of probability distributions, minimizing the KL divergence from $\rho$ to $\pi$.

---

## Convergence Guarantees

### Convergence Theorem

Under mild conditions on $U(x)$:

1. **Strong convexity**: If $U$ is $m$-strongly convex ($\nabla^2 U \succeq m I$), the KL divergence contracts exponentially:
   
   $$
   D_{KL}(\rho_t \| \pi) \leq e^{-2mt} D_{KL}(\rho_0 \| \pi)
   $$

2. **General case**: With Lipschitz gradients and dissipativity conditions, the chain is geometrically ergodic.

### Mixing Time

For a $d$-dimensional Gaussian target, the mixing time scales as $\mathcal{O}(d \cdot \kappa)$, where $\kappa$ is the condition number.

---

## Discretization

The continuous SDE must be discretized for computation. The Euler-Maruyama scheme gives:

$$
x_{t+1} = x_t + \epsilon \, s(x_t) + \sqrt{2\epsilon} \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
$$

This introduces **discretization bias**. Two approaches address this:

1. **Unadjusted Langevin Algorithm (ULA)**: Accept the bias (see [ULA](ula.md))
2. **Metropolis-Adjusted Langevin (MALA)**: Add MH correction (see [MALA](mala.md))

### The Stability Condition

For a quadratic potential $U(x) = \frac{1}{2}x^\top H x$, stability requires $\epsilon < 2/\lambda_{\max}(H)$.

---

## Annealed Langevin Dynamics

Standard Langevin struggles with multimodal distributions. Temperature annealing introduces a schedule $T(t)$ that decreases over time:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T(t)} \, dW_t
$$

Equivalently, define noise levels $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ and run Langevin at each level:

```
Initialise x ~ N(0, σ₁² I)
For l = 1, ..., L:
    ε = α σₗ²
    For k = 1, ..., K:
        x ← x + (ε/2) s(x, σₗ) + √ε η,  η ~ N(0, I)
    σₗ → σₗ₊₁
Return x
```

This algorithm is the foundation of **score-based generative models** and **diffusion models**. For the full connection, see [Score Matching, Langevin, and Diffusion](score_and_diffusion.md).

---

## Connection to Optimization

### Gradient Descent as Zero-Temperature Limit

Setting noise to zero gives deterministic gradient ascent on $\log \pi(x)$, converging to the MAP estimate. Langevin dynamics adds noise to explore the full posterior.

### SGD Connection

Under certain conditions, SGD approximates Langevin dynamics with temperature proportional to learning rate and gradient noise variance. This explains why SGD finds "flat" minima and why learning rate schedules behave like temperature annealing.

### Connection to HMC

HMC is a **second-order** Langevin method with auxiliary momentum for ballistic rather than diffusive exploration — much more efficient in high dimensions.

---

## Preconditioning

When the target has different scales in different directions, use **preconditioned Langevin**:

$$
dx_t = M^{-1} \nabla \log \pi(x_t) \, dt + \sqrt{2} \, M^{-1/2} dW_t
$$

where $M \approx -\nabla^2 \log \pi(x^*)$ whitens the distribution.

---

## PyTorch Implementation

```python
import torch

class LangevinSDE:
    """Continuous-time Langevin dynamics via Euler-Maruyama."""
    
    def __init__(self, score_fn, dim, temperature=1.0):
        self.score_fn = score_fn
        self.dim = dim
        self.temperature = temperature
    
    def step(self, x, dt):
        score = self.score_fn(x)
        drift = score * dt
        diffusion = torch.sqrt(2 * self.temperature * torch.tensor(dt)) * torch.randn_like(x)
        return x + drift + diffusion
    
    def sample(self, x0, n_steps, dt, return_trajectory=False):
        x = x0.clone()
        if return_trajectory:
            trajectory = [x.clone()]
        for _ in range(n_steps):
            x = self.step(x, dt)
            if return_trajectory:
                trajectory.append(x.clone())
        if return_trajectory:
            return torch.stack(trajectory)
        return x


# Example: 2D Gaussian
mu = torch.tensor([2.0, -1.0])
cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
precision = torch.linalg.inv(cov)

score_fn = lambda x: -torch.matmul(x - mu, precision)
sampler = LangevinSDE(score_fn, dim=2)

x0 = torch.randn(500, 2) * 3 + torch.tensor([5.0, 5.0])
x = x0.clone()
for _ in range(1000):
    x = sampler.step(x, dt=0.1)

print(f"Sample mean: [{x[:, 0].mean():.3f}, {x[:, 1].mean():.3f}]")
print(f"True mean:   [{mu[0]:.3f}, {mu[1]:.3f}]")
```

---

## Summary

| Component | Expression | Role |
|-----------|------------|------|
| **SDE** | $dx_t = s(x_t) dt + \sqrt{2} dW_t$ | Dynamics definition |
| **Drift** | $s(x) = \nabla \log \pi(x)$ | Moves toward high probability |
| **Diffusion** | $\sqrt{2} dW_t$ | Enables exploration |
| **Stationary distribution** | $\pi(x)$ | Samples converge to this |
| **Fokker-Planck** | $\partial_t \rho = -\nabla \cdot (\rho s) + \Delta \rho$ | Density evolution |
| **Temperature** | Scales noise; enables annealing | |
| **Zero noise** | Recovers gradient ascent (MAP) | |

---

## Exercises

1. **Fokker-Planck verification.** Starting from $\rho = \pi \propto \exp(-U)$, verify that the Fokker-Planck right-hand side vanishes.

2. **Temperature effects.** Modify the implementation to explore different temperatures. For a bimodal distribution, show how low temperature concentrates samples at the global mode and high temperature allows exploration of both modes.

3. **ULA bias.** For a 1-D Gaussian $\pi(x) = \mathcal{N}(0, 1)$, derive the stationary distribution of ULA as a function of $\epsilon$ and show it is biased for $\epsilon > 0$.

4. **Annealing experiment.** Implement annealed Langevin sampling for a 2-D Gaussian mixture with well-separated modes. Compare convergence with and without annealing.

5. **Convergence rate.** For a 1D Gaussian with variance $\sigma^2$, derive the convergence rate of the sample variance to the true variance as a function of time and step size.

---

## References

1. Langevin, P. (1908). Sur la théorie du mouvement brownien. *Comptes Rendus de l'Académie des Sciences*, 146, 530-533.
2. Gardiner, C. W. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*. Springer.
3. Pavliotis, G. A. (2014). *Stochastic Processes and Applications*. Springer.
4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
5. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
6. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.
