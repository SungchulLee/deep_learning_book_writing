# Hamiltonian Monte Carlo

Hamiltonian Monte Carlo (HMC) is one of the most powerful MCMC algorithms for sampling from continuous distributions. By borrowing ideas from classical mechanics, HMC achieves dramatically better exploration than random-walk methods, especially in high dimensions. This section develops the complete theory: Hamiltonian dynamics, phase space geometry, the leapfrog integrator, geometric interpretations, energy conservation, and why HMC can take multiple steps while Langevin cannot.

---

## Why HMC Matters

### The Random-Walk Problem

Standard Metropolis-Hastings proposes new states via random perturbations. In $d$ dimensions, the optimal proposal scale shrinks as $O(d^{-1/2})$, yielding distance traveled per step of $O(d^{-1/2})$. After $N$ steps, the chain has moved only $O(\sqrt{N} \cdot d^{-1/2})$ from its starting point—**diffusive** exploration that slows dramatically with dimension.

For Bayesian posterior inference in quantitative finance—where models routinely have hundreds of correlated parameters (stochastic volatility surfaces, multi-factor term structure models, hierarchical portfolio models)—random-walk MCMC becomes computationally infeasible.

### HMC's Solution: Physics-Guided Proposals

HMC introduces **auxiliary momentum variables** and simulates Hamiltonian dynamics from classical mechanics. The key insight is that by treating the negative log-posterior as a potential energy landscape, we can use deterministic physics to propose distant, high-probability states:

$$
H(\mathbf{x}, \mathbf{v}) = \underbrace{U(\mathbf{x})}_{\text{potential}} + \underbrace{K(\mathbf{v})}_{\text{kinetic}} = -\log \tilde{p}(\mathbf{x}) + \frac{1}{2}\mathbf{v}^\top \mathbf{M}^{-1}\mathbf{v}
$$

The resulting **ballistic** motion covers distance $\propto L\epsilon$ (vs $\propto \sqrt{L\epsilon}$ for random walks), enabling efficient exploration of complex posteriors.

---

## The Hamiltonian Framework

### Augmenting the State Space

HMC introduces **auxiliary momentum variables** $\mathbf{v}$ and defines the Hamiltonian:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})
$$

where:

- **Potential energy**: $U(\mathbf{x}) = -\log \tilde{p}(\mathbf{x})$ (negative log-density of the target)
- **Kinetic energy**: $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^\top \mathbf{M}^{-1}\mathbf{v}$ (quadratic in momentum)
- **Mass matrix**: $\mathbf{M}$ is a positive definite matrix (often diagonal or the identity)

### The Joint Distribution

The joint distribution on $(\mathbf{x}, \mathbf{v})$ is:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v})) = \exp(-U(\mathbf{x})) \exp(-K(\mathbf{v}))
$$

This factorizes:

$$
\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})
$$

**Key property**: Marginalizing over $\mathbf{v}$ recovers the target $\pi(\mathbf{x})$. The momentum variables are purely auxiliary—they enable the dynamics but carry no information about the target.

### Hamilton's Equations

The dynamics are governed by Hamilton's equations:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = \nabla \log p(\mathbf{x})
$$

These equations describe a particle moving on a surface defined by the potential energy, with the **score function** $\nabla \log p(\mathbf{x})$ acting as a force. High-probability regions exert an attractive force; the particle accelerates toward them but overshoots due to momentum.

---

## Energy Conservation: Why HMC Samples Instead of Optimizes

### The Central Mystery

HMC follows gradients like gradient ascent and explores high-probability regions. Why doesn't it just converge to the mode?

**Answer**: Energy conservation creates oscillatory trajectories that prevent convergence.

### What Energy Conservation Means

For exact (continuous-time) Hamiltonian dynamics:

$$
\frac{d}{dt}H(\mathbf{x}_t, \mathbf{v}_t) = 0
$$

**Proof**:
$$
\frac{dH}{dt} = \frac{\partial H}{\partial \mathbf{x}}\frac{d\mathbf{x}}{dt} + \frac{\partial H}{\partial \mathbf{v}}\frac{d\mathbf{v}}{dt} = \nabla U \cdot \mathbf{M}^{-1}\mathbf{v} + \mathbf{M}^{-1}\mathbf{v} \cdot (-\nabla U) = 0
$$

The cross terms cancel exactly due to the antisymmetric structure of Hamilton's equations.

### The Oscillation Mechanism

**Without energy conservation** (gradient ascent):

1. Start anywhere with $\nabla \log p \neq 0$
2. Follow gradient uphill
3. Reach mode where $\nabla \log p = 0$
4. **Stop** — no force to continue

Result: **Optimization**, not sampling.

**With energy conservation**:

Energy conservation requires $U(\mathbf{x}_t) + K(\mathbf{v}_t) = H_0 = \text{constant}$.

- **Climbing toward mode** (lower $U$): $K$ increases → particle speeds up
- **At the mode**: $U$ is minimal → $K$ is **maximal** → particle moves fastest
- **Past the mode**: $U$ increases → $K$ decreases → particle slows and turns around
- **Result**: Oscillation through the mode

The particle cannot stop at the mode because it has residual kinetic energy.

### Time Allocation and Correct Sampling

Energy conservation creates the correct time allocation. Adding momentum plus a conservation law transforms optimization into sampling:

| System | Energy | Behavior | Outcome |
|--------|--------|----------|---------|
| Gradient ascent | $E(\mathbf{x}) = -\log p(\mathbf{x})$ | Minimize $E$ | Converge to mode |
| Hamiltonian | $H(\mathbf{x},\mathbf{v}) = U + K$ | Conserve $H$ | Oscillate, sample |

---

## Why HMC Takes Multiple Steps but Langevin Cannot

### The Fundamental Difference

- **HMC**: Runs $L$ leapfrog steps **before** MH correction
- **Langevin (MALA)**: Must apply MH correction **after every step**

This is a fundamental consequence of the underlying dynamics.

### HMC: Energy Conservation Enables Chaining

After $L$ leapfrog steps, energy error is bounded: $|\Delta H| = O(L\epsilon^3) = O(T\epsilon^2)$ where $T = L\epsilon$ is total integration time.

For moderate $L$ and small $\epsilon$, acceptance remains high (typically 65–95%).

### Why Langevin Cannot Chain Steps

Langevin dynamics has **no conserved quantity**:

$$
d\mathbf{x}_t = \nabla \log p(\mathbf{x}_t) dt + \sqrt{2} dW_t
$$

Each step adds **independent** Gaussian noise. After $L$ steps, the proposal distribution is:

$$
q(\mathbf{x}^{(L)} | \mathbf{x}^{(0)}) = \int q(\mathbf{x}^{(1)} | \mathbf{x}^{(0)}) \cdots q(\mathbf{x}^{(L)} | \mathbf{x}^{(L-1)}) \, d\mathbf{x}^{(1)} \cdots d\mathbf{x}^{(L-1)}
$$

This **path integral** is intractable to compute, so we cannot evaluate the MH ratio after $L$ steps.

### Deterministic vs Stochastic Trajectories

**HMC**: Trajectory is deterministic given $(\mathbf{x}, \mathbf{v})$. The proposal is a delta function, and reversibility + symplecticity give:

$$
\alpha = \min(1, e^{-\Delta H})
$$

No path integral needed.

**Langevin**: Trajectory is stochastic. Computing the proposal probability requires marginalizing over all intermediate states—intractable.

### Coherent vs Diffusive Motion

| Property | Langevin | HMC |
|----------|----------|-----|
| Dynamics | First-order (dissipative) | Second-order (Hamiltonian) |
| Conserved quantity | None | Energy $H$ |
| Trajectory | Stochastic | Deterministic |
| Exploration | Diffusive ($\sqrt{L\epsilon}$) | Ballistic ($L\epsilon$) |
| Step chaining | ✗ (intractable) | ✓ (energy conservation) |

---

## HMC Preserves the Target Distribution

### Three Mechanisms Working Together

1. **Momentum resampling**: Creates the joint distribution $\pi(\mathbf{x}, \mathbf{v})$
2. **Hamiltonian dynamics**: Preserves $\pi(\mathbf{x}, \mathbf{v})$ via energy conservation + volume preservation
3. **MH correction**: Corrects for discretization errors

### Why Each Step Preserves $\pi(\mathbf{x}, \mathbf{v})$

**Momentum resampling**: Given $\mathbf{x}^{(t)} \sim \pi(\mathbf{x})$, sample $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$. The pair $(\mathbf{x}^{(t)}, \mathbf{v}) \sim \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M}) = \pi(\mathbf{x}, \mathbf{v})$ ✓

**Hamiltonian dynamics** (exact): Energy conservation ($H$ constant) + volume preservation ($|\det \mathbf{J}| = 1$) imply $\pi(\mathbf{x}_t, \mathbf{v}_t) = \exp(-H(\mathbf{x}_t, \mathbf{v}_t)) = \exp(-H(\mathbf{x}_0, \mathbf{v}_0)) = \pi(\mathbf{x}_0, \mathbf{v}_0)$ ✓

**MH correction**: Symplecticity + reversibility ensure detailed balance. Acceptance probability $\alpha = \min(1, e^{-\Delta H})$ corrects for discretization error ✓

---

## Practical Considerations

### Hyperparameter Selection

**Step size $\epsilon$**: Too large causes energy error growth and low acceptance; too small causes slow exploration. Target acceptance rate: 65–80%.

**Number of steps $L$**: Too few gives random-walk behavior; too many wastes computation as the trajectory turns back. Rule of thumb: $L\epsilon \approx$ trajectory length covering the typical set.

**Mass matrix $\mathbf{M}$**: Identity assumes isotropic distribution (rarely true). Diagonal adapts to marginal variances (practical default). Full covariance is optimal but expensive ($O(d^3)$). Best practice: $\mathbf{M} \approx \text{Cov}[\mathbf{x}]$ estimated from warmup.

### When HMC Struggles

- **Multimodal distributions**: Energy barriers prevent mode hopping
- **Varying curvature**: Single $\mathbf{M}$ insufficient
- **Discrete variables**: No gradients available
- **Heavy tails**: Trajectories can escape to infinity

**Remedies**: Tempering/annealing for multimodality, Riemannian HMC for varying curvature, discontinuous HMC for mixed discrete-continuous.

---

## Comparison with Other Methods

| Method | Uses Gradient? | Uses Curvature? | Energy-Preserving? | Result |
|--------|---------------|-----------------|-------------------|--------|
| Random walk MH | No | No | No | Slow exploration |
| Gradient descent | Yes | No | No | Optimization |
| Newton's method | Yes | Yes | No | Fast optimization |
| Langevin (MALA) | Yes | No | No (dissipative) | Sampling (diffusive) |
| HMC | Yes | Via $\mathbf{M}$ | Yes | Efficient sampling |

HMC uniquely combines gradient information with energy-preserving dynamics to achieve efficient sampling in high dimensions.

---

## Application to Quantitative Finance

### Bayesian Posterior Inference

HMC is the backbone of modern Bayesian inference for complex financial models:

- **Stochastic volatility models**: The Heston model posterior $p(\kappa, \theta, \sigma_v, \rho, v_{0:T} | y_{1:T})$ involves hundreds of latent volatility states plus 4–5 structural parameters. HMC efficiently explores this high-dimensional, correlated posterior.
- **Term structure models**: Multi-factor affine models with latent state variables and observation parameters. The strong correlations between factor loadings make random-walk MCMC impractical.
- **Bayesian neural networks for risk**: Posterior inference over network weights for uncertainty quantification in credit risk or tail risk estimation.

### Why Gradients Are Available

Automatic differentiation (PyTorch, JAX, Stan) computes exact gradients of the log-posterior with respect to all continuous parameters. This makes HMC applicable whenever the likelihood and prior are differentiable—which covers the vast majority of parametric financial models.

### Practical Workflow

Modern probabilistic programming frameworks (Stan, PyMC, NumPyro) implement HMC with NUTS as the default sampler, handling step size adaptation, mass matrix estimation, and diagnostics automatically. For most quantitative finance applications, the practitioner specifies the model and the framework handles the sampling.

---

## Summary

| Component | Role | Key Property |
|-----------|------|--------------|
| Momentum $\mathbf{v}$ | Auxiliary variable | Creates joint $\pi(\mathbf{x}, \mathbf{v})$ |
| Hamiltonian $H$ | Total energy | $H = U + K$ |
| Leapfrog | Numerical integrator | Symplectic + reversible |
| Energy conservation | Prevents optimization | Creates oscillations |
| MH correction | Fixes discretization | Ensures exact sampling |

**Why HMC works**:

1. **Gradient information** guides exploration toward high-probability regions
2. **Energy conservation** prevents convergence to mode, enables oscillation
3. **Coherent momentum** enables ballistic (not diffusive) exploration
4. **Symplectic integration** preserves phase space volume
5. **MH correction** ensures exact sampling despite discretization

HMC sits at the intersection of statistical mechanics, differential geometry, and optimization—a synthesis that makes it the gold standard for sampling from continuous distributions in high dimensions.
