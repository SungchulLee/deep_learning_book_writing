# Langevin Dynamics

Langevin dynamics provides the continuous-time foundation for gradient-based MCMC methods. This section develops the theory from physical principles through modern applications, culminating in the deep connection between MCMC sampling and diffusion-based generative models.

## Physical Origins: Brownian Motion and the Langevin Equation

### Historical Context

In 1908, Paul Langevin proposed an equation to describe the motion of a particle suspended in a fluid, subject to both deterministic forces and random collisions with surrounding molecules. This equation unified two perspectives: the macroscopic drift from potential energy and the microscopic fluctuations from thermal noise.

### The Classical Langevin Equation

For a particle with position $x$ and mass $m$ in a potential $U(x)$:

$$
m\frac{d^2 x}{dt^2} = -\gamma \frac{dx}{dt} - \nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

where $\gamma$ is the friction coefficient, $k_B T$ is thermal energy, and $\xi(t)$ is white noise satisfying $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$.

The three terms represent:
- **Friction**: $-\gamma \dot{x}$ opposes motion
- **Force from potential**: $-\nabla U(x)$ drives toward energy minima
- **Thermal fluctuations**: Random kicks from molecular collisions

### The Overdamped Limit

In the **high-friction regime** where inertia is negligible ($m \to 0$), the acceleration term vanishes and we obtain the **overdamped Langevin equation**:

$$
\gamma \frac{dx}{dt} = -\nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

Rescaling time by $\gamma$ and setting $k_B T = 1$:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

This is the **Langevin stochastic differential equation (SDE)** used in MCMC.

## The Score Function: Gradient of Log-Density

### Definition and Properties

The **score function** is the gradient of the log-density:

$$
s(x) = \nabla_x \log p(x)
$$

For a Boltzmann distribution $p(x) \propto \exp(-U(x))$:

$$
s(x) = \nabla_x \log p(x) = -\nabla U(x)
$$

The score points toward regions of **higher probability**:
- At modes: $s(x) = 0$ (gradient vanishes at maxima of $\log p$)
- Away from modes: $s(x)$ points toward the nearest mode
- Magnitude reflects steepness of the log-density landscape

### Independence from Normalization

A crucial property: the score doesn't depend on the normalization constant:

$$
s(x) = \nabla_x \log p(x) = \nabla_x \log \frac{\tilde{p}(x)}{Z} = \nabla_x \log \tilde{p}(x) - \nabla_x \log Z = \nabla_x \log \tilde{p}(x)
$$

Since $Z$ is constant, $\nabla_x \log Z = 0$. This is why gradient-based MCMC methods inherit the ratio trick from Metropolis-Hastings—they only need the unnormalized density.

### Score for Common Distributions

**Gaussian** $p(x) = \mathcal{N}(\mu, \Sigma)$:
$$
s(x) = -\Sigma^{-1}(x - \mu)
$$

The score points toward the mean, with strength determined by the precision matrix.

**Gaussian Mixture** $p(x) = \sum_k \pi_k \mathcal{N}(\mu_k, \Sigma_k)$:
$$
s(x) = \frac{\sum_k \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) \cdot (-\Sigma_k^{-1}(x - \mu_k))}{\sum_k \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)}
$$

A weighted average of component scores—the score points toward the nearest mode(s).

## Langevin Dynamics as an SDE

### The Sampling SDE

To sample from $\pi(x) \propto \exp(-U(x))$, we simulate:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t = s(x_t) \, dt + \sqrt{2} \, dW_t
$$

where $W_t$ is standard Brownian motion.

**Interpretation**:
- **Drift term** $s(x_t) \, dt$: Deterministic flow toward high-probability regions
- **Diffusion term** $\sqrt{2} \, dW_t$: Random exploration via Brownian motion

### Convergence to the Target Distribution

**Theorem** (Langevin convergence): Under mild regularity conditions on $U$, the distribution of $x_t$ converges to $\pi(x)$ as $t \to \infty$.

The key insight is that Langevin dynamics satisfies **detailed balance** in continuous time. The Fokker-Planck equation for the density $\rho(x, t)$ of $x_t$ is:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \, s) + \Delta \rho = \nabla \cdot (\rho \nabla U) + \Delta \rho
$$

At stationarity ($\partial \rho / \partial t = 0$), the solution is $\rho = \pi \propto \exp(-U)$.

### The Fokker-Planck Perspective

The Fokker-Planck (or forward Kolmogorov) equation describes how probability density evolves under an SDE. For general dynamics:

$$
dx_t = f(x_t) \, dt + g(x_t) \, dW_t
$$

the density evolves according to:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (f \rho) + \frac{1}{2} \nabla \cdot (g^2 \nabla \rho)
$$

For Langevin dynamics with $f = s = \nabla \log \pi$ and $g = \sqrt{2}$:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \nabla \log \pi) + \Delta \rho
$$

This can be rewritten as:

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot \left( \pi \nabla \left( \frac{\rho}{\pi} \right) \right)
$$

which shows that $\rho = \pi$ is indeed the unique stationary solution.

## Discretization: From SDE to Algorithm

### Euler-Maruyama Discretization

The simplest discretization of the Langevin SDE uses Euler-Maruyama:

$$
x_{t+1} = x_t + \epsilon \, s(x_t) + \sqrt{2\epsilon} \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
$$

where $\epsilon$ is the step size (discrete time increment).

**This is Unadjusted Langevin Algorithm (ULA)**—it has discretization bias but is simple to implement.

### Metropolis-Adjusted Langevin Algorithm (MALA)

To correct for discretization error, add a Metropolis-Hastings acceptance step:

1. **Propose**: $x' = x + \epsilon \, s(x) + \sqrt{2\epsilon} \, \eta$
2. **Accept** with probability:
$$
\alpha = \min\left(1, \frac{\pi(x') q(x | x')}{\pi(x) q(x' | x)}\right)
$$

where the proposal density is:
$$
q(x' | x) = \mathcal{N}\left(x' \,\Big|\, x + \epsilon \, s(x), 2\epsilon I\right)
$$

**Note**: The proposal is **asymmetric** because the drift depends on the current position. The Hastings correction is essential.

### Comparing ULA and MALA

| Property | ULA | MALA |
|----------|-----|------|
| Bias | Has discretization bias | Exact (in equilibrium) |
| Acceptance step | No | Yes |
| Computational cost | Lower | Higher (two score evaluations) |
| Optimal step size | $\epsilon \sim d^{-1/3}$ | $\epsilon \sim d^{-1/6}$ |
| Mixing time | $\mathcal{O}(d^{5/3})$ | $\mathcal{O}(d^{5/3})$ |

For large step sizes, ULA may not converge to the correct distribution, while MALA always does (given ergodicity).

## Temperature and the Gibbs Distribution

### Adding Temperature

At **temperature** $T$, the target distribution becomes:

$$
\pi_T(x) \propto \exp\left(-\frac{U(x)}{T}\right)
$$

The Langevin dynamics at temperature $T$ is:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T} \, dW_t
$$

**Key insight**: Temperature scales the **noise**, not the gradient.

### Temperature Regimes

| Temperature | Noise Level | Behavior |
|-------------|-------------|----------|
| High $T$ | Large $\sqrt{2T}$ | Wide exploration, modes blurred together |
| $T = 1$ | $\sqrt{2}$ | Standard sampling from $\pi$ |
| Low $T$ | Small $\sqrt{2T}$ | Concentrates near modes |
| $T \to 0$ | No noise | Deterministic gradient descent to minimum |

### The Energy-Entropy Trade-off

The Gibbs distribution balances energy and entropy:

$$
\pi_T(x) = \frac{1}{Z_T} \exp\left(-\frac{U(x)}{T}\right)
$$

- **Low $T$**: Energy dominates → samples concentrate at global minimum
- **High $T$**: Entropy dominates → samples spread uniformly
- **Intermediate $T$**: Balance between low energy and high entropy

This trade-off is fundamental to understanding both MCMC mixing and optimization.

## Annealed Langevin Dynamics

### The Multimodality Problem

Standard Langevin dynamics struggles with **multimodal** distributions:
- The score points toward the **nearest** mode
- Transitions between well-separated modes are exponentially rare
- In high dimensions, the chain may be effectively trapped forever

### Annealing: Interpolating Between Easy and Hard

The solution is to introduce a **temperature schedule** $T(t)$ that decreases over time:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T(t)} \, dW_t
$$

**Strategy**:
1. Start at high temperature: distribution is nearly uniform, easy to sample
2. Gradually cool: track the concentrating distribution
3. End at $T = 1$: arrive at samples from $\pi$

This is the **continuous-time analog of simulated annealing**, but for sampling rather than optimization.

### Noise-Level Annealing (Score-Based View)

Equivalently, consider a sequence of **noise levels** $\sigma_1 > \sigma_2 > \cdots > \sigma_T$. For each $\sigma_t$, define the **noised distribution**:

$$
p_{\sigma_t}(x) = \int p_{\text{data}}(x_0) \cdot \mathcal{N}(x | x_0, \sigma_t^2 I) \, dx_0
$$

This is the data distribution convolved with Gaussian noise of variance $\sigma_t^2$.

**Properties**:
- **Large noise** ($\sigma_1$ large): $p_{\sigma_1}(x) \approx \mathcal{N}(0, \sigma_1^2 I)$, multimodal structure washed out
- **Small noise** ($\sigma_T$ small): $p_{\sigma_T}(x) \approx p_{\text{data}}(x)$, original structure preserved

### The Annealed Langevin Algorithm

```
Initialize: x ~ N(0, σ₁²I)  # Start from high-noise distribution

For t = 1, 2, ..., T:
    # Run Langevin at noise level σₜ
    For k = 1, ..., K:
        x ← x + ε·s(x, σₜ) + √(2ε)·η,  η ~ N(0, I)
    
    # Decrease noise level
    σₜ → σₜ₊₁

Return x  # Sample from p_data
```

The key requirement is a **noise-conditional score** $s(x, \sigma) = \nabla_x \log p_\sigma(x)$ that can be evaluated at any noise level.

## Connection to Diffusion Models

### The Forward Diffusion Process

**Denoising Diffusion Probabilistic Models (DDPMs)** define a forward process that gradually adds noise:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t | \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)
$$

Starting from data $x_0 \sim p_{\text{data}}$, this produces increasingly noisy versions until $x_T \approx \mathcal{N}(0, I)$.

In continuous time, this becomes the **forward SDE**:

$$
dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

This is an **Ornstein-Uhlenbeck process**—a drift toward the origin plus diffusion.

### The Reverse Process: Time-Reversed SDE

**Anderson's Theorem (1982)**: For any forward SDE

$$
dx_t = f(x_t, t) \, dt + g(t) \, dW_t
$$

the time-reversed process is:

$$
dx_t = \left[ f(x_t, t) - g^2(t) \nabla_x \log p_t(x_t) \right] dt + g(t) \, d\bar{W}_t
$$

where $\bar{W}_t$ is a backward Brownian motion and $p_t$ is the marginal distribution at time $t$.

For the DDPM forward process, the reverse is:

$$
dx_t = \left[ -\frac{1}{2}\beta(t) x_t - \beta(t) \nabla_x \log p_t(x_t) \right] dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

**The score** $\nabla_x \log p_t(x_t)$ is the key quantity—it determines how to reverse the diffusion!

### Score Matching: Learning the Score from Data

To run the reverse process, we need $\nabla_x \log p_t(x)$ at all noise levels. **Score matching** learns this from data.

**Naive objective** (intractable):
$$
\mathcal{L}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right]
$$

**Denoising score matching** (tractable): Add noise to data and learn the score of the noised distribution:

$$
\tilde{x} = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The score of the conditional distribution $q(\tilde{x} | x_0) = \mathcal{N}(\tilde{x} | x_0, \sigma^2 I)$ is:

$$
\nabla_{\tilde{x}} \log q(\tilde{x} | x_0) = -\frac{\tilde{x} - x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

**Denoising objective**:
$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, \epsilon, \sigma}\left[\left\|s_\theta(\tilde{x}, \sigma) + \frac{\epsilon}{\sigma}\right\|^2\right]
$$

This is equivalent to training a network to **predict the noise**—exactly what DDPM does!

### The Unified Picture

| Perspective | Forward | Backward | What's Learned |
|-------------|---------|----------|----------------|
| **Langevin MCMC** | — | $dx = s(x) \, dt + \sqrt{2} \, dW$ | Score $s(x)$ from known density |
| **Annealed Langevin** | — | Multi-scale Langevin | Score $s(x, \sigma)$ at multiple noise levels |
| **Score-based models** | Add noise | Denoise via score | Score network $s_\theta(x, \sigma)$ |
| **DDPM** | Forward diffusion $x_0 \to x_T$ | Reverse diffusion $x_T \to x_0$ | Noise predictor $\epsilon_\theta(x_t, t)$ |

All are manifestations of the same principle: **use the score to guide stochastic dynamics toward the target distribution**.

### The MCMC–Diffusion Analogy

| MCMC / Langevin | Diffusion Models |
|-----------------|------------------|
| Target distribution $\pi(x)$ | Data distribution $p_{\text{data}}(x)$ |
| Score $\nabla \log \pi$ | Time-dependent score $\nabla \log p_t$ |
| Temperature $T$ | Noise level $\sigma$ or schedule $\beta(t)$ |
| Simulated annealing | Annealed Langevin / reverse diffusion |
| High $T$: explore | High $\sigma$: diffuse |
| Low $T$: exploit | Low $\sigma$: denoise |
| MALA (discrete) | DDPM (discrete time) |
| Langevin SDE (continuous) | Score SDE (continuous time) |

## Variance Exploding vs. Variance Preserving SDEs

Two main formulations exist for diffusion models:

### Variance Exploding (VE-SDE)

$$
dx_t = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, dW_t
$$

Pure diffusion with increasing variance. The noise level $\sigma(t)$ grows over time.

### Variance Preserving (VP-SDE)

$$
dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

Drift toward origin plus diffusion, with variance approximately preserved.

**Both are equivalent** up to reparametrization—they define the same family of marginal distributions $\{p_t\}$.

## Accelerating Langevin: From SDE to ODE

### The Probability Flow ODE

Song et al. (2021) showed that any SDE has a corresponding **deterministic** ODE with the same marginal distributions:

$$
dx_t = \left[ f(x_t, t) - \frac{1}{2} g^2(t) \nabla_x \log p_t(x_t) \right] dt
$$

For Langevin dynamics, this becomes:

$$
dx_t = \frac{1}{2} s(x_t) \, dt
$$

**Advantages**:
- Deterministic: same initial condition → same trajectory
- Can use ODE solvers (Runge-Kutta, etc.) instead of SDE solvers
- Often faster convergence

**Disadvantage**: Loses the exploration benefit of stochasticity.

### DDIM: Deterministic Sampling for Diffusion

**Denoising Diffusion Implicit Models (DDIM)** use the probability flow ODE for deterministic sampling, enabling:
- Interpolation in latent space
- Fewer sampling steps (10-50 vs. 1000)
- Consistent outputs for the same noise seed

## Practical Considerations

### Step Size Selection

For Langevin-based methods:

| Method | Optimal Step Size | Acceptance Target |
|--------|-------------------|-------------------|
| ULA | $\epsilon \sim d^{-1/3}$ | N/A (no acceptance step) |
| MALA | $\epsilon \sim d^{-1/6}$ | ~57.4% |
| HMC | $\epsilon \sim d^{-1/4}$ | ~65% |

### Preconditioning

When the target has different scales in different directions, use preconditioned Langevin:

$$
dx_t = M^{-1} \nabla \log \pi(x_t) \, dt + \sqrt{2} M^{-1/2} dW_t
$$

where $M \approx -\nabla^2 \log \pi(x^*)$ (Hessian at the mode). This is equivalent to running standard Langevin on a whitened distribution.

### Gradient Estimation

When the score requires computing expectations (e.g., in latent variable models), stochastic gradient Langevin dynamics (SGLD) uses noisy gradient estimates:

$$
x_{t+1} = x_t + \epsilon \, \hat{s}(x_t) + \sqrt{2\epsilon} \, \eta_t
$$

where $\hat{s}$ is a stochastic approximation to $s$. Under appropriate conditions, this still converges to $\pi$.

## Why This Connection Matters

### Theoretical Transfer

The same mathematics governs both MCMC and generative modeling:
- **Fokker-Planck equation**: Describes density evolution
- **Detailed balance / time reversal**: Ensures correct stationary distribution
- **Score functions**: The common language

Understanding one illuminates the other.

### Algorithmic Cross-Pollination

Techniques developed for MCMC accelerate diffusion models:
- Preconditioning → adaptive noise schedules
- Tempering / annealing → noise level annealing
- Higher-order integrators → better ODE solvers for sampling

Conversely, diffusion model innovations (learned schedules, distillation) may improve MCMC.

### The Deep Insight

**Score-based diffusion models are annealed Langevin MCMC with learned energy functions.**

The breakthrough was realizing:
1. Scores can be learned via denoising (no need for explicit density)
2. Annealing solves multimodality (inherited from simulated annealing)
3. The same Fokker-Planck theory explains convergence

Modern image generators (DALL-E, Stable Diffusion, Midjourney) are running annealed Langevin dynamics with scores learned from billions of images. The theory developed for Bayesian inference now powers the generative AI revolution.

## Summary

Langevin dynamics bridges physics, MCMC, and modern generative modeling:

| Aspect | Description |
|--------|-------------|
| **Physical origin** | Overdamped motion in a potential with thermal noise |
| **Mathematical form** | $dx_t = \nabla \log \pi(x_t) \, dt + \sqrt{2} \, dW_t$ |
| **Key quantity** | Score function $s(x) = \nabla \log \pi(x)$ |
| **Discretization** | ULA (biased) or MALA (exact with MH correction) |
| **Temperature** | Scales noise, enables annealing |
| **Connection to diffusion** | Reverse diffusion = annealed Langevin with learned score |

The unifying principle across all these methods: **use the gradient of log-density to guide stochastic dynamics toward the target distribution**. Whether sampling from a Bayesian posterior or generating images from noise, the mathematics is the same.
