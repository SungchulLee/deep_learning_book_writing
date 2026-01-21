# Simulated Annealing and Temperature Methods

Temperature is a fundamental control parameter that appears throughout probabilistic machine learning—from MCMC sampling to optimization to modern generative models. This section develops the theory of temperature-based methods, with simulated annealing as the central example, and reveals the deep connections that unify these seemingly disparate techniques.

## The Boltzmann Distribution: Foundation of Temperature Methods

### Statistical Mechanics Origin

In statistical mechanics, the probability of a system being in state $x$ with energy $E(x)$ at temperature $T$ is given by the **Boltzmann distribution**:

$$
p_T(x) = \frac{1}{Z_T} \exp\left(-\frac{E(x)}{T}\right)
$$

where $Z_T = \int \exp(-E(x)/T) \, dx$ is the **partition function** (normalization constant).

**Inverse temperature notation**: Setting $\beta = 1/T$:
$$
p_\beta(x) = \frac{1}{Z_\beta} \exp(-\beta E(x))
$$

### Physical Interpretation

At **absolute zero** ($T \to 0$, $\beta \to \infty$):
- System occupies minimum energy state
- All probability mass at global minimum
- Zero thermal fluctuations
- Distribution becomes a Dirac delta at $\arg\min E(x)$

At **high temperature** ($T \to \infty$, $\beta \to 0$):
- All energy states equally likely
- Maximum disorder (entropy)
- Distribution approaches uniform
- Pure thermal noise dominates

**Temperature controls the trade-off between energy and entropy.** This principle underlies all temperature-based methods in machine learning.

### Free Energy and the Energy-Entropy Trade-off

The **Helmholtz free energy** at temperature $T$ is:

$$
F_T = -T \log Z_T = \langle E \rangle_T - T S_T
$$

where $\langle E \rangle_T = \mathbb{E}_{p_T}[E(x)]$ is the expected energy and $S_T = -\mathbb{E}_{p_T}[\log p_T(x)]$ is the entropy.

This decomposition reveals:
- **Low $T$**: Free energy $\approx$ minimum energy (entropy negligible)
- **High $T$**: Free energy dominated by entropy (energy less important)

Minimizing free energy at different temperatures traces a path through the energy landscape—the foundation of annealing methods.

## Simulated Annealing: Non-Stationary MCMC for Optimization

### The Key Distinction

Simulated annealing is **not** standard MCMC sampling—it's a **non-stationary** variant of Metropolis-Hastings designed for **optimization**, not sampling.

| Property | Standard MCMC | Simulated Annealing |
|----------|---------------|---------------------|
| Target | Fixed $\pi(x)$ | Time-varying $\pi_{T(t)}(x)$ |
| Goal | Sample from $\pi$ | Find $\arg\max \pi$ (or $\arg\min E$) |
| Stationary? | Yes | No |
| Convergence | To distribution | To point (global optimum) |
| Output | Samples for inference | Single optimal solution |

### The Algorithm

At iteration $t$, the target distribution is:

$$
\pi_{T(t)}(x) = \frac{1}{Z_{T(t)}} \exp\left(-\frac{E(x)}{T(t)}\right)
$$

where $T(t)$ is a **cooling schedule** with $T(t) \to 0$ as $t \to \infty$.

**Simulated Annealing Algorithm**:

```python
def simulated_annealing(E, x0, T_schedule, proposal, max_iter):
    x = x0
    E_best, x_best = E(x), x
    
    for t in range(max_iter):
        T = T_schedule(t)
        x_prop = proposal(x)
        delta_E = E(x_prop) - E(x)
        
        # Metropolis acceptance at temperature T
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            x = x_prop
            if E(x) < E_best:
                x_best, E_best = x, E(x)
    
    return x_best, E_best
```

### Why Temperature Enables Global Optimization

**High Temperature (Exploration)**:

When $T$ is large, even uphill moves ($\Delta E > 0$) are accepted with high probability:

$$
\Pr(\text{accept uphill move}) = \exp\left(-\frac{\Delta E}{T}\right) \approx 1 - \frac{\Delta E}{T} + O(T^{-2})
$$

This allows the algorithm to **escape local minima** and explore the global landscape.

**Low Temperature (Exploitation)**:

When $T$ is small:

$$
\alpha = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right) \approx \begin{cases}
1 & \text{if } \Delta E < 0 \\
0 & \text{if } \Delta E > 0
\end{cases}
$$

The algorithm becomes **greedy descent**, accepting only improvements. It converges to whichever local minimum it currently occupies.

**Annealing**: By gradually decreasing $T$:
1. **High $T$**: Explore widely, find the basin of the global minimum
2. **Moderate $T$**: Transition between nearby basins, refine location
3. **Low $T$**: Settle into the deepest accessible basin

If cooling is sufficiently slow, we end at the **global** minimum.

### Convergence Theory

**Geman & Geman (1984)** proved that if the cooling schedule satisfies:

$$
T(t) \geq \frac{c}{\log(t + t_0)}
$$

for sufficiently large constant $c$, then $\lim_{t \to \infty} \Pr(x_t \in \mathcal{X}^*) = 1$, where $\mathcal{X}^* = \arg\min_x E(x)$.

The constant $c$ must satisfy $c > \Delta E_{\max}$, where $\Delta E_{\max}$ is the largest energy barrier between basins.

**The catch**: This logarithmic schedule is **impractically slow**. To reach temperature $\epsilon$ requires $t \approx \exp(c/\epsilon)$ iterations.

### Practical Cooling Schedules

Since logarithmic cooling is too slow, practical implementations use faster schedules:

**Exponential** (most common):
$$
T(t) = T_0 \cdot \alpha^t, \quad \alpha \in (0.8, 0.99)
$$

**Linear**:
$$
T(t) = T_0 - \beta t, \quad T(t) \geq T_{\min}
$$

**Inverse**:
$$
T(t) = \frac{T_0}{1 + \beta t}
$$

**Trade-off**: Faster cooling risks getting trapped in local minima, but enables practical computation.

### Comparison with Gradient Descent

**Gradient descent** is purely deterministic:
$$
x_{t+1} = x_t - \eta \nabla E(x_t)
$$

It's fast but **trapped** in the first local minimum encountered.

**Simulated annealing** uses controlled randomness:
- Early iterations: high randomness enables global exploration
- Late iterations: low randomness ensures local convergence

This is the **exploration-exploitation trade-off** controlled by temperature.

### When Simulated Annealing Fails

**Too-fast cooling**: System gets trapped in local minimum before finding the global basin.

**Phase transitions**: At critical temperature $T_c$, the optimal basin may switch discontinuously. Cooling too quickly through $T_c$ can trap the algorithm in the wrong basin.

**Example**: In the traveling salesman problem, small temperature changes near critical points can dramatically alter the optimal tour structure.

### Applications

**Combinatorial optimization**:
- Traveling salesman problem (state: tour permutation, energy: total distance)
- Graph coloring (state: color assignment, energy: number of conflicts)
- Circuit design, scheduling, resource allocation

**Continuous optimization**:
- Multimodal functions (Rastrigin, Rosenbrock)
- Neural architecture search
- Hyperparameter tuning

## Deterministic Annealing for EM

### From Stochastic to Deterministic

While simulated annealing uses stochastic sampling, **deterministic annealing** works with expected values directly—making it suitable for EM-type algorithms where we can compute expectations analytically.

### The Annealed Free Energy

Introduce inverse temperature $\beta = 1/T$ and define the **annealed log-likelihood**:

$$
\ell_\beta(\theta) = \frac{1}{\beta} \log \int p(\mathbf{X}, \mathbf{Z} | \theta)^\beta \, d\mathbf{Z}
$$

At $\beta = 1$, this recovers the standard log-likelihood $\ell(\theta) = \log p(\mathbf{X} | \theta)$.

### Temperature Smooths the Landscape

| $\beta$ | Interpretation | Optimization Landscape |
|---------|----------------|------------------------|
| $\beta \to 0$ | High temperature | Smooth, often convex, unique optimum |
| $\beta = 1$ | Physical temperature | Original problem, multimodal |
| $\beta \to \infty$ | Zero temperature | Concentrates on MAP/MLE |

**Why small $\beta$ smooths**: The tempered posterior over latent variables:

$$
p_\beta(\mathbf{Z} | \mathbf{X}, \theta) \propto p(\mathbf{X}, \mathbf{Z} | \theta)^\beta
$$

becomes nearly **uniform** as $\beta \to 0$. In a Gaussian mixture model:
- Each data point belongs equally to all clusters
- The discrete cluster structure washes out
- Only global statistics matter

As $\beta$ increases toward 1, cluster assignments sharpen and the multimodal structure emerges.

### The Annealed EM Algorithm

At each temperature level $\beta$, run EM on the annealed objective:

**E-step (annealed)**:
$$
q_\beta^{(n)}(\mathbf{Z}) \propto p(\mathbf{X}, \mathbf{Z} | \theta^{(n)})^\beta
$$

**M-step**:
$$
\theta^{(n+1)} = \arg\max_\theta \mathbb{E}_{q_\beta^{(n)}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]
$$

Then increase $\beta$ and continue.

```python
def annealed_em(X, K, betas=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]):
    theta = random_init()
    
    for beta in betas:
        # Run EM at current temperature until convergence
        for iteration in range(max_em_iterations):
            responsibilities = e_step_annealed(X, theta, beta)
            theta = m_step(X, responsibilities)
            if converged(theta):
                break
    
    return theta
```

### Example: Gaussian Mixture Model

For GMM with temperature parameter $\beta$:

**Annealed responsibilities**:
$$
r_{nk}^\beta = \frac{\pi_k^\beta \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)^\beta}{\sum_j \pi_j^\beta \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)^\beta}
$$

At $\beta \to 0$: $r_{nk}^\beta \to 1/K$ (uniform assignment)

At $\beta = 1$: Standard EM responsibilities

### Homotopy Continuation Perspective

Deterministic annealing is a form of **homotopy continuation** from numerical analysis:

1. Define a family of problems $P_\beta$ parameterized by $\beta \in [0, 1]$
2. $P_0$ is easy (convex, unique solution)
3. $P_1$ is the target (hard, multimodal)
4. Trace the solution path from $P_0$ to $P_1$

The hope: the solution varies **continuously** in $\beta$, allowing us to follow a path from the easy solution to the hard one.

### Failure Modes

**Phase transitions**: At critical $\beta_c$, the optimal solution may jump discontinuously.

**Bifurcations**: Multiple solution branches emerge at critical temperatures.

**Too-fast annealing**: Coarse temperature grid misses the true solution path.

## Temperature as a Unifying Concept

### The Same Mathematics, Different Contexts

Temperature appears throughout machine learning under different names:

| Domain | Name | Mathematical Form |
|--------|------|-------------------|
| Statistical mechanics | Temperature $T$ | $p \propto \exp(-E/T)$ |
| MCMC | Inverse temperature $\beta$ | $\pi_\beta \propto \pi^\beta$ |
| Simulated annealing | Cooling schedule $T(t)$ | $\alpha \propto \exp(-\Delta E/T)$ |
| Langevin dynamics | Noise level | $dx = s \, dt + \sqrt{2T} \, dW$ |
| Diffusion models | Noise variance $\sigma^2(t)$ | Forward/reverse SDE |
| Neural networks | Softmax temperature | $\text{softmax}(z/T)$ |

All are manifestations of the **same fundamental concept**.

### Temperature in MCMC at Fixed $T$

Standard MCMC samples from $\pi(x) \propto \exp(-E(x))$, equivalent to the Boltzmann distribution at $T = 1$.

We can sample at any fixed temperature:
$$
\pi_T(x) \propto \exp(-E(x)/T) = \pi(x)^{1/T}
$$

**Hot chains** ($T > 1$): Flattened distribution, easy exploration, poor approximation to target.

**Cold chains** ($T < 1$): Concentrated distribution, hard to sample, good local accuracy.

**Parallel tempering** exploits this by running chains at multiple fixed temperatures with occasional swaps.

### Temperature in Langevin Dynamics

The temperature-scaled Langevin SDE:

$$
dx_t = -\nabla E(x_t) \, dt + \sqrt{2T} \, dW_t
$$

has stationary distribution $\pi_T(x) \propto \exp(-E(x)/T)$.

**Note**: Temperature scales the **noise**, not the drift!

| Temperature | Dynamics |
|-------------|----------|
| $T = 1$ | Standard Langevin (samples from $\pi$) |
| $T \to 0$ | $dx = -\nabla E \, dt$ (gradient descent) |
| $T \to \infty$ | $dx = \sqrt{2T} \, dW$ (pure Brownian motion) |

### Temperature in Diffusion Models

Diffusion models use a noise schedule $\beta(t)$ that plays the role of inverse temperature:

**Forward process** (add noise):
$$
dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t
$$

**Reverse process** (denoise):
$$
dx_t = \left[-\frac{1}{2}\beta(t) x_t - \beta(t) \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t
$$

The analogy is precise:

| Simulated Annealing | Diffusion Models |
|---------------------|------------------|
| Start: high $T$ (uniform) | Start: high $\sigma$ (Gaussian noise) |
| End: low $T$ (concentrated) | End: low $\sigma$ (data distribution) |
| Decrease $T$ over time | Decrease $\sigma$ over time |
| Goal: find optimum | Goal: generate sample |

Both use temperature schedules to interpolate between easy and hard problems!

### Softmax Temperature

In neural networks, softmax temperature controls output sharpness:

$$
\text{softmax}_T(z_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

**Low $T$**: Sharp distribution approaching argmax.

**High $T$**: Soft distribution approaching uniform.

Applications include knowledge distillation (soft teacher outputs), reinforcement learning (entropy-regularized policies), and calibration (temperature scaling).

The $1/\sqrt{d_k}$ factor in attention is also a temperature preventing saturation:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

## Advanced Temperature Methods

### Parallel Tempering (Replica Exchange)

Run multiple chains at different fixed temperatures $T_1 < T_2 < \cdots < T_K$:

- Hot chains ($T_k$ large): Explore globally, escape local minima
- Cold chains ($T_k$ small): Sample accurately from target
- Periodically attempt to **swap** configurations between adjacent chains

**Swap acceptance probability** (maintains detailed balance):
$$
\alpha_{\text{swap}} = \min\left(1, \frac{\pi_{T_i}(x_j) \pi_{T_j}(x_i)}{\pi_{T_i}(x_i) \pi_{T_j}(x_j)}\right)
$$

Unlike simulated annealing, parallel tempering is **proper MCMC** with stationary targets.

### Annealed Importance Sampling

Use annealing to improve importance sampling:

1. Start from easy distribution $\pi_0$ (high temperature)
2. Define intermediate distributions $\pi_0 \to \pi_1 \to \cdots \to \pi_T = \pi$
3. Move samples through the sequence, tracking importance weights
4. Final weights correct for the path taken

This reduces variance compared to direct importance sampling from $\pi$.

### Tempered Langevin Dynamics

Combine Langevin dynamics with temperature exchange:

1. Run Langevin step at current temperature
2. Propose temperature swap with neighboring chain
3. Accept/reject swap to maintain detailed balance
4. Repeat

This inherits benefits of both continuous dynamics and temperature-based exploration.

## Practical Guidelines

### Choosing Initial Temperature

**For simulated annealing**:
- Want ~80% acceptance rate initially
- Set $T_0$ such that typical uphill moves are accepted
- Rule of thumb: $T_0 \approx \text{std}(\Delta E)$ over random moves

**For diffusion models**:
- Want to fully destroy data signal
- Set $\sigma_{\max}$ such that SNR $\approx 0$
- Rule of thumb: $\sigma_{\max} \approx \text{std}(\text{data})$

### Choosing Schedule Length

**Too fast**: Miss global optimum (SA), poor sample quality (diffusion), trapped in local modes.

**Too slow**: Computational waste, diminishing returns.

**Typical ranges**:
- Simulated annealing: 1,000–10,000 iterations
- Diffusion models: 100–1,000 steps

### Adaptive Temperature Control

Monitor acceptance rate and adjust:

```python
if acceptance_rate > 0.8:
    T *= 0.95  # Cool faster (too hot)
elif acceptance_rate < 0.2:
    T *= 1.05  # Cool slower (too cold)
```

### Common Pitfalls

**Confusing temperature and learning rate**: Temperature controls noise/exploration; learning rate controls step size. Both affect exploration but are distinct.

**Wrong scaling**: Temperature must scale consistently. The correct Langevin SDE at temperature $T$ is:
$$
dx = -\nabla E(x) \, dt + \sqrt{2T} \, dW
$$
**not** $dx = T \nabla E(x) \, dt + \sqrt{2} \, dW$.

**Using constant temperature**: For optimization, need annealing ($T \to 0$). For sampling, need fixed $T = 1$. Using constant high $T$ gives exploration but wrong distribution; constant low $T$ gives poor exploration.

## Summary: The Unifying Principle

Temperature is a **fundamental control parameter** across probabilistic machine learning:

| Method | Temperature Role | Goal |
|--------|------------------|------|
| MCMC (fixed $T$) | Controls exploration vs. accuracy | Sample from $\pi$ |
| Simulated annealing | Decreasing $T(t) \to 0$ | Find global optimum |
| Deterministic annealing | Increasing $\beta \to 1$ | Escape local optima in EM |
| Parallel tempering | Multiple fixed temperatures | Better mixing via exchange |
| Diffusion models | Noise schedule $\sigma(t)$ | Generate samples |
| Softmax | Output sharpness | Various (distillation, RL) |

**The unifying principle**:
- **High temperature**: Smooth landscape, wide exploration, easy sampling
- **Low temperature**: Sharp landscape, focused exploitation, concentrated distribution
- **Annealing**: Path from easy (high $T$) to hard (low $T$)

These seemingly disparate techniques—MCMC, simulated annealing, diffusion models—are all exploiting the same mathematical structure: the temperature parameter that controls the trade-off between energy and entropy.

Understanding this unification provides insight into why these methods work, how to design new algorithms, and how techniques transfer across domains. The temperature parameter is one of the most powerful tools in probabilistic machine learning.
