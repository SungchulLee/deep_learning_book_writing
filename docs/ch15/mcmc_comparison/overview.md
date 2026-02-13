# MCMC Methods Comparison

All MCMC methods aim to sample from a target distribution $\pi(x)$, but they employ fundamentally different strategies with distinct trade-offs. This section provides a comprehensive comparison of the four foundational methods—Metropolis-Hastings, Gibbs sampling, Langevin dynamics, and Hamiltonian Monte Carlo—along with practical guidance for method selection.

## Overview of the Four Methods

### Metropolis-Hastings (MH)

**Proposal**: Any distribution $q(x'|x)$

**Update rule**:
1. Propose $x' \sim q(\cdot|x)$
2. Accept with probability $\alpha = \min\left(1, \frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}\right)$

**Simplest variant**: Random walk with $q(x'|x) = \mathcal{N}(x, \sigma^2 I)$

### Gibbs Sampling

**Proposal**: Full conditional distributions $\pi(x_i | x_{-i})$

**Update rule**:
```
For each coordinate i = 1, ..., d:
    Sample x_i ~ π(x_i | x_{-i})
```

**Key property**: Acceptance probability is always 1 (by construction).

### Langevin Dynamics (MALA)

**Proposal**: Gradient-guided with noise
$$
x' = x + \frac{\epsilon^2}{2}\nabla \log \pi(x) + \epsilon \eta, \quad \eta \sim \mathcal{N}(0, I)
$$

**Update rule**: Apply Metropolis-Hastings correction with this asymmetric proposal.

### Hamiltonian Monte Carlo (HMC)

**Proposal**: $L$ leapfrog steps with auxiliary momentum
```
1. Resample momentum: v ~ N(0, M)
2. Simulate Hamiltonian dynamics: (x, v) → (x', v') via L leapfrog steps
3. Negate momentum: v' → -v'
4. Accept with probability: α = min(1, exp(-ΔH))
```

## Master Comparison Table

| Aspect | MH (Random Walk) | Gibbs | MALA | HMC |
|--------|------------------|-------|------|-----|
| **Information order** | 0th (density only) | 0th (conditionals) | 1st (gradient) | 1st (gradient) |
| **Gradient required?** | No | No | Yes | Yes |
| **Auxiliary variables** | None | None | None | Momentum $v$ |
| **Typical acceptance** | 20–40% | 100% | 50–70% | 60–80% |
| **Optimal acceptance** | 23.4% | 100% | 57.4% | ~65% |
| **Tuning parameters** | Step size $\sigma$ | None | Step size $\epsilon$ | $\epsilon$, $L$, $M$ |
| **Cost per iteration** | $\mathcal{O}(1)$ | $\mathcal{O}(d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(Ld)$ |
| **Mixing time** | $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ to $\mathcal{O}(d^2)$ | $\mathcal{O}(d^{5/3})$ | $\mathcal{O}(d^{5/4})$ |
| **Step size scaling** | $\sigma \sim d^{-1/2}$ | N/A | $\epsilon \sim d^{-1/6}$ | $\epsilon \sim d^{-1/4}$ |
| **Handles discrete?** | Yes | Yes | No | No |
| **Correlation sensitivity** | High | Very high | Medium | Low |
| **Implementation complexity** | Simple | Simple | Medium | Complex |

## Information Hierarchy

The methods form a hierarchy based on how much information about the target they exploit:

**0th order** (MH, Gibbs):
- Can evaluate $\pi(x)$ at any point
- Don't know which direction increases $\pi$
- Must explore blindly or use problem structure

**1st order** (Langevin, HMC):
- Can evaluate both $\pi(x)$ and $\nabla \log \pi(x)$
- Know which direction increases $\pi$
- Can guide proposals toward high-probability regions

**Why more information helps**: Random walk MH tries all directions equally, while gradient-based methods focus on promising directions. The result is faster mixing with fewer wasted proposals.

## Detailed Pairwise Comparisons

### Random Walk MH vs. Langevin

Both make local proposals, but Langevin uses gradient information.

**Random Walk MH**:
$$
x' = x + \sigma \eta, \quad \eta \sim \mathcal{N}(0, I)
$$
- Proposal is **uninformed** (doesn't use $\nabla \log \pi$)
- Isotropic exploration in all directions
- Works for non-differentiable targets

**Langevin (MALA)**:
$$
x' = x + \frac{\epsilon^2}{2}\nabla \log \pi(x) + \epsilon \eta
$$
- Proposal is **informed** (uses gradient)
- Anisotropic exploration (follows gradient direction)
- Requires differentiable target

**Performance comparison** (steps to convergence):

| Dimension | Random Walk | Langevin | Speedup |
|-----------|-------------|----------|---------|
| $d = 10$ | 1,000 | 500 | 2× |
| $d = 100$ | 100,000 | 5,000 | 20× |
| $d = 1000$ | 10,000,000 | 100,000 | 100× |

Langevin scales as $\mathcal{O}(d^{5/3})$ vs. $\mathcal{O}(d^2)$ for random walk mixing time.

### Gibbs vs. Metropolis-Hastings

**Use Gibbs when**:
- All conditionals $\pi(x_i|x_{-i})$ are tractable (easy to sample)
- Examples: Gaussian models, conjugate priors, many hierarchical models

**Use MH when**:
- Conditionals are intractable
- Examples: Logistic regression, neural network parameters

**Hybrid approach** (Metropolis-within-Gibbs):
```python
for i in range(d):
    if conditional_tractable[i]:
        x[i] = sample_from_conditional(x, i)  # Gibbs
    else:
        x[i] = metropolis_update(x, i)  # MH
```

**Example**: Bayesian logistic regression with shrinkage prior
$$
\pi(\beta, \tau | y) \propto p(y|\beta) \cdot \mathcal{N}(\beta|0, \tau^{-1}I) \cdot \text{Gamma}(\tau|a, b)
$$

- $\tau | \beta, y$ has Gamma conditional → **Gibbs**
- $\beta | \tau, y$ is intractable → **Metropolis**

### Langevin vs. HMC

Both use gradients, but HMC adds momentum for ballistic (rather than diffusive) motion.

**Langevin**:
- First-order dynamics: $\dot{x} = \nabla \log \pi + \text{noise}$
- No momentum variable
- Diffusive motion (random walk with drift)
- Each step requires MH correction

**HMC**:
- Second-order dynamics: $\dot{x} = v$, $\dot{v} = \nabla \log \pi$
- Auxiliary momentum variable $v$
- Ballistic motion (inertia carries particle)
- Single MH correction after $L$ steps

**Trajectory comparison** (same number of gradient evaluations):

*Langevin ($L$ single-step MALA)*:
```
x₀ → x₁ → x₂ → x₃ → ... → xₗ
     MH   MH   MH        MH
```
Total distance: $\mathcal{O}(\sqrt{L}\epsilon)$ (diffusive scaling)

*HMC (one $L$-step trajectory)*:
```
x₀ → x₁ → x₂ → x₃ → ... → xₗ
                              MH
```
Total distance: $\mathcal{O}(L\epsilon)$ (ballistic scaling)

**HMC explores $\sqrt{L}$ times more efficiently per gradient evaluation!**

## Scaling with Dimension

How does the cost to obtain one effective (independent) sample scale with dimension $d$?

| Method | Mixing Time | Cost per Iteration | Cost per Effective Sample |
|--------|-------------|--------------------|-----------------------------|
| Random Walk MH | $\mathcal{O}(d^2)$ | $\mathcal{O}(1)$ | $\mathcal{O}(d^2)$ |
| Gibbs | $\mathcal{O}(d)$ to $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ | $\mathcal{O}(d^2)$ to $\mathcal{O}(d^3)$ |
| MALA | $\mathcal{O}(d^{5/3})$ | $\mathcal{O}(d)$ | $\mathcal{O}(d^{8/3})$ |
| HMC | $\mathcal{O}(d^{5/4})$ | $\mathcal{O}(Ld)$ | $\mathcal{O}(Ld^{9/4})$ |

**With optimal tuning**, HMC can achieve $\mathcal{O}(d)$ or even $\mathcal{O}(\sqrt{d})$ cost per effective sample—far better than other methods.

### Why HMC Scales Better

**Random walk**: Each dimension needs independent exploration → quadratic scaling.

**Langevin**: Gradient provides direction but motion is still diffusive.

**HMC**: Momentum enables coherent motion across many dimensions simultaneously. The particle "surfs" along level sets of the Hamiltonian.

**Example**: $d$-dimensional standard Gaussian

| Method | Mixing Time |
|--------|-------------|
| Random walk | $\sim d^2$ |
| Langevin | $\sim d^{5/3}$ |
| HMC (well-tuned) | $\sim 1$ (nearly dimension-independent!) |

## Handling Correlations

Consider a bivariate Gaussian with correlation $\rho = 0.99$:
$$
\pi(x) = \mathcal{N}\left(0, \begin{pmatrix} 1 & 0.99 \\ 0.99 & 1 \end{pmatrix}\right)
$$

**Random Walk MH**:
- Proposes in all directions equally
- Most proposals rejected (miss the narrow ridge)
- Mixing time $\sim 1/(1-\rho)^2 \approx 10{,}000$

**Gibbs Sampling**:
- Updates $x_1 | x_2$ (narrow conditional), then $x_2 | x_1$
- Zig-zags slowly along the ridge
- Mixing time $\sim 1/(1-\rho) \approx 100$

**Langevin**:
- Gradient points along the ridge
- Drifts toward mode more efficiently
- Mixing time $\sim 1/(1-\rho)^{2/3} \approx 20$

**HMC**:
- Momentum carries particle along the ridge
- With proper mass matrix $M \approx \Sigma^{-1}$: nearly independent of $\rho$
- Mixing time $\sim 1$ (well-tuned)

**Winner**: HMC with adaptive mass matrix.

## Multimodal Distributions

**Challenge**: Target has multiple well-separated modes with deep valleys between them.

**All standard methods struggle!**

| Method | Behavior |
|--------|----------|
| Random Walk / Gibbs | Trapped in one mode; exponentially unlikely to jump |
| Langevin | Gradient points to nearest mode; occasional jumps via noise |
| HMC | Energy conservation helps traverse low-probability regions; still struggles with well-separated modes |

**Solutions**:
- **Parallel tempering**: Run chains at multiple temperatures; hot chains explore, cold chains refine
- **Simulated annealing**: For optimization (finding the highest mode)
- **Variational inference**: Optimization-based alternative to sampling
- **Normalizing flows**: Learn a transformation to a simple distribution

## Non-Differentiable Targets

When $\nabla \log \pi$ doesn't exist (discrete variables, discontinuities):

**Can use**:
- Metropolis-Hastings (any proposal)
- Gibbs (if conditionals are tractable)

**Cannot use**:
- Langevin (MALA)
- HMC
- Any gradient-based method

**Hybrid strategy** for mixed discrete-continuous models:
```python
# Gibbs for discrete variables
z = sample_categorical(p(z | theta, data))

# HMC for continuous variables
theta = hmc_step(theta, z, data)
```

## Acceptance Rates and Optimal Tuning

| Method | Optimal Acceptance | Source |
|--------|-------------------|--------|
| Random Walk MH | 23.4% (high-d) | Roberts et al. (1997) |
| Gibbs | 100% | By construction |
| MALA | 57.4% | Roberts & Rosenthal (1998) |
| HMC | ~65% | Empirical |
| NUTS | Variable | Auto-tunes |

**Why 23.4% for random walk?** Balances exploration (large steps, low acceptance) vs. efficiency (small steps, high acceptance). The optimal trade-off yields this specific rate.

**Why 57.4% for MALA?** Higher than random walk because gradient-informed proposals are more likely to be accepted, so optimal step size is larger.

**Why ~65% for HMC?** With $L$ leapfrog steps, we want high acceptance for the entire trajectory. Empirically, 65% works well across problems.

## Tuning Requirements

| Method | Parameters | Difficulty |
|--------|------------|------------|
| Random Walk MH | Step size $\sigma$ | Easy (target 23% acceptance) |
| Gibbs | None | None (if conditionals are tractable) |
| MALA | Step size $\epsilon$ | Easy (target 57% acceptance) |
| HMC | $\epsilon$, $L$, $M$ | Hard (three interacting parameters) |
| NUTS | $\epsilon$, $M$ | Medium (auto-tunes $L$) |

**NUTS** (No-U-Turn Sampler) automatically determines trajectory length by detecting when the particle starts to turn back. Combined with dual averaging for $\epsilon$ and warmup adaptation for $M$, it requires minimal manual tuning.

## Memory Requirements

| Method | Memory |
|--------|--------|
| Random Walk, Gibbs, MALA | $\mathcal{O}(d)$ |
| HMC (diagonal $M$) | $\mathcal{O}(d)$ |
| HMC (full $M$) | $\mathcal{O}(d^2)$ |

For very high dimensions ($d > 10^6$), use diagonal mass matrix or switch to MALA.

## Parallelization

**Between chains**: All methods support running $K$ independent chains in parallel. Combine samples and check convergence with Gelman-Rubin $\hat{R}$.

**Within chains**:
- MH, Gibbs, MALA: Sequential by nature
- HMC: Can parallelize gradient computation if target factorizes

**Parallel tempering**: Run chains at different temperatures, occasionally swap states. Helps with multimodality.

## Computational Cost Analysis

**Per-iteration costs** (assuming gradient costs $\mathcal{O}(d)$):

| Method | Density Evals | Gradient Evals | Total |
|--------|---------------|----------------|-------|
| Random Walk MH | 2 | 0 | $\mathcal{O}(1)$ |
| Gibbs | $d$ | 0 | $\mathcal{O}(d)$ |
| MALA | 2 | 1 | $\mathcal{O}(d)$ |
| HMC | 1 | $L$ | $\mathcal{O}(Ld)$ |

**Effective cost** (per independent sample):

| Method | Iterations Needed | Total Cost |
|--------|-------------------|------------|
| Random Walk MH | $\mathcal{O}(d^2)$ | $\mathcal{O}(d^2)$ |
| Gibbs | $\mathcal{O}(d)$ | $\mathcal{O}(d^2)$ |
| MALA | $\mathcal{O}(d^{5/3})$ | $\mathcal{O}(d^{8/3})$ |
| HMC | $\mathcal{O}(d^{5/4})$ | $\mathcal{O}(Ld^{9/4})$ |

**Conclusion**: Despite higher per-iteration cost, HMC is most efficient for large $d$.

## Practical Decision Tree

```
Start
  │
  ├─ Are all conditionals tractable?
  │    ├─ Yes → Try Gibbs
  │    │         ├─ Fast convergence? → Use Gibbs ✓
  │    │         └─ Slow mixing? → Switch to HMC
  │    └─ No → Continue
  │
  ├─ Is target differentiable?
  │    ├─ No → Use Metropolis-Hastings ✓
  │    └─ Yes → Continue
  │
  ├─ What's the dimension?
  │    ├─ Low (d < 10) → MALA is fine ✓
  │    ├─ Medium (10 ≤ d ≤ 1000) → Use HMC/NUTS ✓
  │    └─ High (d > 1000) → HMC with diagonal M ✓
  │
  ├─ Strong correlations?
  │    ├─ Yes → HMC with adaptive mass matrix ✓
  │    └─ No → MALA or HMC ✓
  │
  ├─ Multimodal?
  │    ├─ Yes → Parallel tempering ✓
  │    └─ No → Standard HMC/NUTS ✓
  │
  └─ Want minimal tuning?
       ├─ Yes → Use NUTS (Stan/PyMC) ✓
       └─ No → Custom implementation ✓
```

## Practical Scenarios

### Scenario 1: Bayesian Logistic Regression

**Target**: $\pi(\beta|y,X) \propto p(y|\beta,X)p(\beta)$

**Properties**: Smooth, differentiable, unimodal, correlated parameters

**Recommendations**:
1. **HMC** (best): Fast mixing, handles correlations
2. **MALA** (good): Simpler, still uses gradients
3. **Random Walk MH** (slow): Works but inefficient
4. **Gibbs** (impossible): Conditionals intractable

### Scenario 2: Gaussian Mixture Model

**Target**: $\pi(\mu, \Sigma, \pi, z | X)$ (parameters + latent assignments)

**Properties**: Discrete latents $z$, continuous parameters, conjugacy available

**Recommendations**:
1. **Gibbs** (best): Sample $z|\text{params}$ and $\text{params}|z$ from conjugate conditionals
2. **MH** (okay): For parameters, enumerate for $z$
3. **HMC** (impossible for $z$): Can't handle discrete latents

### Scenario 3: High-Dimensional Gaussian ($d = 1000$)

**Properties**: Smooth, unimodal, known covariance structure

**Recommendations**:
1. **HMC with $M = \Sigma^{-1}$** (best): $\mathcal{O}(1)$ mixing!
2. **MALA with preconditioner** (good): Still fast
3. **Random Walk with matched proposal** (okay): Works
4. **Gibbs** (slow): Coordinate updates very slow if correlated

### Scenario 4: Multimodal Distribution

**Target**: Mixture of well-separated Gaussians

**Recommendations**:
1. **Parallel tempering** (best): Hot chains explore, cold chains refine
2. **None of basic methods work well**: All get trapped
3. **Consider alternatives**: Variational inference, normalizing flows

## Software Recommendations

**For general Bayesian inference**:
- **Stan**: Excellent NUTS implementation, automatic tuning
- **PyMC**: Python interface, NUTS + other samplers
- **NumPyro**: JAX-based, very fast HMC/NUTS

**For custom models**:
- **TensorFlow Probability**: Flexible, GPU support
- **PyTorch + custom**: Full control for research

**For simple problems**:
- **Emcee**: Affine-invariant ensemble MCMC
- **PyMC**: Simple syntax for quick prototyping

## Method Selection Summary

| Situation | Recommended Method |
|-----------|-------------------|
| Conjugate Bayesian model | Gibbs |
| Non-conjugate, low-d | MALA |
| Non-conjugate, high-d | HMC/NUTS |
| Non-differentiable target | Metropolis-Hastings |
| Highly correlated | HMC with adaptive $M$ |
| Multimodal | Parallel tempering |
| Want minimal tuning | NUTS (via Stan) |
| Need fast results | HMC on GPU |
| Mixed discrete + continuous | Gibbs (discrete) + HMC (continuous) |

## The Modern Consensus

For most Bayesian inference problems today:

**Use NUTS (via Stan or PyMC) unless you have a specific reason not to.**

NUTS provides:
- Automatic tuning of trajectory length (no U-turns)
- Adaptive step size via dual averaging
- Adaptive mass matrix during warmup
- Works well across a wide range of problems

**When to use something else**:
- Conjugate models → Gibbs (faster, simpler)
- Very high dimensions ($d > 10^6$) → MALA or diagonal HMC
- Multimodal → Add tempering
- Non-differentiable → Metropolis-Hastings
- Research / special structure → Custom implementation

## Summary: The Evolution of MCMC

The development of MCMC methods reflects increasing exploitation of problem structure:

1. **Metropolis-Hastings**: Uses only $\pi(x)$ (density values)
2. **Gibbs**: Uses $\pi(x_i|x_{-i})$ (conditional structure)
3. **Langevin**: Uses $\nabla \log \pi(x)$ (gradient information)
4. **HMC**: Uses $\nabla \log \pi(x)$ + Hamiltonian dynamics (geometric structure)

**General principle**: More structure → better performance, but stricter requirements on the target.

**The no-free-lunch reality**: No single method dominates all others. Method choice depends on problem structure, and hybrid approaches often work best in practice.

The field has largely converged on HMC/NUTS as the default workhorse for continuous Bayesian computation, with Gibbs remaining essential for conjugate subproblems and MH as the fallback for non-differentiable targets. Understanding all four methods—their mechanisms, trade-offs, and failure modes—is essential for effective probabilistic programming.
