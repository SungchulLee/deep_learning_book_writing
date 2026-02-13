# The Metropolis-Hastings Algorithm

The Metropolis-Hastings (MH) algorithm is the foundational MCMC method that enables sampling from distributions known only up to a normalizing constant. This section develops the complete theory: why MH works without computing intractable integrals, how to design effective proposals, and the remarkable theoretical results on optimal tuning.

## The Fundamental Problem: Intractable Normalization

In Bayesian inference, we want to sample from the posterior:

$$
\pi(\theta | \mathbf{X}) = \frac{p(\mathbf{X} | \theta) p(\theta)}{p(\mathbf{X})}
$$

where $p(\mathbf{X} | \theta)$ is the likelihood (computable), $p(\theta)$ is the prior (computable), but

$$
p(\mathbf{X}) = \int p(\mathbf{X} | \theta) p(\theta) \, d\theta
$$

is the marginal likelihood—an integral over the entire parameter space that is almost always **intractable**.

More generally, we encounter distributions of the form:

$$
\pi(\theta) = \frac{\tilde{p}(\theta)}{Z}, \quad Z = \int \tilde{p}(\theta) \, d\theta
$$

where $\tilde{p}(\theta)$ is an **unnormalized density** we can evaluate pointwise, but $Z$ is a normalization constant we cannot compute.

The profound insight of Metropolis-Hastings is that we can sample from $\pi$ without ever computing $Z$.

## The Ratio Trick: Why We Don't Need $Z$

### The MH Acceptance Probability

The Metropolis-Hastings algorithm proposes moves according to a proposal distribution $q(\theta' | \theta)$ and accepts with probability:

$$
\alpha(\theta, \theta') = \min\left(1, \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}\right)
$$

Substituting $\pi(\theta) = \tilde{p}(\theta)/Z$:

$$
\alpha = \min\left(1, \frac{\tilde{p}(\theta')/Z \cdot q(\theta | \theta')}{\tilde{p}(\theta)/Z \cdot q(\theta' | \theta)}\right) = \min\left(1, \frac{\tilde{p}(\theta')}{\tilde{p}(\theta)} \cdot \frac{q(\theta | \theta')}{q(\theta' | \theta)}\right)
$$

**The $Z$ terms cancel!** We only need the **ratio** of unnormalized densities.

### Why Ratios Eliminate Constants

This cancellation is not accidental—it follows from a fundamental principle. For any constant $c$ and function $f$:

$$
\frac{c \cdot f(\theta')}{c \cdot f(\theta)} = \frac{f(\theta')}{f(\theta)}
$$

The normalization constant $Z = \int \tilde{p}(\theta) \, d\theta$ is precisely such a constant: it depends on the global shape of $\tilde{p}$ but not on the specific value of $\theta$. In any ratio of densities, this constant cancels.

### Concrete Example: Bayesian Posterior

For the posterior $\pi(\theta | \mathbf{X}) = p(\mathbf{X}|\theta)p(\theta)/p(\mathbf{X})$, the acceptance ratio becomes:

$$
\frac{\pi(\theta' | \mathbf{X})}{\pi(\theta | \mathbf{X})} = \frac{p(\mathbf{X}|\theta')p(\theta')/p(\mathbf{X})}{p(\mathbf{X}|\theta)p(\theta)/p(\mathbf{X})} = \frac{p(\mathbf{X}|\theta')p(\theta')}{p(\mathbf{X}|\theta)p(\theta)}
$$

The intractable marginal likelihood $p(\mathbf{X})$ vanishes. The MH acceptance probability:

$$
\alpha = \min\left(1, \frac{p(\mathbf{X}|\theta')p(\theta')}{p(\mathbf{X}|\theta)p(\theta)} \cdot \frac{q(\theta|\theta')}{q(\theta'|\theta)}\right)
$$

involves only quantities we can compute: the likelihood, prior, and proposal density.

## Detailed Balance: Why MH Converges to the Target

### The Detailed Balance Condition

For a Markov chain with transition kernel $T(\theta' | \theta)$ to have $\pi$ as its stationary distribution, it suffices to satisfy **detailed balance**:

$$
\pi(\theta) T(\theta' | \theta) = \pi(\theta') T(\theta | \theta')
$$

This says the probability flux from $\theta$ to $\theta'$ equals the flux from $\theta'$ to $\theta$—the chain is "reversible" with respect to $\pi$.

### MH Satisfies Detailed Balance

For Metropolis-Hastings, the transition kernel is:

$$
T(\theta' | \theta) = q(\theta' | \theta) \alpha(\theta, \theta')
$$

With the acceptance probability $\alpha(\theta, \theta') = \min\left(1, \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}\right)$, detailed balance holds by construction.

**Proof**: Consider the case where $\frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)} < 1$. Then:

$$
\alpha(\theta, \theta') = \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}, \quad \alpha(\theta', \theta) = 1
$$

The left side of detailed balance:

$$
\pi(\theta) q(\theta' | \theta) \cdot \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)} = \pi(\theta') q(\theta | \theta')
$$

The right side:

$$
\pi(\theta') q(\theta | \theta') \cdot 1 = \pi(\theta') q(\theta | \theta')
$$

They match. The symmetric case ($\alpha(\theta', \theta) < 1$) follows identically.

### The Key Insight

The acceptance ratio was **designed** so that:

$$
\frac{\alpha(\theta, \theta')}{\alpha(\theta', \theta)} = \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}
$$

This ratio automatically adjusts transition probabilities to satisfy detailed balance. Crucially, only **ratios** of $\pi$ appear, so constants like $Z$ cancel.

## Why Ratios Are Fundamental

### Invariance Properties

Using ratios rather than absolute values provides essential invariances:

**Dimensionless**: The ratio $\pi(\theta')/\pi(\theta)$ is dimensionless—units cancel.

**Reparametrization invariant**: Under a change of variables $\phi = g(\theta)$, the Jacobians in the density transformation cancel in the ratio.

**Scale invariant**: Multiplying $\tilde{p}$ by any constant doesn't change the ratio.

### What Would Go Wrong with Absolute Probabilities?

A naive algorithm using $\alpha = \min(1, \pi(\theta'))$ would fail because:

1. It's not dimensionally consistent (comparing density to probability)
2. It's not parametrization invariant
3. It doesn't satisfy detailed balance: $\pi(\theta)\min(1, \pi(\theta')) \neq \pi(\theta')\min(1, \pi(\theta))$ in general

## Examples Where $Z$ Is Intractable

### Bayesian Logistic Regression

For the model $y_i | \mathbf{x}_i, \boldsymbol{\beta} \sim \text{Bernoulli}(\sigma(\mathbf{x}_i^T \boldsymbol{\beta}))$ where $\sigma$ is the logistic function:

$$
\tilde{p}(\boldsymbol{\beta}) = \prod_{i=1}^n \sigma(\mathbf{x}_i^T \boldsymbol{\beta})^{y_i} (1 - \sigma(\mathbf{x}_i^T \boldsymbol{\beta}))^{1-y_i} \cdot p(\boldsymbol{\beta})
$$

The marginal likelihood $p(\mathcal{D}) = \int \tilde{p}(\boldsymbol{\beta}) \, d\boldsymbol{\beta}$ has no closed form, but MH doesn't need it.

### Gaussian Mixture Models

For a mixture of $K$ Gaussians with parameters $\boldsymbol{\theta} = (\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$:

$$
p(\mathbf{X} | \boldsymbol{\theta}) = \prod_{i=1}^n \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

The posterior normalization requires integrating over $K(d + d^2/2 + 1)$ parameters—intractable, but unnecessary for MH.

### Ising Model (Statistical Physics)

The Boltzmann distribution $\pi(\mathbf{s}) = Z^{-1}\exp(-\beta E(\mathbf{s}))$ has partition function:

$$
Z = \sum_{\mathbf{s} \in \{-1,+1\}^N} \exp(-\beta E(\mathbf{s}))
$$

This sum over $2^N$ configurations is exponentially large, yet the Metropolis algorithm samples from $\pi$ by using only energy **differences** $\Delta E = E(\mathbf{s}') - E(\mathbf{s})$.

---

## Proposal Design: The Exploration-Exploitation Trade-off

The proposal distribution $q(\theta'|\theta)$ is the only tunable component in MH. Its design involves a fundamental tension:

**Large steps** (exploration): Cover more ground per iteration, but often land in low-probability regions and get rejected.

**Small steps** (exploitation): Almost always accepted, but explore the space very slowly.

The optimal proposal balances these competing demands.

### Random Walk Proposals

The most common proposal is a symmetric random walk:

$$
\theta' = \theta + \epsilon \eta, \quad \eta \sim \mathcal{N}(0, I)
$$

equivalently $q(\theta'|\theta) = \mathcal{N}(\theta, \epsilon^2 I)$. The step size $\epsilon$ controls the scale.

**Small $\epsilon$**: Proposals stay close to current state, acceptance rate $\approx 95\%$+, but exploration is glacially slow.

**Large $\epsilon$**: Proposals venture far, acceptance rate $\approx 5\%$, most computation is wasted on rejections.

**Moderate $\epsilon$**: Balanced trade-off, acceptance rate $\approx 20$-$50\%$, best mixing.

### Symmetric Proposals: The Original Metropolis Algorithm

When the proposal is symmetric, $q(\theta'|\theta) = q(\theta|\theta')$, the Hastings ratio cancels:

$$
\alpha = \min\left(1, \frac{\pi(\theta')}{\pi(\theta)}\right)
$$

This is the original **Metropolis algorithm** (1953). Accept if the proposal has higher density; if lower, accept with probability equal to the density ratio.

### Independent Proposals

An independent proposal $q(\theta'|\theta) = g(\theta')$ doesn't depend on the current state:

$$
\alpha = \min\left(1, \frac{\pi(\theta')g(\theta)}{\pi(\theta)g(\theta')}\right)
$$

This works well when $g$ is a good approximation to $\pi$, but if $g$ is poor, acceptance rate collapses.

### Gradient-Informed Proposals: MALA

The Metropolis-Adjusted Langevin Algorithm uses gradient information:

$$
q(\theta'|\theta) = \mathcal{N}\left(\theta + \frac{\epsilon^2}{2}\nabla \log \pi(\theta), \epsilon^2 I\right)
$$

Proposals drift toward high-probability regions. This is asymmetric, so the full Hastings correction is needed. The benefit is much better scaling with dimension.

---

## Optimal Acceptance Rates: The Roberts-Gelman-Gilks Theory

### The Remarkable 23.4% Result

For a $d$-dimensional Gaussian target with random walk proposal, Roberts, Gelman, and Gilks (1997) proved:

**Optimal acceptance rate**: $\alpha_{\text{opt}} \approx 0.234$ (23.4%)

**Optimal step size**: $\sigma_{\text{opt}} \approx 2.38/\sqrt{d}$

This means we should accept only about 1 in 4 proposals!

### Why Not 50%?

The intuition that "50% acceptance is optimal" (half accepted, half rejected) is wrong. The key is that exploration speed depends on both acceptance rate **and** step size:

$$
\text{Speed} \propto \sigma \cdot \alpha(\sigma)
$$

For Gaussian targets, the acceptance rate decays exponentially with step size in high dimensions:

$$
\alpha(\sigma) \approx 2\Phi\left(-\frac{c\sigma\sqrt{d}}{2}\right)
$$

where $\Phi$ is the standard normal CDF. The product $\sigma \cdot \alpha(\sigma)$ is maximized at an intermediate value yielding 23.4% acceptance.

### The Derivation (Sketch)

In the limit $d \to \infty$ with appropriately scaled proposals, the MH chain converges to a diffusion process. The efficiency of this diffusion (measured by its speed constant) is:

$$
h(\ell) = 2\ell^2 \Phi(-\ell\sqrt{I}/2)
$$

where $\ell$ is the scaled step size and $I$ is Fisher information. Maximizing $h(\ell)$ gives:

$$
\ell^* \approx 2.38, \quad \alpha^* = 2\Phi(-\ell^*\sqrt{I}/2) \approx 0.234
$$

### Robustness Beyond Gaussians

This result extends to targets beyond Gaussians. For "smooth" targets with approximately independent components, the 23.4% rule holds in high dimensions. "Smooth" means continuous, differentiable, with reasonably Gaussian-like tails.

### MALA: 57.4% Optimal Acceptance

For the Metropolis-Adjusted Langevin Algorithm:

**Optimal acceptance rate**: $\alpha_{\text{opt}} \approx 0.574$ (57.4%)

**Optimal step size**: $\epsilon_{\text{opt}} \sim d^{-1/6}$

The higher optimal acceptance reflects that gradient-informed proposals are more efficient—more proposals go in the right direction, so we can afford to accept more.

### HMC: 60-80% Typical

Hamiltonian Monte Carlo has different characteristics because it uses long trajectories before the MH step. Typical acceptance rates are 60-80%, but the algorithm is less sensitive to precise tuning because energy conservation keeps acceptance high over a wide range.

## Dimension Scaling: The Curse and Its Amelioration

### The Random Walk Bottleneck

For random walk MH in dimension $d$:

**Optimal step size**: $\sigma_{\text{opt}} = 2.38/\sqrt{d}$

As dimension increases, we must take ever smaller steps:

| Dimension | Optimal $\sigma$ |
|-----------|------------------|
| 1 | 2.38 |
| 100 | 0.238 |
| 10,000 | 0.0238 |

**Mixing time scales as** $\mathcal{O}(d^2)$ for random walk MH—quadratic in dimension.

### Why Gradient Methods Win

Different algorithms have different scaling:

| Method | Mixing Time | Step Size Scaling |
|--------|-------------|-------------------|
| Random Walk MH | $\mathcal{O}(d^2)$ | $d^{-1/2}$ |
| MALA | $\mathcal{O}(d^{5/3})$ | $d^{-1/6}$ |
| HMC | $\mathcal{O}(d^{5/4})$ or better | $d^{-1/4}$ |

Gradient-based methods exploit the geometry of the target, enabling much larger effective steps. This is why HMC dominates in high-dimensional problems.

## Practical Tuning Guidelines

### Acceptance Rate Targets

| Method | Target Rate | If Too High | If Too Low |
|--------|-------------|-------------|------------|
| Random Walk MH (high-d) | 20-25% | Increase $\epsilon$ | Decrease $\epsilon$ |
| Random Walk MH (low-d) | 40-50% | Increase $\epsilon$ | Decrease $\epsilon$ |
| MALA | 55-60% | Increase $\epsilon$ | Decrease $\epsilon$ |
| HMC | 60-80% | Increase $\epsilon$ | Decrease $\epsilon$ |

### Adaptive Tuning During Warmup

Modern MCMC software adapts step sizes automatically during warmup:

```python
def adapt_step_size(sigma, acceptance_rate, target_rate=0.234):
    if acceptance_rate > target_rate:
        sigma *= 1.1  # Increase step size
    else:
        sigma *= 0.9  # Decrease step size
    return sigma
```

More sophisticated methods like **dual averaging** (used in Stan) provide robust, theoretically-grounded adaptation.

### Covariance Adaptation

When the target has different scales in different directions, an isotropic proposal is inefficient. **Adaptive Metropolis** (Haario et al., 2001) learns the proposal covariance:

1. Start with $q(\theta'|\theta) = \mathcal{N}(\theta, \sigma^2 I)$
2. After $N$ samples, estimate $\hat{\Sigma} = \text{Cov}(\text{samples})$
3. Switch to $q(\theta'|\theta) = \mathcal{N}(\theta, (2.38^2/d)\hat{\Sigma})$

This matches the proposal geometry to the target, dramatically improving efficiency.

**Important**: Adaptation must stop before final sampling to preserve the Markov property.

### Preconditioning with Known Structure

If you have prior knowledge of the target's covariance structure:

$$
q(\theta'|\theta) = \mathcal{N}(\theta, (2.38^2/d)\Sigma)
$$

where $\Sigma \approx \text{Cov}(\pi)$ or $\Sigma \approx [-\nabla^2 \log \pi(\theta_{\text{MAP}})]^{-1}$ (inverse Hessian at the mode).

## When Standard Theory Breaks Down

### Multimodality

Well-separated modes present a fundamental challenge. The chain must make rare, large jumps to switch between modes. These jumps have very low acceptance probability, while within-mode moves have normal acceptance.

**Solutions**: Parallel tempering, replica exchange, multi-modal proposals.

### Heavy Tails

Targets with heavy tails need different step sizes in the center versus the tails. A single $\sigma$ can't work everywhere.

**Solutions**: Adaptive proposals, $t$-distributed proposals, slice sampling.

### Strong Correlations

When parameters are highly correlated, the optimal step size differs dramatically across directions. Isotropic proposals waste most effort on rejected moves.

**Solutions**: Covariance adaptation, reparametrization, Gibbs sampling.

## The Ideal Proposal: Sampling from the Target

The theoretically optimal proposal is $q(\theta'|\theta) = \pi(\theta')$—propose from the target itself:

$$
\alpha = \min\left(1, \frac{\pi(\theta') \cdot \pi(\theta)}{\pi(\theta) \cdot \pi(\theta')}\right) = 1
$$

Every proposal is accepted! But of course, if we could sample from $\pi$ directly, we wouldn't need MCMC.

This explains why **Gibbs sampling** is so powerful when applicable: it proposes from exact conditionals $\pi(\theta_j | \theta_{-j})$, achieving 100% acceptance for each coordinate update.

## Effective Sample Size and Diagnostics

### Effective Sample Size (ESS)

MCMC samples are correlated. The effective sample size measures how many independent samples the chain provides:

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

where $N$ is the number of samples and $\rho_k$ is the autocorrelation at lag $k$.

**Goal**: Maximize ESS per unit computational time.

### Key Diagnostics

**Acceptance rate**: First check—should be in target range.

**Trace plots**: Visual inspection of chain behavior.

**$\hat{R}$ statistic**: Gelman-Rubin diagnostic comparing within-chain and between-chain variance. Should be $< 1.01$.

**ESS**: Should be large enough for reliable estimates (typically $> 100$ per parameter).

## Summary: The MH Algorithm in Full

The Metropolis-Hastings algorithm is elegant in its simplicity:

1. **Initialize**: Start at some $\theta^{(0)}$
2. **Propose**: Generate $\theta' \sim q(\theta' | \theta^{(t)})$
3. **Compute acceptance ratio**:
$$
r = \frac{\tilde{p}(\theta')}{\tilde{p}(\theta^{(t)})} \cdot \frac{q(\theta^{(t)} | \theta')}{q(\theta' | \theta^{(t)})}
$$
4. **Accept/reject**: With probability $\min(1, r)$, set $\theta^{(t+1)} = \theta'$; otherwise $\theta^{(t+1)} = \theta^{(t)}$
5. **Iterate**: Repeat steps 2-4

**What makes it work**:
- Only **ratios** of unnormalized densities are needed—$Z$ cancels
- The acceptance formula is **designed** to satisfy detailed balance
- Samples converge to the target distribution (exact in the limit)

**Practical success requires**:
- Target acceptance rate $\approx 23.4\%$ for high-dimensional random walk
- Step size scaling as $\sigma \propto 1/\sqrt{d}$
- Covariance matching when target has anisotropic structure
- Gradient information (MALA, HMC) for high-dimensional problems

The genius of Metropolis-Hastings is turning an intractable problem (computing $Z$) into a tractable algorithm (accept/reject based on ratios). This single insight—**use ratios**—transformed Bayesian inference from a theoretical framework into a practical computational method.

| Quantity | Computable? | Needed for MH? |
|----------|-------------|----------------|
| $\tilde{p}(\theta)$ (unnormalized) | ✓ | ✓ |
| $Z = \int \tilde{p}(\theta) \, d\theta$ | ✗ | ✗ |
| $\pi(\theta) = \tilde{p}(\theta)/Z$ | ✗ | ✗ |
| $\tilde{p}(\theta')/\tilde{p}(\theta)$ (ratio) | ✓ | ✓ |

We only need what we **can** compute—and that's enough.
