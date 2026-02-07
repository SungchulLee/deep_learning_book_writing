# Theoretical Comparison of MCMC Methods

This section provides a rigorous theoretical comparison of MCMC methods, covering convergence rates, spectral analysis, optimal scaling theory, and the fundamental limits of different approaches.

---

## Convergence Theory Framework

### Measuring Convergence

For a Markov chain with transition kernel $P$ and stationary distribution $\pi$, we measure convergence via:

**Total variation distance**:
$$
\|P^n(x, \cdot) - \pi\|_{TV} = \frac{1}{2} \int |P^n(x, dy) - \pi(dy)|
$$

**$\chi^2$ divergence**:
$$
\chi^2(P^n(x, \cdot) \| \pi) = \int \frac{(P^n(x, dy) - \pi(dy))^2}{\pi(dy)}
$$

**Wasserstein distance** (for continuous state spaces):
$$
W_2(\mu, \nu) = \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \int \|x - y\|^2 d\gamma(x, y) \right)^{1/2}
$$

### Mixing Time

The **mixing time** $\tau_{mix}(\epsilon)$ is the time to reach $\epsilon$-closeness to stationarity:

$$
\tau_{mix}(\epsilon) = \min\{n : \sup_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}
$$

For theoretical comparisons, we typically use $\tau_{mix} = \tau_{mix}(1/4)$.

### Spectral Gap

For reversible chains, the **spectral gap** $\gamma$ determines the convergence rate:

$$
\|P^n(x, \cdot) - \pi\|_{TV} \leq C(x) (1 - \gamma)^n
$$

The spectral gap is:

$$
\gamma = 1 - \lambda_2
$$

where $\lambda_2$ is the second-largest eigenvalue of $P$.

**Key relation**: $\tau_{mix} \asymp 1/\gamma$ (up to logarithmic factors).

---

## Spectral Analysis by Method

### Random Walk Metropolis

For a $d$-dimensional Gaussian target $\pi = \mathcal{N}(0, \Sigma)$ with proposal $q(x'|x) = \mathcal{N}(x, \sigma^2 I)$:

**Spectral gap**:
$$
\gamma_{RWM} = O\left(\frac{\sigma^2}{\text{tr}(\Sigma)}\right) = O\left(\frac{\sigma^2}{d \cdot \bar{\lambda}}\right)
$$

where $\bar{\lambda}$ is the average eigenvalue of $\Sigma$.

**Optimal scaling**: $\sigma_{opt} \sim d^{-1/2}$, giving acceptance rate $\approx 23.4\%$.

**Mixing time**: $\tau_{mix} = O(d^2)$ for isotropic targets.

### Gibbs Sampling

For a $d$-dimensional target with systematic scan Gibbs:

**Spectral gap**: Depends on correlations between variables.

For Gaussian $\pi = \mathcal{N}(0, \Sigma)$:
$$
\gamma_{Gibbs} \geq 1 - \rho_{max}^2
$$

where $\rho_{max}$ is the maximum correlation between any coordinate and the others.

**Best case** (independent coordinates): $\gamma = 1 - O(1/d)$, so $\tau_{mix} = O(d)$.

**Worst case** (highly correlated): $\gamma = O(1/d)$, so $\tau_{mix} = O(d^2)$.

### MALA (Metropolis-Adjusted Langevin)

For smooth, log-concave targets in $d$ dimensions:

**Spectral gap**:
$$
\gamma_{MALA} = O(\epsilon^2)
$$

where $\epsilon$ is the step size.

**Optimal scaling**: $\epsilon_{opt} \sim d^{-1/6}$, giving acceptance rate $\approx 57.4\%$.

**Mixing time**: $\tau_{mix} = O(d^{1/3} \cdot d^{4/3}) = O(d^{5/3})$.

### Hamiltonian Monte Carlo

For smooth, log-concave targets:

**Spectral gap**: With optimal tuning,
$$
\gamma_{HMC} = O(\epsilon^2 L^2) = O(1)
$$

when $\epsilon \sim d^{-1/4}$ and $L \sim d^{1/4}$.

**Mixing time**: $\tau_{mix} = O(d^{1/4})$ to $O(d^{1/2})$ depending on the target.

---

## Optimal Scaling Theory

### The Roberts-Rosenthal Framework

Roberts, Gelman, and Gilks (1997) established optimal scaling for random walk Metropolis:

**Theorem**: For a product target $\pi(x) = \prod_{i=1}^d f(x_i)$ with proposal $\mathcal{N}(x, \sigma^2/d \cdot I)$, as $d \to \infty$:

1. The optimal acceptance rate is $\alpha^* = 0.234$
2. The diffusion limit is $dX_t = h(\alpha)^{1/2} dW_t$
3. The optimal efficiency $h(\alpha)$ is maximized at $\alpha = 0.234$

### Optimal Acceptance Rates by Method

| Method | Optimal Acceptance | Scaling |
|--------|-------------------|---------|
| Random Walk MH | 23.4% | $\sigma \sim d^{-1/2}$ |
| MALA | 57.4% | $\epsilon \sim d^{-1/6}$ |
| HMC | ~65% | $\epsilon \sim d^{-1/4}$ |
| Gibbs | 100% (by construction) | N/A |

### The MALA Scaling Result

Roberts and Rosenthal (1998) proved:

**Theorem**: For MALA on a product target, the optimal acceptance rate is:
$$
\alpha^* = 2\Phi\left(-\frac{I^{1/2}}{2}\right) \approx 0.574
$$

where $I$ is the Fisher information and $\Phi$ is the standard normal CDF.

### HMC Scaling

For HMC, the scaling is more complex:

**Beskos et al. (2013)**: For Gaussian targets with optimal tuning:
- Step size: $\epsilon^* \sim d^{-1/4}$
- Number of steps: $L^* \sim d^{1/4}$
- Acceptance rate: ~65%
- Cost per effective sample: $O(d^{1/4} \cdot d) = O(d^{5/4})$

---

## Convergence Rate Comparison

### For Gaussian Targets

Consider $\pi = \mathcal{N}(0, I_d)$:

| Method | Mixing Time | Steps per ESS |
|--------|-------------|---------------|
| Random Walk MH | $O(d^2)$ | $O(d^2)$ |
| Gibbs | $O(d)$ | $O(d^2)$ |
| MALA | $O(d^{5/3})$ | $O(d^{5/3})$ |
| HMC | $O(d^{1/4})$ | $O(d^{5/4})$ |

### For Ill-Conditioned Targets

For $\pi = \mathcal{N}(0, \Sigma)$ with condition number $\kappa = \lambda_{max}/\lambda_{min}$:

| Method | Mixing Time |
|--------|-------------|
| Random Walk MH | $O(d^2 \kappa)$ |
| Gibbs | $O(d \kappa)$ to $O(d^2 \kappa)$ |
| MALA | $O(d^{5/3} \kappa^{1/3})$ |
| HMC | $O(d^{1/4} \kappa^{1/2})$ |

**Key insight**: HMC's dependence on condition number is weaker than other methods, making it more robust to ill-conditioning.

### For Log-Concave Targets

For strongly log-concave targets with $m I \preceq -\nabla^2 \log \pi \preceq M I$:

**Langevin convergence** (Dalalyan, 2017):
$$
W_2(\mu_n, \pi) \leq e^{-m n \epsilon} W_2(\mu_0, \pi) + O(\epsilon \sqrt{d/m})
$$

**HMC convergence** (Mangoubi & Smith, 2017):
$$
W_2(\mu_n, \pi) \leq e^{-\gamma n} W_2(\mu_0, \pi)
$$

with $\gamma = O(m/M)$ independent of dimension under certain conditions.

---

## Information-Theoretic Perspective

### Gradient Information Value

The gradient $\nabla \log \pi(x)$ provides directional information:
- Points toward higher probability regions
- Magnitude indicates "steepness" of the log-density

**Information content**: At point $x$, the gradient provides $O(d)$ bits of information (one derivative per dimension).

### Methods by Information Order

| Order | Information Used | Methods |
|-------|-----------------|---------|
| 0th | $\pi(x)$ only | RW-MH, Gibbs |
| 1st | $\pi(x)$, $\nabla \log \pi(x)$ | MALA, HMC |
| 2nd | Also $\nabla^2 \log \pi(x)$ | Riemannian HMC |

### Theoretical Limits

**Proposition**: Any MCMC method using only 0th-order information has mixing time $\Omega(d)$ for $d$-dimensional Gaussians.

**Proposition**: Gradient-based methods can achieve mixing time $O(d^{\alpha})$ with $\alpha < 1$ for well-conditioned targets.

This establishes a fundamental separation between gradient-free and gradient-based approaches.

---

## Geometric Ergodicity

### Definition

A Markov chain is **geometrically ergodic** if:
$$
\|P^n(x, \cdot) - \pi\|_{TV} \leq M(x) \rho^n
$$

for some $\rho < 1$ and function $M(x)$.

### Conditions by Method

**Random Walk MH**: Geometrically ergodic if:
- Target has exponentially decaying tails
- Proposal variance is appropriate

**MALA**: Geometrically ergodic under weaker conditions:
- Log-concavity at infinity suffices
- Can handle heavier tails than RW-MH

**HMC**: Geometrically ergodic under:
- Smooth, log-concave targets
- Appropriate step size and trajectory length

### Importance for Practice

Geometric ergodicity implies:
- Central limit theorems for MCMC averages
- Finite variance of importance sampling estimators
- Reliable confidence intervals

---

## Lower Bounds

### Fundamental Limits

**Theorem** (Complexity lower bound): For any MCMC method sampling from a $d$-dimensional Gaussian with condition number $\kappa$:
$$
\tau_{mix} = \Omega(\sqrt{\kappa})
$$

This is achieved (up to $d$-dependent factors) by HMC with optimal mass matrix.

### Query Complexity

**Theorem**: For gradient-free methods with oracle access to $\pi(x)$:
$$
\text{Queries to reach } \epsilon\text{-accuracy} = \Omega(d^2/\epsilon^2)
$$

**Theorem**: For gradient-based methods with oracle access to $\nabla \log \pi(x)$:
$$
\text{Gradient queries to reach } \epsilon\text{-accuracy} = O(d/\epsilon)
$$

for strongly log-concave targets.

---

## Asymptotic Efficiency

### Peskun Ordering

**Theorem** (Peskun, 1973): For two reversible chains $P_1, P_2$ on the same state space with the same stationary distribution:

If $P_1(x, A) \geq P_2(x, A)$ for all $x \notin A$ (i.e., $P_1$ has smaller rejection probability), then $P_1$ has smaller asymptotic variance for any function $f$.

**Implication**: Among Metropolis-Hastings variants, those with higher acceptance rates (for similar proposals) are more efficient.

### Asymptotic Variance

For estimating $\mathbb{E}_\pi[f(X)]$ from a chain $(X_1, \ldots, X_n)$:
$$
\text{Var}\left(\frac{1}{n}\sum_{i=1}^n f(X_i)\right) \approx \frac{\sigma_f^2}{n} \cdot \tau_f
$$

where $\tau_f = 1 + 2\sum_{k=1}^\infty \rho_k$ is the **integrated autocorrelation time** and $\rho_k$ is the lag-$k$ autocorrelation.

**Effective sample size**: $n_{eff} = n/\tau_f$.

### Efficiency by Method

| Method | Typical $\tau_f$ | $n_{eff}/n$ |
|--------|-----------------|-------------|
| Random Walk MH | $O(d^2)$ | $O(d^{-2})$ |
| Gibbs | $O(d)$ to $O(d^2)$ | $O(d^{-1})$ to $O(d^{-2})$ |
| MALA | $O(d^{5/3})$ | $O(d^{-5/3})$ |
| HMC | $O(d^{1/4})$ | $O(d^{-1/4})$ |

---

## Robustness Properties

### Sensitivity to Tuning

| Method | Tuning Parameters | Sensitivity |
|--------|------------------|-------------|
| Random Walk MH | $\sigma$ | Moderate |
| Gibbs | None | Low |
| MALA | $\epsilon$ | Moderate |
| HMC | $\epsilon$, $L$, $M$ | High (without NUTS) |

### Robustness to Target Properties

**Heavy tails**:
- RW-MH: Can fail (proposal doesn't match tail)
- Gibbs: Depends on conditionals
- MALA: More robust (gradient adapts)
- HMC: Can have issues (trajectory escapes)

**Multimodality**:
- All local methods struggle
- Parallel tempering helps all methods
- HMC can cross shallow barriers via momentum

**Discontinuities**:
- Gradient methods fail at discontinuities
- RW-MH and Gibbs handle discontinuities

---

## Theoretical Summary

### Convergence Rate Hierarchy

For smooth, well-conditioned targets in $d$ dimensions:

$$
\tau_{HMC} \ll \tau_{MALA} \ll \tau_{Gibbs} \lesssim \tau_{RWM}
$$

Specifically:
$$
O(d^{1/4}) \ll O(d^{5/3}) \ll O(d) \text{ to } O(d^2) \lesssim O(d^2)
$$

### Key Theoretical Insights

| Insight | Implication |
|---------|-------------|
| Gradient information reduces complexity | Use gradients when available |
| Momentum enables ballistic exploration | HMC > MALA |
| Optimal scaling depends on target | Tune acceptance rates |
| Condition number affects all methods | Precondition when possible |
| Geometric ergodicity ensures CLT | Verify for reliable inference |

### When Theory Applies

**Theory is most accurate for**:
- High-dimensional targets ($d \gg 1$)
- Log-concave or near-log-concave targets
- Well-tuned algorithms

**Theory may be misleading for**:
- Low dimensions (constants matter)
- Highly multimodal targets
- Poorly tuned implementations

---

## Exercises

1. **Spectral gap computation**. Compute the spectral gap of random walk MH on a 1D Gaussian with variance $\sigma^2$ and proposal variance $\tau^2$. How does it depend on $\tau$?

2. **Optimal scaling verification**. Implement random walk MH and verify the 23.4% optimal acceptance rate on a high-dimensional Gaussian.

3. **Mixing time comparison**. Empirically estimate mixing times for RW-MH, MALA, and HMC on a 50-dimensional Gaussian. Compare to theoretical predictions.

4. **Condition number effect**. For a 2D Gaussian with correlation $\rho$, measure how mixing time depends on $\kappa = (1+\rho)/(1-\rho)$ for each method.

5. **Peskun ordering**. Construct two MH chains differing only in proposal variance. Verify that the one with higher acceptance has lower asymptotic variance.

---

## References

1. Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). "Weak Convergence and Optimal Scaling of Random Walk Metropolis Algorithms." *Annals of Applied Probability*.
2. Roberts, G. O., & Rosenthal, J. S. (1998). "Optimal Scaling of Discrete Approximations to Langevin Diffusions." *JRSS-B*.
3. Beskos, A., Pillai, N., Roberts, G., Sanz-Serna, J. M., & Stuart, A. (2013). "Optimal Tuning of the Hybrid Monte Carlo Algorithm." *Bernoulli*.
4. Dalalyan, A. S. (2017). "Theoretical Guarantees for Approximate Sampling from Smooth and Log-Concave Densities." *JRSS-B*.
5. Peskun, P. H. (1973). "Optimum Monte-Carlo Sampling Using Markov Chains." *Biometrika*.
