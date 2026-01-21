# The MCMC Sampling Problem and Ergodicity

## The Fundamental Sampling Problem

### What We Want

Given a probability distribution $\pi(x)$ on $\mathbb{R}^d$, we want to:

1. **Generate samples** $x^{(1)}, x^{(2)}, \ldots, x^{(N)} \sim \pi$
2. **Estimate expectations** $\mathbb{E}_\pi[f] = \int f(x)\pi(x)\,dx$

### Why This Is Hard

**Direct sampling** requires:
- Knowing the normalization constant $Z$ in $\pi(x) = \tilde{\pi}(x)/Z$
- Being able to transform uniform random numbers to samples from $\pi$

**Problem**: For most interesting distributions, we only know $\tilde{\pi}(x)$ (unnormalized), and:

$$
Z = \int \tilde{\pi}(x)\,dx
$$

is **intractable** (high-dimensional integral).

### Examples Where Direct Sampling Fails

**Bayesian posterior**:
$$
\pi(\theta|D) = \frac{p(D|\theta)p(\theta)}{\int p(D|\theta)p(\theta)\,d\theta}
$$

The denominator (marginal likelihood) is intractable for most models.

**Boltzmann distribution** (statistical physics):
$$
\pi(x) = \frac{1}{Z}\exp(-E(x)/T)
$$

where $Z = \int \exp(-E(x)/T)\,dx$ is the partition function.

**Graphical models**:
$$
\pi(x) = \frac{1}{Z}\prod_{c \in \mathcal{C}}\phi_c(x_c)
$$

where $Z = \sum_x \prod_c \phi_c(x_c)$ requires summing over all configurations.

## The MCMC Solution

**Key insight**: We don't need to compute $Z$ to sample!

**MCMC idea**: Construct a Markov chain whose stationary distribution is $\pi$.

### Markov Chain Basics

A **Markov chain** is a sequence $X^{(0)}, X^{(1)}, X^{(2)}, \ldots$ where:

$$
X^{(t+1)} | X^{(0)}, \ldots, X^{(t)} \sim X^{(t+1)} | X^{(t)}
$$

The future depends only on the present, not the past.

**Transition kernel**: $T(x'|x) = P(X^{(t+1)} = x' | X^{(t)} = x)$

### Stationary Distribution

A distribution $\pi$ is **stationary** if:

$$
\pi(x') = \int \pi(x)T(x'|x)\,dx
$$

**Interpretation**: If $X^{(t)} \sim \pi$, then $X^{(t+1)} \sim \pi$ (the distribution doesn't change).

### The MCMC Strategy

1. **Design** a transition kernel $T$ such that $\pi$ is stationary
2. **Run** the chain for many iterations
3. **Use** samples $x^{(t)}$ for large $t$ as approximate samples from $\pi$

## Ergodicity: Why MCMC Converges

Having $\pi$ as a stationary distribution is **not enough**!

**Question**: Does the chain actually **converge** to $\pi$?

**Answer**: Yes, if the chain is **ergodic**.

### What is Ergodicity?

A Markov chain is **ergodic** if it satisfies:

1. **Irreducibility**: Can reach any state from any other state
2. **Aperiodicity**: Doesn't cycle with fixed period
3. **Positive recurrence**: Returns to any state infinitely often (in finite expected time)

### 1. Irreducibility

**Definition**: For any states $x, x'$ with $\pi(x) > 0$ and $\pi(x') > 0$, there exists $n$ such that:

$$
P(X^{(n)} \in A | X^{(0)} = x) > 0
$$

for any neighborhood $A$ containing $x'$.

**Intuitive**: The chain can eventually reach $x'$ starting from $x$.

**Why it matters**: Without irreducibility, chain can get trapped in subset of state space.

### Example: Reducible Chain

**Setup**: Target on $\mathbb{R}$ is $\pi(x) = \frac{1}{2}\mathcal{N}(-5, 1) + \frac{1}{2}\mathcal{N}(5, 1)$ (two separated modes).

**Bad proposal**: Random walk with small step size $\sigma = 0.1$.

**Result**: 
- If start near $-5$, chain never reaches $+5$
- If start near $+5$, chain never reaches $-5$
- Chain is **reducible** (practically)

**Fix**: Larger step size, or algorithm that can jump between modes.

### 2. Aperiodicity

**Definition**: $\gcd\{n : P(X^{(n)} = x | X^{(0)} = x) > 0\} = 1$

**Intuitive**: Chain doesn't return to states with fixed period.

**Example of periodic chain**:
```
States: {1, 2, 3}
Transitions: 1→2→3→1→2→3→...
```

Always returns to state 1 after exactly 3 steps (period = 3).

**Why it matters**: Periodic chains don't converge to stationary distribution (they oscillate).

**How to ensure**: Add possibility of staying in current state (e.g., rejected proposals in MH).

### 3. Positive Recurrence

**Definition**: For any state $x$, expected time to return is finite:

$$
\mathbb{E}[\tau_x | X^{(0)} = x] < \infty
$$

where $\tau_x = \min\{n \geq 1 : X^{(n)} = x\}$.

**Intuitive**: Chain visits every state infinitely often, doesn't drift to infinity.

**Why it matters**: Ensures samples from all parts of the distribution.

## The Ergodic Theorem

**Theorem**: If a Markov chain is **ergodic** and has **stationary distribution** $\pi$, then:

$$
\lim_{N \to \infty} \frac{1}{N}\sum_{t=1}^N f(X^{(t)}) = \mathbb{E}_\pi[f(X)]
$$

with probability 1, for any integrable function $f$.

**This is why MCMC works!**

### What This Means

1. **Convergence**: Regardless of initial state $X^{(0)}$, the empirical average converges to the true expectation.

2. **Law of large numbers**: Time averages equal ensemble averages.

3. **Practical**: After sufficiently many iterations, we can estimate any expectation using samples.

### The Catch: Burn-in

Convergence is **asymptotic**: $N \to \infty$.

For finite $N$:
- Early samples may be far from $\pi$ (depend on initial state)
- Need to **discard** initial samples (burn-in/warmup)
- Only use samples after chain has "forgotten" initial state

## Detailed Balance: Sufficient Condition for Stationarity

**Detailed balance** is a strong condition that **implies** stationarity.

**Definition**: $T$ satisfies detailed balance with respect to $\pi$ if:

$$
\pi(x)T(x'|x) = \pi(x')T(x|x')
$$

for all $x, x'$.

### Why Detailed Balance Implies Stationarity

Integrate both sides over $x$:

$$
\int \pi(x)T(x'|x)\,dx = \int \pi(x')T(x|x')\,dx
$$

Left side:
$$
\int \pi(x)T(x'|x)\,dx
$$

This is the probability of being at $x'$ after one step.

Right side:
$$
\pi(x')\int T(x|x')\,dx = \pi(x') \cdot 1 = \pi(x')
$$

Therefore:
$$
\int \pi(x)T(x'|x)\,dx = \pi(x')
$$

**$\pi$ is stationary!** ✓

### Intuition

Detailed balance says: **probability flow from $x$ to $x'$ equals flow from $x'$ to $x$**.

This is stronger than global balance (stationarity), which only requires net flow to be zero.

## Ensuring Ergodicity in MCMC

### Design Principles

**For irreducibility**:
- Proposal must have support everywhere target does
- Example: Gaussian proposal $q(x'|x) = \mathcal{N}(x, \sigma^2 I)$ has support on all $\mathbb{R}^d$

**For aperiodicity**:
- Allow staying in current state
- MH automatically satisfies this (rejection → stay)

**For positive recurrence**:
- Automatically satisfied for bounded state spaces
- For unbounded: need target to have suitable tails

### Common Violations

**Reducibility**:
- Discrete proposal on continuous space (e.g., $q(x') \in \{x + \Delta, x - \Delta\}$)
- Restricted support in proposal

**Periodicity**:
- Deterministic cycle of states
- Fixed-scan Gibbs on certain structures

## Mixing Time

**Definition**: Time for chain to get "close" to stationary distribution.

**Formal**: Total variation mixing time
$$
\tau_{\text{mix}}(\epsilon) = \min\{t : \sup_x \|P^t(x, \cdot) - \pi\|_{\text{TV}} \leq \epsilon\}
$$

### What Affects Mixing Time?

1. **Dimension**: Higher dimension → longer mixing
2. **Correlation**: Strong correlations → slower mixing
3. **Multimodality**: Separated modes → much longer mixing
4. **Curvature**: Flat regions vs steep → affects exploration

### Typical Scaling

**Random walk MH**: $\tau \sim d^2$ (very poor in high dimensions)

**Gradient-based**:
- Langevin: $\tau \sim d^{5/3}$
- HMC: $\tau \sim d^{5/4}$ or better

This is why HMC dominates in high dimensions!

## The Monte Carlo Error

Even with perfect convergence, Monte Carlo has **statistical error**.

### Variance of Estimator

$$
\text{Var}\left[\frac{1}{N}\sum_{t=1}^N f(X^{(t)})\right] = \frac{\sigma_f^2}{N_{\text{eff}}}
$$

where $\sigma_f^2 = \text{Var}_\pi[f]$ and $N_{\text{eff}}$ is the **effective sample size**:

$$
N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

where $\rho_k$ is the autocorrelation at lag $k$.

### Interpretation

- **High correlation**: $N_{\text{eff}} \ll N$ (many samples needed)
- **Low correlation**: $N_{\text{eff}} \approx N$ (efficient)
- **Independent**: $N_{\text{eff}} = N$ (ideal, impossible in MCMC)

## Practical Guidelines

### Starting the Chain

**Random initialization**: Often works, but may require long burn-in

**Informed initialization**: Use MAP estimate, prior mode, or previous fit

**Multiple chains**: Start from different points to check convergence

### Running the Chain

**Burn-in**: Discard initial 50% (conservative) or use diagnostics

**Thinning**: Keep every $k$-th sample (debated — usually unnecessary)

**Monitoring**: Use $\hat{R}$, ESS, trace plots

### Stopping the Chain

**Minimum**: Until $\hat{R} < 1.01$ and ESS $> 100$

**Preferred**: ESS $> 1000$ for reliable inference

**Critical**: Even more samples for tails, quantiles

## Theoretical Guarantees vs Practice

### What Theory Guarantees

- **Ergodic theorem**: Convergence as $N \to \infty$
- **Consistency**: Estimates converge to true values
- **Central limit theorem**: Normal distribution for averages (under conditions)

### What Theory Doesn't Guarantee

- **How long** to run (problem-dependent)
- **Detection** of all modes (may not visit rare modes)
- **Finite-sample** behavior (could be terrible for practical $N$)

### The Art of MCMC

Using MCMC well requires:
- Understanding the target distribution
- Choosing appropriate algorithm
- Monitoring convergence carefully
- Interpreting results with suitable skepticism

## Summary

The MCMC sampling problem:

**Goal**: Sample from $\pi(x)$ when we only know $\tilde{\pi}(x)$ (unnormalized)

**Solution**: Markov chain with stationary distribution $\pi$

**Requirements**:
1. **Ergodicity**: Irreducibility + aperiodicity + positive recurrence
2. **Stationarity**: $\pi(x') = \int \pi(x)T(x'|x)\,dx$
3. **Sufficient time**: Run long enough to forget initial state

**Guarantees** (ergodic theorem):
$$
\frac{1}{N}\sum_{t=1}^N f(X^{(t)}) \xrightarrow{N \to \infty} \mathbb{E}_\pi[f]
$$

**In practice**:
- Design transition to satisfy detailed balance
- Ensure support of proposal covers support of target
- Monitor convergence with diagnostics
- Use effective sample size to quantify efficiency

The beauty of MCMC: we can sample from arbitrarily complex distributions using only:
1. Ability to evaluate $\tilde{\pi}(x)$ (unnormalized)
2. A clever transition kernel (MH, Gibbs, HMC, etc.)
3. Patience (enough iterations)

This is why MCMC revolutionized Bayesian statistics and computational physics!
