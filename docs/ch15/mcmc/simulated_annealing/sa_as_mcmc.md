# SA as Non-Stationary MCMC

Simulated annealing can be understood as a non-stationary variant of the Metropolis-Hastings algorithm where the target distribution changes over time. This perspective clarifies the theoretical foundations, explains why convergence guarantees require slow cooling, and reveals the precise relationship to standard MCMC methods.

---

## The Non-Stationary MCMC Framework

### Standard MCMC Review

In standard MCMC, we construct a Markov chain with a **fixed** stationary distribution $\pi(x)$:

$$
\pi(x) = \int \pi(y) K(x | y) \, dy
$$

where $K(x | y)$ is the transition kernel. The chain converges to $\pi$ regardless of initialization (under mild conditions).

### Non-Stationary MCMC

In **non-stationary MCMC**, the target distribution changes with time:

$$
\pi_t(x) \propto \exp\left(-\frac{E(x)}{T(t)}\right)
$$

At each step $t$, we apply a transition kernel $K_t$ that would have $\pi_t$ as its stationary distribution—but before equilibrium is reached, we change to $\pi_{t+1}$.

**Key distinction**: The chain never equilibrates to any single distribution. Instead, it tracks a moving target.

### The Simulated Annealing Chain

Simulated annealing uses Metropolis-Hastings transitions with time-varying acceptance:

$$
\alpha_t(x' | x) = \min\left(1, \frac{\pi_t(x') q(x | x')}{\pi_t(x) q(x' | x)}\right) = \min\left(1, \exp\left(-\frac{E(x') - E(x)}{T(t)}\right) \cdot \frac{q(x | x')}{q(x' | x)}\right)
$$

For symmetric proposals $q(x' | x) = q(x | x')$:

$$
\alpha_t(x' | x) = \min\left(1, \exp\left(-\frac{\Delta E}{T(t)}\right)\right)
$$

---

## Detailed Balance at Each Step

### Instantaneous Detailed Balance

At each time $t$, the transition kernel satisfies detailed balance with respect to $\pi_t$:

$$
\pi_t(x) K_t(x' | x) = \pi_t(x') K_t(x | x')
$$

**Proof**: This follows directly from the Metropolis-Hastings construction. The acceptance probability is designed to ensure:

$$
\pi_t(x) q(x' | x) \alpha_t(x' | x) = \pi_t(x') q(x | x') \alpha_t(x | x')
$$

### Non-Equilibrium Dynamics

Although detailed balance holds instantaneously, the chain is **not in equilibrium** because:

1. The distribution of $X_t$ is not $\pi_t$ (it is the result of the previous step)
2. By the time the chain could equilibrate to $\pi_t$, the target has changed to $\pi_{t+1}$
3. The actual distribution of $X_t$ depends on the entire history of temperatures

---

## The Cooling Schedule as Time Transformation

### Continuous-Time Interpretation

Consider the cooling schedule $T(t)$ as defining a "time" transformation. The chain experiences:

- **Subjective time**: The iteration count $t$
- **Physical time**: Related to how far from equilibrium the chain is

Fast cooling compresses "physical time"—the chain does not have time to equilibrate.

### Adiabatic Limit

In the **adiabatic limit** (infinitely slow cooling), the chain remains in quasi-equilibrium:

$$
\text{Distribution of } X_t \approx \pi_{T(t)} \quad \text{for slow cooling}
$$

The system tracks the instantaneous Boltzmann distribution as temperature changes.

### Quasi-Static Process

Borrowing from thermodynamics, a **quasi-static process** is one slow enough that the system remains in equilibrium at each instant.

**Simulated annealing is quasi-static if and only if** cooling is logarithmic ($T(t) = c/\log(t)$) and the chain has time to mix before temperature changes significantly.

---

## Inhomogeneous Markov Chain Analysis

### Time-Inhomogeneous Chains

Simulated annealing defines a **time-inhomogeneous** Markov chain with transition matrices $P_1, P_2, \ldots$

The distribution after $n$ steps is:

$$
\mu_n = \mu_0 P_1 P_2 \cdots P_n
$$

This is **not** a simple power of a single matrix—the standard spectral gap analysis for homogeneous chains does not directly apply.

### Weak Ergodicity

A sequence of transition matrices is **weakly ergodic** if:

$$
\lim_{n \to \infty} \sup_{x, x'} \| P_n P_{n+1} \cdots P_m (x, \cdot) - P_n P_{n+1} \cdots P_m (x', \cdot) \|_{TV} = 0
$$

for all $m > n$.

Weak ergodicity means the chain "forgets" its initial condition, even though there is no fixed stationary distribution.

### Strong Ergodicity

**Strong ergodicity** requires convergence to a specific limit:

$$
\lim_{n \to \infty} \mu_n = \delta_{x^*}
$$

where $x^*$ is the global minimum (or uniform over global minima if multiple exist).

**Logarithmic cooling achieves strong ergodicity** to the global minimum.

---

## The Mixing Time Challenge

At temperature $T$, the Metropolis chain has a mixing time $\tau_{\text{mix}}(T)$—the time to approach $\pi_T$.

**Key observation**: $\tau_{\text{mix}}(T) \to \infty$ as $T \to 0$.

At low temperatures, acceptance probability for uphill moves approaches zero, the chain becomes nearly deterministic (greedy descent), and mixing between basins becomes exponentially slow.

### The Fundamental Trade-off

**Fast cooling**: Spend little time at each temperature, but the chain does not equilibrate → may get stuck in wrong basin.

**Slow cooling**: Chain equilibrates at each temperature, but takes very long → computationally expensive.

The logarithmic cooling theorem (see [Convergence Theory](convergence.md)) resolves this by specifying the precise rate at which this trade-off is optimally balanced.

---

## Acceptance Probability Analysis

### Energy-Dependent Acceptance

At temperature $T$, the acceptance probability for a move with energy change $\Delta E$ is:

$$
\alpha(\Delta E, T) = \min(1, e^{-\Delta E / T})
$$

**Downhill moves** ($\Delta E < 0$): Always accepted ($\alpha = 1$).

**Uphill moves** ($\Delta E > 0$):

$$
\alpha = e^{-\Delta E / T} = \begin{cases}
\approx 1 & \text{if } T \gg \Delta E \\
\approx 0 & \text{if } T \ll \Delta E
\end{cases}
$$

### Expected Acceptance Rate

If $\Delta E$ has distribution $F$ under the proposal, the expected acceptance rate is:

$$
\bar{\alpha}(T) = \int_{-\infty}^{0} dF(\Delta E) + \int_{0}^{\infty} e^{-\Delta E / T} dF(\Delta E)
$$

**High $T$**: $\bar{\alpha} \to 1$ (accept everything). **Low $T$**: $\bar{\alpha} \to P(\Delta E < 0)$ (accept only improvements).

### Monitoring Acceptance Rate

The acceptance rate provides a diagnostic for the cooling schedule:

| Acceptance Rate | Interpretation | Action |
|-----------------|----------------|--------|
| > 90% | Temperature too high | Cool faster |
| 20-50% | Good exploration-exploitation balance | Continue |
| < 10% | Temperature too low | Cool slower or stop |

---

## Comparison with Stationary MCMC

### Parallel Tempering vs. Simulated Annealing

| Aspect | Simulated Annealing | Parallel Tempering |
|--------|--------------------|--------------------|
| Target | Time-varying $\pi_{T(t)}$ | Multiple fixed $\pi_{T_k}$ |
| Chains | Single chain | Multiple parallel chains |
| Temperature | Decreasing over time | Fixed per chain, swaps |
| Goal | Optimization | Sampling |
| Stationary? | No | Yes (product distribution) |
| Output | Single optimum | Samples from $\pi$ |

### When to Use Which

**Use simulated annealing when** the goal is finding the global optimum, posterior samples are not needed, the cooling schedule budget is affordable, and the landscape has deep local minima.

**Use MCMC/parallel tempering when** the goal is posterior inference, uncertainty quantification is needed, multiple samples are required, or stationary distribution guarantees matter.

---

## SA Variants as Non-Stationary MCMC

### Adaptive Simulated Annealing

Adjust the cooling schedule based on observed acceptance:

```python
def adaptive_sa(E, x0, T0, target_accept=0.4):
    T = T0
    x = x0
    
    for t in range(max_iter):
        # Metropolis step
        x_new = propose(x)
        accept = np.random.rand() < min(1, np.exp(-(E(x_new) - E(x)) / T))
        if accept:
            x = x_new
        
        # Adapt temperature based on recent acceptance
        recent_accept_rate = compute_recent_rate()
        if recent_accept_rate > target_accept:
            T *= 0.99  # Cool faster
        else:
            T *= 1.01  # Warm slightly
    
    return x
```

This is non-stationary MCMC with **adaptive** target distribution.

### Threshold Accepting

A deterministic variant that accepts if $\Delta E < \tau(t)$ for decreasing threshold $\tau(t)$:

$$
\alpha(\Delta E, t) = \mathbf{1}[\Delta E < \tau(t)]
$$

This is the $T \to 0$ limit of SA with threshold $\tau(t) \sim T(t) \log(\text{const})$.

### Great Deluge Algorithm

Accept if the new energy is below a "water level" $W(t)$ that slowly rises:

$$
\alpha(x', t) = \mathbf{1}[E(x') < W(t)]
$$

where $W(t)$ increases from $E_{\min}$ toward some target level.

---

## Theoretical Connections

### Large Deviation Theory

The probability of being in a "wrong" basin at low temperature follows large deviation asymptotics:

$$
P(X_t \notin \mathcal{X}^*) \sim \exp\left(-\frac{\Delta F}{T(t)}\right)
$$

where $\Delta F$ is a free energy difference between the global basin and local basins.

### Metastability

At intermediate temperatures, the chain exhibits **metastability**: rapid mixing within each basin, rare transitions between basins, and effective dynamics described by basin-to-basin jumps. This separation of timescales is key to understanding SA behavior.

### Connection to Gradient Flow

As $T \to 0$, simulated annealing becomes gradient descent:

$$
X_{t+1} - X_t \approx -\eta \nabla E(X_t) + \sqrt{2T(t)} \, \xi_t
$$

The noise term $\sqrt{2T(t)} \, \xi_t$ vanishes, leaving deterministic optimization.

---

## Finite-Time Behavior

### Practical Convergence

In practice, we run for finite time $T_{\text{total}}$ and want:

$$
P(X_{T_{\text{total}}} \in \mathcal{X}^*) \geq 1 - \epsilon
$$

For logarithmic cooling, this requires $T_{\text{total}} = O(e^{d^*/\epsilon})$—often impractical.

### Multiple Restarts

A practical strategy: run multiple independent annealing chains and take the best:

```python
def multi_start_sa(E, n_restarts, schedule, max_iter):
    best_x, best_E = None, float('inf')
    
    for _ in range(n_restarts):
        x0 = random_initialization()
        x, E_x = simulated_annealing(E, x0, schedule, max_iter)
        if E_x < best_E:
            best_x, best_E = x, E_x
    
    return best_x, best_E
```

With $k$ restarts, the probability of missing the global basin decreases as $(1 - p)^k$ where $p$ is the per-run success probability.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Non-stationary MCMC** | Target distribution $\pi_t$ changes with time |
| **Instantaneous detailed balance** | Each step satisfies detailed balance for current $\pi_t$ |
| **Quasi-static limit** | Infinitely slow cooling maintains equilibrium |
| **Weak ergodicity** | Chain forgets initial condition despite changing target |
| **Strong ergodicity** | Convergence to global optimum under logarithmic cooling |
| **Metastability** | Fast intra-basin mixing, slow inter-basin transitions |

Understanding simulated annealing as non-stationary MCMC clarifies why it works (instantaneous detailed balance), why convergence is slow (mixing time grows at low $T$), why guarantees require logarithmic cooling (quasi-static process), and how it relates to standard MCMC methods.
