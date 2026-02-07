# Temperature Schedules

The cooling schedule is the heart of simulated annealing—it determines the trade-off between exploration and exploitation, and ultimately whether the algorithm finds the global optimum. This section covers the theory and practice of temperature schedule design.

---

## The Role of the Cooling Schedule

### What the Schedule Controls

The temperature schedule $T(t)$ determines:

1. **Acceptance probability**: $\alpha = \min(1, e^{-\Delta E / T(t)})$
2. **Exploration vs. exploitation**: High $T$ explores; low $T$ exploits
3. **Convergence rate**: How quickly we approach the optimum
4. **Solution quality**: Whether we find global or local optimum

### The Fundamental Trade-off

| Cool Too Fast | Cool Too Slow |
|---------------|---------------|
| Get trapped in local minima | Waste computation |
| Miss global basin | Diminishing returns |
| Poor solution quality | Same quality, more time |
| Like greedy optimization | Like random search |

The optimal schedule balances these extremes.

---

## Theoretical Schedules

### Logarithmic Cooling

The **only** schedule with guaranteed convergence to the global optimum:

$$
T(t) = \frac{c}{\log(t + t_0)}
$$

where $c \geq d^*$ (the critical depth) and $t_0 \geq 2$.

**Properties**:

- Converges to global optimum with probability 1
- Impractically slow: reaching $T = \epsilon$ requires $t \approx e^{c/\epsilon}$
- Theoretical benchmark, rarely used in practice

**Derivation intuition**: The chain needs time $\tau(T) \sim e^{d^*/T}$ to cross energy barriers of height $d^*$. For the sum $\sum_t e^{-d^*/T(t)}$ to diverge (ensuring all barriers are eventually crossed), we need $T(t) \sim c/\log(t)$.

### Inverse Linear Cooling

A faster theoretical schedule with weaker guarantees:

$$
T(t) = \frac{T_0}{1 + \alpha t}
$$

**Properties**:

- Faster than logarithmic
- Convergence to global optimum not guaranteed
- Provides polynomial-time approximation guarantees for some problems

### Inverse Logarithmic-Linear

A hybrid schedule:

$$
T(t) = \frac{T_0}{\log(1 + \alpha t)}
$$

Interpolates between inverse linear (early) and logarithmic (late) behavior.

---

## Practical Schedules

### Exponential (Geometric) Cooling

The most widely used practical schedule:

$$
T(t) = T_0 \cdot \alpha^t, \quad \alpha \in (0, 1)
$$

or equivalently $T_{k+1} = \alpha \cdot T_k$.

**Typical values**: $\alpha \in [0.85, 0.99]$

**Properties**:

- Simple to implement
- Constant ratio between successive temperatures
- Reaches low temperatures quickly
- No convergence guarantee, but often effective

**Time to reach temperature $T_f$**:

$$
t_f = \frac{\log(T_f / T_0)}{\log(\alpha)}
$$

### Linear Cooling

$$
T(t) = T_0 - \beta t = T_0 \left(1 - \frac{t}{t_{\max}}\right)
$$

where $\beta = T_0 / t_{\max}$.

**Properties**: Reaches $T = 0$ at $t = t_{\max}$, uniform rate of temperature decrease. May cool too slowly at high $T$ and too fast at low $T$—simple but often suboptimal.

### Quadratic Cooling

$$
T(t) = T_0 \left(1 - \frac{t}{t_{\max}}\right)^2
$$

Slower initial cooling, faster final cooling. Spends more time at intermediate temperatures and can help with phase transitions.

### Inverse Cooling

$$
T(t) = \frac{T_0}{1 + \beta t}
$$

Hyperbolic decay that never reaches zero (approaches asymptotically). Slower than exponential at low temperatures.

---

## Comparison of Schedules

### Decay Profiles

For $T_0 = 100$, $t_{\max} = 1000$:

| Schedule | $T(100)$ | $T(500)$ | $T(900)$ |
|----------|----------|----------|----------|
| Logarithmic ($c = 100$) | 21.7 | 16.1 | 14.7 |
| Exponential ($\alpha = 0.995$) | 60.6 | 8.2 | 1.1 |
| Linear | 90.0 | 50.0 | 10.0 |
| Quadratic | 81.0 | 25.0 | 1.0 |

### Choosing a Schedule

| Problem Characteristic | Recommended Schedule |
|----------------------|---------------------|
| Many shallow local minima | Exponential (faster) |
| Few deep local minima | Slower exponential or logarithmic |
| Unknown landscape | Adaptive schedule |
| Limited time budget | Aggressive exponential |
| High-quality solution needed | Slow schedule + restarts |

---

## Adaptive Schedules

### Acceptance Rate-Based Adaptation

Adjust temperature to maintain a target acceptance rate:

```python
def adaptive_cooling(E, x, T, target_accept=0.44, window=100):
    accepts = []
    
    for t in range(max_iter):
        x_new = propose(x)
        delta_E = E(x_new) - E(x)
        
        if np.random.rand() < min(1, np.exp(-delta_E / T)):
            x = x_new
            accepts.append(1)
        else:
            accepts.append(0)
        
        # Adapt temperature
        if len(accepts) >= window:
            recent_rate = np.mean(accepts[-window:])
            if recent_rate > target_accept + 0.05:
                T *= 0.95  # Cool faster
            elif recent_rate < target_accept - 0.05:
                T *= 1.02  # Warm up slightly
    
    return x
```

**Target acceptance rates**: Early (exploration) 80-90%, middle 40-60%, late (exploitation) 10-30%.

### Lam-Delosme Schedule

Adapts cooling rate based on observed acceptance:

$$
T_{k+1} = T_k \cdot \exp\left(-\frac{\lambda T_k}{\sigma_E}\right)
$$

where $\sigma_E$ is the standard deviation of energy changes and $\lambda$ is a control parameter. The intuition is to cool faster when energy changes are small (smooth landscape) and cool slower when energy changes are large (rough landscape).

### Huang-Romeo-Sangiovanni Schedule

Based on entropy estimation:

$$
T_{k+1} = T_k \cdot \exp\left(-\frac{T_k}{\sigma_E + \epsilon}\right)
$$

Automatically adjusts based on the local energy landscape.

---

## Multi-Stage Schedules

### Piecewise Schedules

Different cooling rates for different phases:

```python
def piecewise_schedule(t, T0=100, t_max=1000):
    # Fast initial cooling
    if t < 0.2 * t_max:
        return T0 * (0.98 ** t)
    # Slow middle phase
    elif t < 0.8 * t_max:
        T_start = T0 * (0.98 ** (0.2 * t_max))
        return T_start * (0.995 ** (t - 0.2 * t_max))
    # Fast final cooling
    else:
        T_start = T0 * (0.98 ** (0.2 * t_max)) * (0.995 ** (0.6 * t_max))
        return T_start * (0.99 ** (t - 0.8 * t_max))
```

### Reheating (Non-Monotonic Schedules)

Occasionally increase temperature to escape local minima:

```python
def schedule_with_reheat(t, T0=100, reheat_interval=500):
    base_T = T0 * (0.995 ** t)
    # Periodic reheat
    if t % reheat_interval == 0 and t > 0:
        return base_T * 5  # Reheat by factor of 5
    return base_T
```

Reheating is particularly useful when the landscape has multiple deep basins separated by high barriers. Periodic temperature spikes allow the chain to escape basins it might otherwise remain trapped in.

---

## Implementation Considerations

### Discrete vs. Continuous Schedules

**Discrete (step-wise)**:

```python
temperatures = [100, 80, 60, 40, 20, 10, 5, 2, 1, 0.5, 0.1]
for T in temperatures:
    for _ in range(steps_per_temp):
        # Metropolis step at temperature T
```

**Continuous**:

```python
for t in range(max_iter):
    T = schedule(t)
    # Metropolis step at temperature T
```

Discrete schedules allow equilibration at each level; continuous schedules are simpler to implement.

### Vectorized Implementation

For running multiple chains in parallel:

```python
def vectorized_sa(E, x0_batch, schedule, n_steps):
    """Run SA on multiple starting points in parallel."""
    n_chains = len(x0_batch)
    x = x0_batch.copy()
    E_x = np.array([E(xi) for xi in x])
    
    for t in range(n_steps):
        T = schedule(t)
        
        # Propose for all chains
        x_prop = propose_batch(x)
        E_prop = np.array([E(xi) for xi in x_prop])
        
        # Acceptance probabilities
        delta_E = E_prop - E_x
        accept_prob = np.minimum(1, np.exp(-delta_E / T))
        accept = np.random.rand(n_chains) < accept_prob
        
        # Update
        x[accept] = x_prop[accept]
        E_x[accept] = E_prop[accept]
    
    # Return best across all chains
    best_idx = np.argmin(E_x)
    return x[best_idx], E_x[best_idx]
```

### Numerical Stability

At very low temperatures, $e^{-\Delta E / T}$ can underflow:

```python
def stable_accept_prob(delta_E, T, min_T=1e-10):
    """Numerically stable acceptance probability."""
    T = max(T, min_T)
    
    if delta_E <= 0:
        return 1.0
    
    exponent = -delta_E / T
    if exponent < -700:  # Underflow threshold
        return 0.0
    
    return np.exp(exponent)
```

---

## Diagnostics

### Tracking Temperature and Acceptance

```python
def sa_with_diagnostics(E, x0, schedule, n_steps):
    x = x0
    history = {
        'temperature': [], 'energy': [],
        'best_energy': [], 'accept_rate': []
    }
    
    E_best = E(x)
    accepts = []
    
    for t in range(n_steps):
        T = schedule(t)
        x_new = propose(x)
        delta_E = E(x_new) - E(x)
        
        accept = np.random.rand() < min(1, np.exp(-delta_E / T))
        accepts.append(accept)
        
        if accept:
            x = x_new
        
        E_x = E(x)
        E_best = min(E_best, E_x)
        
        history['temperature'].append(T)
        history['energy'].append(E_x)
        history['best_energy'].append(E_best)
        
        if len(accepts) >= 100:
            history['accept_rate'].append(np.mean(accepts[-100:]))
    
    return x, history
```

The acceptance rate over time is one of the most informative diagnostics. A well-designed schedule produces a smoothly declining acceptance rate from ~80% to ~10% over the course of the run.

---

## Summary

| Schedule | Formula | Convergence | Speed | Use Case |
|----------|---------|-------------|-------|----------|
| Logarithmic | $c/\log(t)$ | Guaranteed | Very slow | Theoretical |
| Exponential | $T_0 \alpha^t$ | No guarantee | Fast | General purpose |
| Linear | $T_0(1 - t/t_{\max})$ | No guarantee | Medium | Simple problems |
| Adaptive | Based on acceptance | No guarantee | Variable | Unknown landscapes |
| Piecewise | Phase-dependent rates | No guarantee | Medium | Better exploration |

**Practical recommendations**:

1. Start with exponential ($\alpha \approx 0.95$)
2. Set $T_0$ for ~80% initial acceptance
3. Monitor acceptance rate during run
4. Use adaptive schedules for difficult problems
5. Consider multiple restarts rather than very slow cooling
