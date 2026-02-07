# Simulated Annealing Fundamentals

Simulated annealing (SA) is a **non-stationary variant of Metropolis-Hastings** designed for **optimization** rather than sampling. While standard MCMC targets a fixed distribution, SA gradually changes its target distribution to concentrate on global optima. The method draws its name and intuition from metallurgical annealing, where controlled cooling of metals produces low-energy crystalline structures.

This section develops the Boltzmann distribution foundation, the core SA algorithm, and the key distinctions from standard MCMC.

---

## The Boltzmann Distribution

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

### Temperature Limits

At **absolute zero** ($T \to 0$, $\beta \to \infty$):

- System occupies minimum energy state
- All probability mass concentrates at global minimum
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

---

## The Simulated Annealing Algorithm

### MCMC for Optimization

The key distinction between SA and standard MCMC:

| Property | Standard MCMC | Simulated Annealing |
|----------|---------------|---------------------|
| Target | Fixed $\pi(x)$ | Time-varying $\pi_{T(t)}(x)$ |
| Goal | Sample from $\pi$ | Find $\arg\max \pi$ (or $\arg\min E$) |
| Stationary? | Yes | No |
| Convergence | To distribution | To point (global optimum) |
| Output | Samples for inference | Single optimal solution |

At iteration $t$, the target distribution is:

$$
\pi_{T(t)}(x) = \frac{1}{Z_{T(t)}} \exp\left(-\frac{E(x)}{T(t)}\right)
$$

where $T(t)$ is a **cooling schedule** with $T(t) \to 0$ as $t \to \infty$.

### The Algorithm

```python
def simulated_annealing(E, x0, T_schedule, proposal, max_iter):
    """Simulated annealing for global optimization.
    
    Args:
        E: Energy function to minimize
        x0: Initial state
        T_schedule: Callable mapping iteration to temperature
        proposal: Callable generating proposal given current state
        max_iter: Maximum number of iterations
        
    Returns:
        Best state found and its energy
    """
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

The algorithm becomes **greedy descent**, accepting only improvements and converging to whichever local minimum it currently occupies.

**Annealing**: By gradually decreasing $T$:

1. **High $T$**: Explore widely, find the basin of the global minimum
2. **Moderate $T$**: Transition between nearby basins, refine location
3. **Low $T$**: Settle into the deepest accessible basin

If cooling is sufficiently slow, the algorithm ends at the **global** minimum.

### Phase Transitions and Failure Modes

**Too-fast cooling**: System gets trapped in local minimum before finding the global basin.

**Phase transitions**: At critical temperature $T_c$, the optimal basin may switch discontinuously. Cooling too quickly through $T_c$ can trap the algorithm in the wrong basin. In the traveling salesman problem, for example, small temperature changes near critical points can dramatically alter the optimal tour structure.

---

## Comparison with Gradient-Based Methods

**Gradient descent** is purely deterministic:

$$
x_{t+1} = x_t - \eta \nabla E(x_t)
$$

It is fast but **trapped** in the first local minimum encountered.

**Simulated annealing** uses controlled randomness via the exploration-exploitation trade-off controlled by temperature. Early iterations provide high randomness enabling global exploration, while late iterations provide low randomness ensuring local convergence.

For continuous domains, SA with small step sizes approximates the time-inhomogeneous Langevin diffusion:

$$
dX_t = -\nabla E(X_t) \, dt + \sqrt{2T(t)} \, dW_t
$$

As $T(t) \to 0$, this reduces to gradient descent. The noise term $\sqrt{2T(t)} \, dW_t$ provides the thermal fluctuations that enable barrier crossing.

---

## Applications

**Combinatorial optimization**:

- Traveling salesman problem (state: tour permutation, energy: total distance)
- Graph coloring (state: color assignment, energy: number of conflicts)
- Circuit design, scheduling, resource allocation

**Continuous optimization**:

- Multimodal functions (Rastrigin, Rosenbrock)
- Neural architecture search
- Hyperparameter tuning

**Quantitative finance applications**:

- Portfolio optimization with discrete constraints (integer share counts, cardinality constraints)
- Calibration of models with multiple local optima (stochastic volatility, jump-diffusion)
- Optimal execution scheduling with non-convex transaction cost structures

---

## Practical Guidelines

### Choosing Initial Temperature

Set $T_0$ so that the initial acceptance rate is approximately 80%. A useful rule of thumb:

$$
T_0 \approx \text{std}(\Delta E)
$$

computed over random proposal moves from the initial state.

### Common Pitfalls

**Confusing temperature and learning rate**: Temperature controls noise/exploration; learning rate controls step size. Both affect exploration but are distinct concepts.

**Using constant temperature**: For optimization, annealing ($T \to 0$) is essential. Constant high $T$ gives exploration but the wrong distribution; constant low $T$ gives poor exploration.

**Wrong scaling**: The correct Langevin SDE at temperature $T$ is $dx = -\nabla E(x) \, dt + \sqrt{2T} \, dW$, **not** $dx = T \nabla E(x) \, dt + \sqrt{2} \, dW$. Temperature scales the noise, not the drift.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Boltzmann distribution** | $p_T(x) \propto \exp(-E(x)/T)$, foundation of temperature methods |
| **Free energy** | $F_T = \langle E \rangle - TS$, energy-entropy trade-off |
| **Simulated annealing** | Non-stationary MH with $T(t) \to 0$ for optimization |
| **High $T$** | Exploration: accepts uphill moves, escapes local minima |
| **Low $T$** | Exploitation: greedy descent into current basin |
| **Annealing path** | Gradual cooling traces path from easy to hard problem |
