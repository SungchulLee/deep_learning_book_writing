# Convergence Theory

This section develops the mathematical theory of simulated annealing convergence: when and why the algorithm finds the global optimum, the role of energy barriers, and practical implications of the theoretical results.

---

## Types of Convergence

### Convergence in Probability

Simulated annealing converges **in probability** to the global optimum if:

$$
\lim_{t \to \infty} P(X_t \in \mathcal{X}^*) = 1
$$

where $\mathcal{X}^* = \arg\min_x E(x)$ is the set of global minimizers. This is the standard notion of convergence for SA—the probability of being at the optimum approaches 1.

### Almost Sure Convergence

**Almost sure convergence** is stronger:

$$
P\left(\lim_{t \to \infty} X_t \in \mathcal{X}^*\right) = 1
$$

This means that with probability 1, the sequence eventually reaches and stays at the global optimum. For finite state spaces, convergence in probability implies almost sure convergence.

### Convergence of the Distribution

We can also ask whether the distribution of $X_t$ converges:

$$
\lim_{t \to \infty} \mu_t = \delta_{\mathcal{X}^*}
$$

where $\mu_t$ is the distribution of $X_t$ and $\delta_{\mathcal{X}^*}$ is the uniform distribution on global minima.

---

## The Main Convergence Theorem

### Statement (Geman & Geman, 1984; Hajek, 1988)

**Theorem**: Let $X_t$ be a simulated annealing Markov chain on a finite state space $\mathcal{X}$ with cooling schedule $T(t)$. Assume:

1. The proposal distribution allows transitions between any two states (irreducibility)
2. The cooling schedule satisfies $T(t) \geq \frac{d^*}{\log(t + 2)}$ for all $t$

Then:

$$
\lim_{t \to \infty} P(X_t \in \mathcal{X}^*) = 1
$$

where $d^*$ is the **critical depth** of the energy landscape.

### The Critical Depth

The critical depth $d^*$ is the key quantity characterizing the optimization landscape:

$$
d^* = \max_{x \notin \mathcal{X}^*} D(x, \mathcal{X}^*)
$$

where $D(x, A)$ is the **depth** of state $x$ relative to set $A$:

$$
D(x, A) = \min_{\gamma: x \to A} \max_{y \in \gamma} E(y) - E(x)
$$

**Interpretation**: $D(x, A)$ is the minimum "elevation gain" required to travel from $x$ to any state in $A$. The critical depth $d^*$ is then the maximum such elevation gain over all non-optimal states—the height of the highest mountain pass that must be crossed to reach the deepest valley from any other valley.

### Computing Critical Depth

For a finite state space, $d^*$ can be computed by:

1. For each local minimum $x_i \notin \mathcal{X}^*$: find all paths to any global minimum, compute the maximum elevation minus $E(x_i)$ for each path, and take the minimum over all paths to get $D(x_i, \mathcal{X}^*)$
2. Take the maximum over all local minima: this is $d^*$

**Example**: Three-well potential with $E(x_1) = 2$, $E(x_2) = 3$, $E(x^*) = 0$, and barrier heights of 5 and 4:

$$
D(x_1, x^*) = 5 - 2 = 3, \quad D(x_2, x^*) = 4 - 3 = 1, \quad d^* = \max(3, 1) = 3
$$

---

## Proof Sketch

### Key Ideas

The proof proceeds in several steps:

**Step 1: Mixing at fixed temperature.** At any fixed temperature $T > 0$, the Metropolis chain is ergodic and converges to $\pi_T(x) = e^{-E(x)/T} / Z_T$.

**Step 2: Low-temperature concentration.** As $T \to 0$:

$$
\pi_T(x) \to \begin{cases}
\frac{1}{|\mathcal{X}^*|} & x \in \mathcal{X}^* \\
0 & x \notin \mathcal{X}^*
\end{cases}
$$

**Step 3: Escape time analysis.** The expected time to escape from a local minimum $x$ to a lower-energy state scales as:

$$
\tau_{\text{escape}}(x, T) \sim e^{D(x)/T}
$$

where $D(x)$ is the minimum barrier height.

**Step 4: Sufficient cooling rate.** For the chain to escape all local minima with high probability, we need:

$$
\sum_{t=1}^{\infty} e^{-d^*/T(t)} = \infty
$$

This is satisfied if and only if $T(t) \geq d^*/\log(t)$.

### The Weak Reversibility Condition

A key technical condition is **weak reversibility**: for any two states $x, y$, there exists a sequence of states connecting them such that each transition has positive probability at any temperature. This ensures the chain can eventually reach the global minimum from any starting point.

---

## Necessary and Sufficient Conditions

### Hajek's Characterization

**Theorem (Hajek, 1988)**: For simulated annealing on a finite state space:

$$
\lim_{t \to \infty} P(X_t \in \mathcal{X}^*) = 1 \quad \Longleftrightarrow \quad \sum_{t=1}^{\infty} e^{-d^*/T(t)} = \infty
$$

This gives a complete characterization:

- Logarithmic cooling $T(t) = c/\log(t)$ with $c \geq d^*$ satisfies the condition
- Any faster cooling (e.g., exponential) violates it
- The critical depth $d^*$ is the sharp threshold

### Implications

**Logarithmic cooling is necessary**: Any schedule with $T(t) < d^*/\log(t)$ infinitely often will fail to converge with positive probability.

**The constant matters**: If $T(t) = c/\log(t)$ with $c < d^*$, convergence fails.

**Problem-dependent**: The required cooling rate depends on the specific energy landscape through $d^*$.

---

## Finite-Time Analysis

### Probability of Finding Optimum

For finite time $t$, we can bound the probability of not being at the optimum:

$$
P(X_t \notin \mathcal{X}^*) \leq C \cdot \exp\left(-\alpha \sum_{s=1}^{t} e^{-d^*/T(s)}\right)
$$

for some constants $C, \alpha > 0$ depending on the problem.

### Time to Reach Optimum

The expected hitting time to $\mathcal{X}^*$ from the worst starting point scales as $\mathbb{E}[\tau^*] \sim e^{d^*/T_{\text{final}}}$ for constant-temperature algorithms. For annealing with $T(t^*) \approx d^*/\log(t^*)$, this gives $t^* \sim e^{d^*}$—exponential in the critical depth.

### Approximation Guarantees

For faster schedules, we cannot guarantee finding the exact optimum, but we can bound the suboptimality:

**Theorem**: With exponential cooling $T(t) = T_0 \alpha^t$, after $t$ iterations:

$$
\mathbb{E}[E(X_t)] - E^* \leq O\left(\frac{d^*}{\log(1/\alpha) \cdot t}\right)
$$

The gap decreases polynomially in $t$ rather than vanishing completely.

---

## Energy Landscape Analysis

### The Transition Graph

Define a graph $G = (\mathcal{X}, \mathcal{E})$ where states $x, y$ are connected if transitions are possible. Assign edge weights:

$$
w(x \to y) = \max(0, E(y) - E(x))
$$

The critical depth is related to min-cut problems on this graph.

### Basins of Attraction

A **basin** $B(x^*)$ around local minimum $x^*$ consists of all states from which gradient descent would reach $x^*$:

$$
B(x^*) = \{x : \text{greedy descent from } x \text{ reaches } x^*\}
$$

At low temperature, the SA chain becomes trapped in basins. Escape requires crossing energy barriers.

### Funnel Structure

Many practical problems have a **funnel** structure: most of state space drains into a few deep basins, the global basin is the deepest, and barriers between basins are moderate. Funnel structures are favorable for SA because the critical depth is relatively small.

### Rugged Landscapes

**Rugged landscapes** have many local minima of similar depth, high barriers between nearby states, and large critical depth. Such landscapes are difficult for SA and may require exponentially long runs.

---

## Spectral Gap and Mixing Time

At fixed temperature $T$, the Metropolis chain has a spectral gap $\gamma(T)$ determining mixing time:

$$
\tau_{\text{mix}}(T) \sim \frac{1}{\gamma(T)}
$$

The spectral gap typically satisfies:

$$
\gamma(T) \geq C \cdot \exp\left(-\frac{\Delta E_{\max}}{T}\right)
$$

where $\Delta E_{\max}$ is the maximum energy difference and $|\mathcal{X}|$ is the state space size.

The cooling schedule must be slow enough that the chain approximately mixes at each temperature, requiring time at temperature $T$ on the order of $\tau_{\text{mix}}(T) \sim e^{\Delta/T}$. Logarithmic cooling provides this; faster cooling does not.

---

## Continuous State Spaces

### Extension of Theory

For continuous state spaces $\mathcal{X} \subseteq \mathbb{R}^d$, the theory is more complex. The challenges include infinite state space, care needed in defining barriers, and harder mixing time analysis. Under regularity conditions (smoothness, compactness), similar convergence results hold with logarithmic cooling.

### The Critical Depth in Continuous Spaces

For smooth energy functions, define:

$$
d^* = \max_{x \in \mathcal{X}} \min_{\gamma: x \to \mathcal{X}^*} \max_{y \in \gamma} E(y) - E(x)
$$

where the minimum is over continuous paths. This equals the height of the lowest mountain pass between any local minimum and the global minimum.

### Langevin Interpretation

Simulated annealing with small step sizes approximates the time-inhomogeneous Langevin diffusion:

$$
dX_t = -\nabla E(X_t) \, dt + \sqrt{2T(t)} \, dW_t
$$

Convergence results for this SDE parallel the discrete case.

---

## Practical Diagnostics

### Detecting Convergence

**Energy plateau**: If $E(X_t)$ has not improved for many iterations at low temperature, the algorithm has likely converged to some local minimum.

**Multiple runs**: If independent runs consistently find the same energy, the global optimum has likely been found.

**Acceptance rate**: At very low temperature, near-zero acceptance suggests convergence to a local minimum.

### Estimating Critical Depth

For problems where $d^*$ is unknown:

- **Upper bound**: Run SA many times, track the worst local minimum found. The barrier from this minimum to the best found gives a lower bound on $d^*$.
- **Lower bound**: If SA with a given schedule consistently finds the optimum, the schedule is sufficient, giving an upper bound on $d^*$.

### Quality Assessment

```python
def assess_convergence(E, run_sa, n_runs=20, schedule_params=None):
    """Assess SA convergence through multiple runs."""
    results = []
    
    for _ in range(n_runs):
        x0 = random_initialization()
        x_final, E_final = run_sa(E, x0, **schedule_params)
        results.append({'x': x_final, 'E': E_final})
    
    energies = [r['E'] for r in results]
    
    return {
        'best_energy': min(energies),
        'worst_energy': max(energies),
        'mean_energy': np.mean(energies),
        'std_energy': np.std(energies),
        'success_rate': sum(1 for e in energies 
                          if e == min(energies)) / n_runs
    }
```

---

## Comparison with Other Optimization Methods

### Theoretical Guarantees

| Method | Convergence | Rate | Conditions |
|--------|-------------|------|------------|
| Gradient descent | Local minimum | Geometric | Smooth, convex → global |
| Newton's method | Local minimum | Quadratic | Smooth, convex → global |
| Simulated annealing | Global minimum | Sub-linear | Logarithmic cooling |
| Genetic algorithms | Global minimum | Problem-dependent | Population diversity |
| Random search | Global minimum | $O(|\mathcal{X}|)$ | Finite state space |

### The Exploration-Exploitation Spectrum

SA's temperature parameter allows moving along the exploration-exploitation spectrum during optimization. At one extreme lies pure exploration (random search), at the other pure exploitation (gradient descent). Temperature controls the position on this spectrum, transitioning from exploration-dominant to exploitation-dominant over the course of the run.

---

## Limitations of the Theory

### The Practicality Gap

Theory says logarithmic cooling guarantees convergence. Practice shows logarithmic cooling is too slow for real problems. The resolution is to use faster schedules, accept approximate solutions, and use multiple restarts.

### Unknown Critical Depth

The critical depth $d^*$ is typically unknown and hard to estimate: it requires knowledge of the entire energy landscape, computing it exactly is as hard as solving the optimization problem, and bounds are often loose.

### Non-Markovian Improvements

The theory assumes a simple Markov chain. Practical improvements often break Markovianity through momentum methods, memory of past states, and adaptive proposals. These can improve performance but complicate analysis.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Critical depth** $d^*$ | Maximum barrier height to global optimum |
| **Logarithmic cooling** | $T(t) \geq d^*/\log(t)$ guarantees convergence |
| **Necessary and sufficient** | Hajek's condition: $\sum e^{-d^*/T(t)} = \infty$ |
| **Escape time** | $\tau \sim e^{d^*/T}$ at temperature $T$ |
| **Finite-time bounds** | Suboptimality decreases polynomially |
| **Practical gap** | Theory requires impractically slow cooling |

The convergence theory for simulated annealing provides understanding of why the algorithm works, reveals the fundamental role of energy barriers, explains why fast cooling fails, and offers guidance for schedule design even when logarithmic cooling is impractical.
