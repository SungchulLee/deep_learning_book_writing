# Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo is one of the most powerful MCMC algorithms for sampling from continuous distributions. By borrowing ideas from classical mechanics, HMC achieves dramatically better exploration than random-walk methods, especially in high dimensions. This section develops the complete theory: the leapfrog integrator, geometric interpretations, energy conservation, and why HMC can take multiple steps while Langevin cannot.

---

## The Hamiltonian Framework

### Augmenting the State Space

HMC introduces **auxiliary momentum variables** $\mathbf{v}$ and defines the Hamiltonian:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})
$$

where:

- **Potential energy**: $U(\mathbf{x}) = -\log \tilde{p}(\mathbf{x})$ (negative log-density)
- **Kinetic energy**: $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^\top \mathbf{M}^{-1}\mathbf{v}$ (quadratic in momentum)
- **Mass matrix**: $\mathbf{M}$ is a positive definite matrix (often diagonal or the identity)

### The Joint Distribution

The joint distribution on $(\mathbf{x}, \mathbf{v})$ is:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v})) = \exp(-U(\mathbf{x})) \exp(-K(\mathbf{v}))
$$

This factorizes:

$$
\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})
$$

**Key property**: Marginalizing over $\mathbf{v}$ recovers the target $\pi(\mathbf{x})$.

### Hamilton's Equations

The dynamics are governed by Hamilton's equations:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = \nabla \log p(\mathbf{x})
$$

These equations describe a particle moving on a surface defined by the potential energy, with the gradient of the log-density acting as a force.

---

## Leapfrog Integrator (Operator Splitting)

### The Integration Problem

We need to numerically integrate Hamilton's equations. For MCMC sampling, we need an integrator that:

1. **Preserves volume** (symplectic)
2. Is **reversible** (for detailed balance)
3. Has **bounded error** (for high acceptance)
4. Is **computationally efficient**

The leapfrog integrator achieves all four.

### Operator Splitting (Lie-Trotter-Suzuki)

Define two operators:

**Operator A** (momentum update):
$$
\mathcal{A}: \quad \mathbf{v} \to \mathbf{v} + \epsilon \nabla \log p(\mathbf{x}), \quad \mathbf{x} \to \mathbf{x}
$$

**Operator B** (position update):
$$
\mathcal{B}: \quad \mathbf{x} \to \mathbf{x} + \epsilon \mathbf{M}^{-1}\mathbf{v}, \quad \mathbf{v} \to \mathbf{v}
$$

The exact solution $e^{\epsilon(\mathcal{A} + \mathcal{B})}$ cannot be computed directly because the operators don't commute: $[\mathcal{A}, \mathcal{B}] \neq 0$.

**Splitting approximations**:

- **First-order (Lie-Trotter)**: $e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\epsilon \mathcal{A}} e^{\epsilon \mathcal{B}}$ — Error: $O(\epsilon^2)$
- **Second-order (Strang)**: $e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\epsilon \mathcal{A}/2} e^{\epsilon \mathcal{B}} e^{\epsilon \mathcal{A}/2}$ — Error: $O(\epsilon^3)$

The second-order Strang splitting is the **leapfrog integrator**.

### The Leapfrog Algorithm

Starting from $(\mathbf{x}_n, \mathbf{v}_n)$:

**Step 1** — Half-step momentum update:
$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n + \frac{\epsilon}{2} \nabla \log p(\mathbf{x}_n)
$$

**Step 2** — Full-step position update:
$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1} \mathbf{v}_{n+1/2}
$$

**Step 3** — Half-step momentum update:
$$
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} + \frac{\epsilon}{2} \nabla \log p(\mathbf{x}_{n+1})
$$

The name "leapfrog" comes from the staggered time points—positions and momenta leap over each other.

### Why Second-Order Matters

The Baker-Campbell-Hausdorff formula shows that the symmetric structure cancels the $O(\epsilon^2)$ error:

$$
e^{\epsilon \mathcal{A}/2} e^{\epsilon \mathcal{B}} e^{\epsilon \mathcal{A}/2} = e^{\epsilon(\mathcal{A}+\mathcal{B}) + O(\epsilon^3)}
$$

After $L$ steps over time $T = L\epsilon$:

- **First-order**: Global error $O(\epsilon)$
- **Second-order**: Global error $O(\epsilon^2)$

### Symplectic Property

A transformation is **symplectic** if it preserves phase space volume: $|\det \mathbf{J}| = 1$.

The leapfrog is symplectic because:

- Each elementary update (momentum or position) is symplectic
- Composition of symplectic maps is symplectic

**Consequence**: No Jacobian correction needed in the MH acceptance ratio.

### Reversibility

The leapfrog is **time-reversible**: applying leapfrog, negating momentum, and applying leapfrog again returns to the starting point.

**Proof**: Starting from $(\mathbf{x}_0, \mathbf{v}_0)$, applying leapfrog gives $(\mathbf{x}_1, \mathbf{v}_1)$. Starting from $(\mathbf{x}_1, -\mathbf{v}_1)$ and applying leapfrog returns to $(\mathbf{x}_0, -\mathbf{v}_0)$.

This ensures the proposal distribution is symmetric, simplifying the MH acceptance ratio.

### Energy Error Bound

- **Local error**: $|H(\mathbf{x}_1, \mathbf{v}_1) - H(\mathbf{x}_0, \mathbf{v}_0)| = O(\epsilon^3)$
- **Global error** (after $L$ steps): $|\Delta H| = O(L\epsilon^3) = O(T\epsilon^2)$

Typical values: $\epsilon = 0.01$–$0.1$, $L = 10$–$100$, giving $\Delta H \sim 0.1$–$1.0$ and acceptance $\alpha \approx 0.65$–$0.90$.

---

## Energy Conservation: Why HMC Samples Instead of Optimizes

### The Central Mystery

HMC follows gradients like gradient ascent and explores high-probability regions. Why doesn't it just converge to the mode?

**Answer**: Energy conservation creates oscillatory trajectories that prevent convergence.

### What Energy Conservation Means

For exact (continuous-time) Hamiltonian dynamics:

$$
\frac{d}{dt}H(\mathbf{x}_t, \mathbf{v}_t) = 0
$$

**Proof**:
$$
\frac{dH}{dt} = \frac{\partial H}{\partial \mathbf{x}}\frac{d\mathbf{x}}{dt} + \frac{\partial H}{\partial \mathbf{v}}\frac{d\mathbf{v}}{dt} = \nabla U \cdot \mathbf{M}^{-1}\mathbf{v} + \mathbf{M}^{-1}\mathbf{v} \cdot (-\nabla U) = 0
$$

The cross terms cancel exactly.

### The Oscillation Mechanism

**Without energy conservation** (gradient ascent):

1. Start anywhere with $\nabla \log p \neq 0$
2. Follow gradient uphill
3. Reach mode where $\nabla \log p = 0$
4. **Stop** — no force to continue

Result: **Optimization**, not sampling.

**With energy conservation**:

Energy conservation requires $U(\mathbf{x}_t) + K(\mathbf{v}_t) = H_0 = \text{constant}$.

- **Climbing toward mode** (lower $U$): $K$ increases → particle speeds up
- **At the mode**: $U$ is minimal → $K$ is **maximal** → particle moves fastest
- **Past the mode**: $U$ increases → $K$ decreases → particle slows and turns around
- **Result**: Oscillation through the mode

The particle cannot stop at the mode because it has residual kinetic energy.

### Time Allocation and Correct Sampling

Energy conservation creates the correct time allocation:

- **High probability** (low $U$): Large momentum volume, but fast motion
- **Low probability** (high $U$): Small momentum volume, but slow motion

The balance gives: time spent at $\mathbf{x} \propto p(\mathbf{x})$.

### Comparison: Optimization vs Sampling

| System | Energy | Behavior | Outcome |
|--------|--------|----------|---------|
| Gradient ascent | $E(\mathbf{x}) = -\log p(\mathbf{x})$ | Minimize $E$ | Converge to mode |
| Hamiltonian | $H(\mathbf{x},\mathbf{v}) = U + K$ | Conserve $H$ | Oscillate, sample |

Adding momentum + conservation law transforms optimization into sampling.

---

## Why HMC Takes Multiple Steps but Langevin Cannot

### The Fundamental Difference

- **HMC**: Runs $L$ leapfrog steps **before** MH correction
- **Langevin (MALA)**: Must apply MH correction **after every step**

This is a fundamental consequence of the underlying dynamics.

### HMC's Advantage: Energy Conservation

For exact Hamiltonian dynamics, $H$ is constant along trajectories, so:

$$
\alpha = \min\left(1, \frac{p(\mathbf{x}', \mathbf{v}')}{p(\mathbf{x}, \mathbf{v})}\right) = \min(1, e^{-\Delta H}) = 1
$$

With leapfrog (approximate), energy drift is bounded: $|\Delta H| = O(L\epsilon^3)$.

For moderate $L$ and small $\epsilon$, acceptance remains high (typically 65–95%).

### Why Langevin Cannot Chain Steps

Langevin dynamics has **no conserved quantity**:

$$
d\mathbf{x}_t = \nabla \log p(\mathbf{x}_t) dt + \sqrt{2} dW_t
$$

Each step adds **independent** Gaussian noise. After $L$ steps, the proposal distribution is:

$$
q(\mathbf{x}^{(L)} | \mathbf{x}^{(0)}) = \int q(\mathbf{x}^{(1)} | \mathbf{x}^{(0)}) \cdots q(\mathbf{x}^{(L)} | \mathbf{x}^{(L-1)}) \, d\mathbf{x}^{(1)} \cdots d\mathbf{x}^{(L-1)}
$$

This **path integral** is intractable to compute, so we cannot evaluate the MH ratio after $L$ steps.

### Deterministic vs Stochastic Trajectories

**HMC**: Trajectory is deterministic given $(\mathbf{x}, \mathbf{v})$. The proposal is a delta function, and reversibility + symplecticity give:

$$
\alpha = \min(1, e^{-\Delta H})
$$

No path integral needed.

**Langevin**: Trajectory is stochastic. Computing the proposal probability requires marginalizing over all intermediate states—intractable.

### Coherent vs Diffusive Motion

**Langevin**: Independent noise at each step → cumulative noise variance grows as $O(L\epsilon)$ → proposal becomes diffuse → acceptance drops with $L$.

**HMC**: Momentum sampled once, then evolves deterministically → coherent ballistic motion → distance traveled $\propto L\epsilon$ (vs $\sqrt{L\epsilon}$ for Langevin).

| Property | Langevin | HMC |
|----------|----------|-----|
| Dynamics | First-order (dissipative) | Second-order (Hamiltonian) |
| Conserved quantity | None | Energy $H$ |
| Trajectory | Stochastic | Deterministic |
| Exploration | Diffusive ($\sqrt{L\epsilon}$) | Ballistic ($L\epsilon$) |
| Step chaining | ✗ (intractable) | ✓ (energy conservation) |

---

## Geometric Interpretations

HMC can be understood from three complementary perspectives.

### 1. Constrained Optimization View

At each step, HMC approximately solves:

$$
\max_{\Delta \mathbf{x}} \left\{ \nabla \log p(\mathbf{x})^\top \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^\top \mathbf{H}(\mathbf{x}) \Delta \mathbf{x} \right\}
$$

subject to the kinetic energy constraint:

$$
\frac{1}{2} \Delta \mathbf{x}^\top \mathbf{M}^{-1} \Delta \mathbf{x} = K
$$

**Interpretation**:

- **Objective**: Second-order Taylor approximation of log-likelihood gain
- **Constraint**: "Speed limit" determined by kinetic energy

When $\mathbf{M}^{-1} \approx -\mathbf{H}(\mathbf{x})$ (mass matrix matches curvature):

- The constraint surface aligns with local geometry
- HMC takes **Newton-like steps** automatically
- Step size adapts to both steep and flat directions

### 2. Energy Conservation View

Energy conservation creates orbital motion:

1. Particle rolls through high-probability valleys with **high speed**
2. Climbs probability hills and **slows down**
3. **Turns around** when kinetic energy depletes
4. Returns to valleys, overshoots, oscillates
5. Time spent $\propto p(\mathbf{x})$ ✓

This is like a marble rolling on a curved surface where valleys are high-probability regions.

### 3. Second-Order Approximation View

Near $\mathbf{x}$, approximate:

$$
\log p(\mathbf{x} + \Delta \mathbf{x}) \approx \log p(\mathbf{x}) + \nabla \log p(\mathbf{x})^\top \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^\top \mathbf{H}(\mathbf{x}) \Delta \mathbf{x}
$$

Hamilton's equations are the gradient flow on this quadratic surface, but with momentum that:

1. Remembers past gradient directions
2. Continues motion beyond local optima
3. Explores along curved trajectories

**Optimal mass matrix**: $\mathbf{M}^{-1} \approx -\mathbf{H}(\mathbf{x}^*) = \mathbf{I}(\mathbf{x}^*)$ (Fisher information).

---

## HMC Preserves the Target Distribution

### Three Mechanisms Working Together

1. **Momentum resampling**: Creates the joint distribution $\pi(\mathbf{x}, \mathbf{v})$
2. **Hamiltonian dynamics**: Preserves $\pi(\mathbf{x}, \mathbf{v})$ via energy conservation + volume preservation
3. **MH correction**: Corrects for discretization errors

### Why Each Step Preserves $\pi(\mathbf{x}, \mathbf{v})$

**Momentum resampling**: Given $\mathbf{x}^{(t)} \sim \pi(\mathbf{x})$, sample $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$. The pair $(\mathbf{x}^{(t)}, \mathbf{v}) \sim \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M}) = \pi(\mathbf{x}, \mathbf{v})$ ✓

**Hamiltonian dynamics** (exact): Energy conservation ($H$ constant) + volume preservation ($|\det \mathbf{J}| = 1$) imply $\pi(\mathbf{x}_t, \mathbf{v}_t) = \exp(-H(\mathbf{x}_t, \mathbf{v}_t)) = \exp(-H(\mathbf{x}_0, \mathbf{v}_0)) = \pi(\mathbf{x}_0, \mathbf{v}_0)$ ✓

**MH correction**: Symplecticity + reversibility ensure detailed balance. Acceptance:

$$
\alpha = \min(1, e^{-\Delta H})
$$

corrects for discretization error ✓

### The Complete HMC Iteration

```
Given (x⁽ᵗ⁾, v⁽ᵗ⁾):

1. Resample momentum: v ~ N(0, M)
2. Run L leapfrog steps: (x', v') = Leapfrog(x⁽ᵗ⁾, v, L, ε)
3. Negate momentum: v' ← -v'
4. Compute acceptance: α = min(1, exp(-H(x', v') + H(x⁽ᵗ⁾, v)))
5. With probability α: accept (x⁽ᵗ⁺¹⁾, v⁽ᵗ⁺¹⁾) = (x', v')
   Otherwise: reject (x⁽ᵗ⁺¹⁾, v⁽ᵗ⁺¹⁾) = (x⁽ᵗ⁾, v)
```

---

## Practical Considerations

### Hyperparameter Selection

**Step size $\epsilon$**:

- Too large: Energy error grows, low acceptance
- Too small: Slow exploration, wasted computation
- Target acceptance: 65–80%

**Number of steps $L$**:

- Too few: Random-walk behavior persists
- Too many: Wasted computation (trajectory turns back)
- Rule of thumb: $L\epsilon \approx$ trajectory length covering typical set

**Mass matrix $\mathbf{M}$**:

- Identity: Assumes isotropic distribution (rarely true)
- Diagonal: Adapts to marginal variances (practical default)
- Full covariance: Optimal but expensive ($O(d^3)$)
- Best: $\mathbf{M} \approx \text{Cov}[\mathbf{x}]$ (estimated from warmup)

### The No-U-Turn Sampler (NUTS)

NUTS (Hoffman & Gelman, 2014) automatically selects $L$:

- Simulates trajectory until it starts to "turn back"
- Uses binary tree construction for efficiency
- Eliminates the need to tune $L$
- Standard in Stan and PyMC

### When HMC Struggles

- **Multimodal distributions**: Energy barriers prevent mode hopping
- **Varying curvature**: Single $\mathbf{M}$ insufficient
- **Discrete variables**: No gradients available
- **Heavy tails**: Trajectories can escape to infinity

**Remedies**:

- Tempering/annealing for multimodality
- Riemannian HMC for varying curvature
- Discontinuous HMC for mixed discrete-continuous

---

## Comparison with Other Methods

| Method | Uses Gradient? | Uses Curvature? | Energy-Preserving? | Result |
|--------|---------------|-----------------|-------------------|--------|
| Random walk MH | No | No | No | Slow exploration |
| Gradient descent | Yes | No | No | Optimization |
| Newton's method | Yes | Yes | No | Fast optimization |
| Langevin (MALA) | Yes | No | No (dissipative) | Sampling (diffusive) |
| HMC | Yes | Via $\mathbf{M}$ | Yes | Efficient sampling |

HMC uniquely combines gradient information with energy-preserving dynamics to achieve efficient sampling in high dimensions.

---

## Summary

| Component | Role | Key Property |
|-----------|------|--------------|
| Momentum $\mathbf{v}$ | Auxiliary variable | Creates joint $\pi(\mathbf{x}, \mathbf{v})$ |
| Hamiltonian $H$ | Total energy | $H = U + K$ |
| Leapfrog | Numerical integrator | Symplectic + reversible |
| Energy conservation | Prevents optimization | Creates oscillations |
| MH correction | Fixes discretization | Ensures exact sampling |

**Why HMC works**:

1. **Gradient information** guides exploration toward high-probability regions
2. **Energy conservation** prevents convergence to mode, enables oscillation
3. **Coherent momentum** enables ballistic (not diffusive) exploration
4. **Symplectic integration** preserves phase space volume
5. **MH correction** ensures exact sampling despite discretization

HMC sits at the intersection of statistical mechanics, differential geometry, and optimization—a beautiful synthesis that makes it the gold standard for sampling from continuous distributions in high dimensions.
