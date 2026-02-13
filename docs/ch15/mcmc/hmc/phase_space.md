# Phase Space

Phase space is the mathematical arena where Hamiltonian Monte Carlo operates. This section provides a detailed treatment of phase space geometry, the augmented state space for sampling, and the geometric structures that enable efficient exploration.

---

## Definition and Motivation

### Why Augment the State Space?

To sample from a target distribution $\pi(\mathbf{x})$, HMC introduces auxiliary **momentum variables** $\mathbf{v}$ and works in the extended space $(\mathbf{x}, \mathbf{v})$. This augmentation might seem to complicate the problem—we now have $2d$ variables instead of $d$—but it provides crucial benefits:

1. **Deterministic dynamics**: Given $(\mathbf{x}, \mathbf{v})$, the trajectory is fully determined
2. **Gradient utilization**: The score $\nabla \log \pi(\mathbf{x})$ drives coherent motion
3. **Energy conservation**: Proposals stay on approximately constant-energy surfaces
4. **Efficient exploration**: Ballistic motion covers distance $\propto L$ vs diffusive $\propto \sqrt{L}$

### The Phase Space

**Definition**: For a system with $d$-dimensional position space, the **phase space** is the $2d$-dimensional manifold:

$$
\Gamma = \{(\mathbf{x}, \mathbf{v}) : \mathbf{x} \in \mathbb{R}^d, \mathbf{v} \in \mathbb{R}^d\}
$$

Each point in phase space represents a complete **state** of the system: both where it is ($\mathbf{x}$) and how it's moving ($\mathbf{v}$).

### Position Space vs Phase Space

| Property | Position Space | Phase Space |
|----------|---------------|-------------|
| Dimension | $d$ | $2d$ |
| State | Position $\mathbf{x}$ | Position + momentum $(\mathbf{x}, \mathbf{v})$ |
| Dynamics | None (static) | Hamilton's equations |
| Target | $\pi(\mathbf{x})$ | $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})$ |

---

## The Joint Distribution

### Construction

Given target $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$, define the joint distribution on phase space:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v}))
$$

where the Hamiltonian is:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v}) = -\log \tilde{\pi}(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}
$$

### Factorization

Because the Hamiltonian separates into position and momentum terms:

$$
\pi(\mathbf{x}, \mathbf{v}) = \frac{1}{Z} \exp(-U(\mathbf{x})) \exp(-K(\mathbf{v})) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v}; \mathbf{0}, \mathbf{M})
$$

**Key properties**:

1. **Independence**: Position and momentum are independent under the joint distribution
2. **Marginal recovery**: $\int \pi(\mathbf{x}, \mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x})$
3. **Easy momentum sampling**: $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$ is straightforward

### The Normalization Constant

The partition function factors:

$$
Z = \int \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v} = Z_x \cdot Z_v
$$

where $Z_x = \int \exp(-U(\mathbf{x})) \, d\mathbf{x}$ (intractable) and $Z_v = (2\pi)^{d/2} |\mathbf{M}|^{1/2}$ (known).

For MCMC, we never need to compute $Z_x$—we only need ratios of $\pi$.

---

## Energy Surfaces

### Definition

An **energy surface** (or **level set**) is the set of all phase space points with a fixed energy:

$$
\Sigma_E = \{(\mathbf{x}, \mathbf{v}) \in \Gamma : H(\mathbf{x}, \mathbf{v}) = E\}
$$

Since Hamiltonian dynamics conserves energy, trajectories are confined to these $(2d-1)$-dimensional surfaces.

### Geometry of Energy Surfaces

For the standard HMC Hamiltonian $H = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$:

**At a fixed position $\mathbf{x}$**: The constraint $K(\mathbf{v}) = E - U(\mathbf{x})$ defines an ellipsoid in momentum space (assuming $E > U(\mathbf{x})$):

$$
\frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v} = E - U(\mathbf{x})
$$

**At a fixed momentum $\mathbf{v}$**: The constraint $U(\mathbf{x}) = E - K(\mathbf{v})$ defines a level set of the potential energy (a contour of the target density).

### Energy and Probability

Higher energy corresponds to lower probability:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H(\mathbf{x}, \mathbf{v})) = \exp(-E)
$$

**The typical set**: Most of the probability mass lies not at the minimum of $H$ (the mode), but on a "shell" of intermediate energy. This is the **concentration of measure** phenomenon in high dimensions.

For a $d$-dimensional standard Gaussian, the typical energy is $E \approx d$ (not $E = 0$ at the mode).

---

## Volume Elements and Measures

### The Liouville Measure

The natural measure on phase space is the **Liouville measure**:

$$
d\mu = d\mathbf{x} \, d\mathbf{v} = dx_1 \cdots dx_d \, dv_1 \cdots dv_d
$$

This is the standard Lebesgue measure on $\mathbb{R}^{2d}$.

**Liouville's theorem**: Hamiltonian flow preserves the Liouville measure. If $\phi_t$ is the time-$t$ flow map, then for any measurable set $A$:

$$
\mu(\phi_t(A)) = \mu(A)
$$

### Microcanonical Measure

On an energy surface $\Sigma_E$, the natural measure is the **microcanonical measure**:

$$
d\sigma_E = \frac{d\mu}{|\nabla H|}
$$

where the denominator normalizes by the gradient magnitude (the "thickness" of the energy shell).

Hamiltonian flow also preserves this measure on each energy surface.

### Canonical Measure

The **canonical measure** (Boltzmann distribution) is:

$$
d\nu = \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v}
$$

This is the measure we want to sample from. Hamiltonian dynamics preserves this measure because it preserves $H$ (energy conservation) and preserves volume (Liouville's theorem).

---

## Symplectic Geometry of Phase Space

### The Symplectic Form

Phase space carries a fundamental geometric structure: the **symplectic 2-form**:

$$
\omega = \sum_{i=1}^{d} dv_i \wedge dx_i
$$

In matrix notation, if $\mathbf{z} = (\mathbf{x}, \mathbf{v})^T$, then:

$$
\omega(\mathbf{u}, \mathbf{w}) = \mathbf{u}^T \mathbf{J} \mathbf{w}, \quad \text{where } \mathbf{J} = \begin{pmatrix} \mathbf{0} & -\mathbf{I} \\ \mathbf{I} & \mathbf{0} \end{pmatrix}
$$

### Properties of the Symplectic Form

1. **Antisymmetric**: $\omega(\mathbf{u}, \mathbf{w}) = -\omega(\mathbf{w}, \mathbf{u})$
2. **Non-degenerate**: If $\omega(\mathbf{u}, \mathbf{w}) = 0$ for all $\mathbf{w}$, then $\mathbf{u} = \mathbf{0}$
3. **Closed**: $d\omega = 0$ (the exterior derivative vanishes)

### Symplectic Volume

The symplectic form induces a volume form:

$$
\omega^d = \omega \wedge \omega \wedge \cdots \wedge \omega = d! \, dx_1 \wedge dv_1 \wedge \cdots \wedge dx_d \wedge dv_d
$$

This is (up to a constant) the Liouville measure.

### Darboux's Theorem

**Theorem**: Any symplectic manifold locally looks like standard phase space with coordinates $(\mathbf{x}, \mathbf{v})$ and symplectic form $\omega = \sum dv_i \wedge dx_i$.

This universality is why the standard Hamiltonian formulation works regardless of the specific target distribution.

---

## Canonical Transformations

### Definition

A smooth map $\phi: \Gamma \to \Gamma$ is a **canonical transformation** (or **symplectomorphism**) if it preserves the symplectic form:

$$
\phi^* \omega = \omega
$$

Equivalently, if $\mathbf{J}_\phi$ is the Jacobian matrix of $\phi$:

$$
\mathbf{J}_\phi^T \mathbf{J} \mathbf{J}_\phi = \mathbf{J}
$$

### Examples

**Time evolution**: The flow $\phi_t$ of any Hamiltonian system is canonical.

**Momentum scaling**: $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, c\mathbf{v})$ is canonical only if $c = \pm 1$.

**Linear canonical transformations**: Any matrix $\mathbf{A}$ satisfying $\mathbf{A}^T \mathbf{J} \mathbf{A} = \mathbf{J}$ defines a canonical transformation $\mathbf{z} \mapsto \mathbf{A}\mathbf{z}$.

**Momentum flip**: $(\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$ is canonical (and is its own inverse).

### Why Canonical Transformations Matter for HMC

1. **Leapfrog steps** are canonical transformations
2. **Composition** of leapfrog steps is canonical
3. **Volume preservation** follows automatically
4. **No Jacobian correction** needed in the MH ratio

---

## Phase Space Trajectories

### Hamiltonian Flow

Given initial conditions $(\mathbf{x}_0, \mathbf{v}_0)$, Hamilton's equations define a unique trajectory $(\mathbf{x}(t), \mathbf{v}(t))$.

**Properties of Hamiltonian trajectories**:

1. **Deterministic**: Given initial conditions, the trajectory is unique
2. **Reversible**: Running time backward (and flipping momentum) retraces the path
3. **Non-intersecting**: Trajectories cannot cross in phase space (uniqueness of solutions)
4. **Energy-preserving**: The trajectory stays on $\Sigma_{H(\mathbf{x}_0, \mathbf{v}_0)}$

### Orbits and Periods

For bounded motion, trajectories may be **periodic** (return exactly to the starting point after time $T$), **quasi-periodic** (fill a torus densely but never exactly repeat), or **chaotic** (sensitive dependence on initial conditions, in non-integrable systems).

For Gaussian targets, HMC trajectories are periodic or quasi-periodic (integrable system).

### Phase Portraits

A **phase portrait** visualizes trajectories in phase space. For a 1D system $(x, v)$:

**Harmonic oscillator** ($U(x) = \frac{1}{2}kx^2$): Trajectories are ellipses centered at the origin. Higher energy = larger ellipse.

**Double well** ($U(x) = (x^2 - 1)^2$): Two stable equilibria at $x = \pm 1$, an unstable equilibrium at $x = 0$. Low-energy trajectories oscillate around one well; high-energy trajectories traverse both.

```
        v (momentum)
        ↑
        |     ╭─────╮
        |   ╭─┘     └─╮    Energy contours
        |  ╭┘  mode   └╮
        | ╭┘    •      └╮
    ────┼─┼─────────────┼────→ x (position)
        | ╰╮           ╭╯
        |  ╰╮         ╭╯
        |   ╰─╮     ╭─╯
        |     ╰─────╯
```

Trajectories follow these contours (constant energy). The mode is at the center; the typical set is the ring at intermediate energy.

---

## The Typical Set in Phase Space

### Concentration of Measure

In high dimensions, probability concentrates on a thin shell rather than at the mode. For the joint distribution $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H)$:

**Mode**: The maximum of $\pi$ occurs at minimum $H$, typically $\mathbf{x} = \mathbf{x}^*$ (mode of target), $\mathbf{v} = \mathbf{0}$.

**Typical set**: Most probability mass lies where $H \approx \mathbb{E}[H] = \mathbb{E}[U] + \mathbb{E}[K]$.

For a $d$-dimensional standard Gaussian target with $\mathbf{M} = \mathbf{I}$:

- $\mathbb{E}[U] = \frac{d}{2}$, $\mathbb{E}[K] = \frac{d}{2}$
- Typical energy: $H \approx d$
- The typical set is an energy shell of width $O(\sqrt{d})$ around $H = d$

### Implications for Sampling

**Starting at the mode is bad**: A sample at $(\mathbf{x}^*, \mathbf{0})$ has atypically low energy. Hamiltonian dynamics will explore, but the trajectory stays on this low-energy surface.

**Momentum resampling is essential**: Drawing fresh $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{M})$ samples kinetic energy from its marginal distribution, allowing exploration of typical energy surfaces.

---

## Projection and Marginalization

### From Phase Space to Position Space

After HMC generates samples $\{(\mathbf{x}^{(t)}, \mathbf{v}^{(t)})\}$ from $\pi(\mathbf{x}, \mathbf{v})$, we extract position samples $\{\mathbf{x}^{(t)}\}$.

**Theorem**: If $(\mathbf{x}, \mathbf{v}) \sim \pi(\mathbf{x}, \mathbf{v})$, then $\mathbf{x} \sim \pi(\mathbf{x})$.

**Proof**: By construction, $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \pi(\mathbf{v})$ with $\pi(\mathbf{v}) = \mathcal{N}(\mathbf{0}, \mathbf{M})$. Integrating out $\mathbf{v}$:

$$
\int \pi(\mathbf{x}, \mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x}) \int \pi(\mathbf{v}) \, d\mathbf{v} = \pi(\mathbf{x})
$$

### Discarding Momentum

We discard the momentum samples—they carry no information about the target $\pi(\mathbf{x})$. The momentum's role is purely auxiliary: enable deterministic dynamics, carry gradient information between steps, and allow coherent exploration.

---

## Phase Space for Non-Euclidean Targets

### Manifold-Valued Positions

When $\mathbf{x}$ lives on a manifold $\mathcal{M}$ (e.g., sphere, torus, positive definite matrices), phase space becomes the **cotangent bundle** $T^*\mathcal{M}$.

At each point $\mathbf{x} \in \mathcal{M}$, the momentum $\mathbf{v}$ lives in the cotangent space $T^*_\mathbf{x}\mathcal{M}$.

### Riemannian Structure

For Riemannian HMC, the mass matrix $\mathbf{M}(\mathbf{x})$ depends on position, defining a metric on the manifold. The kinetic energy becomes:

$$
K(\mathbf{x}, \mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}(\mathbf{x})^{-1} \mathbf{v}
$$

The phase space structure is more complex: the Hamiltonian is no longer separable, standard leapfrog doesn't apply, and implicit or generalized integrators are needed.

---

## Summary

| Concept | Definition | Importance |
|---------|------------|------------|
| Phase space | $\Gamma = \{(\mathbf{x}, \mathbf{v})\}$ | Arena for HMC dynamics |
| Joint distribution | $\pi(\mathbf{x}, \mathbf{v}) = \pi(\mathbf{x}) \cdot \mathcal{N}(\mathbf{v})$ | Factorizes; marginal is target |
| Energy surface | $\Sigma_E = \{H = E\}$ | Trajectories confined here |
| Symplectic form | $\omega = \sum dv_i \wedge dx_i$ | Preserved by Hamiltonian flow |
| Canonical transformation | Preserves $\omega$ | Volume-preserving; no Jacobian |
| Typical set | $H \approx \mathbb{E}[H]$ | Where samples concentrate |

Phase space geometry provides the mathematical foundation for understanding why HMC works: the symplectic structure ensures volume preservation, energy surfaces constrain trajectories, and the factored joint distribution allows easy marginalization to recover the target.

---

## Exercises

1. **Energy surface visualization**. For a 1D Gaussian target $\pi(x) \propto \exp(-x^2/2)$ with $M = 1$, sketch the energy contours in the $(x, v)$ phase plane. What shape are they? Where is the mode? Where is the typical set?

2. **Symplectic verification**. Show that the map $(x, v) \mapsto (x \cos\theta + v\sin\theta, -x\sin\theta + v\cos\theta)$ is canonical. What Hamiltonian generates this flow?

3. **Typical set calculation**. For a $d$-dimensional standard Gaussian target with $\mathbf{M} = \mathbf{I}$, compute $\mathbb{E}[H]$ and $\text{Var}[H]$. How does the width of the typical set scale with $d$?

4. **Non-factorized joint**. Suppose we used $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-U(\mathbf{x}) - \frac{1}{2}\mathbf{v}^T\mathbf{M}(\mathbf{x})^{-1}\mathbf{v})$ where $\mathbf{M}$ depends on $\mathbf{x}$. Does the joint still factorize? What is the marginal over $\mathbf{x}$?

---

## References

1. Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*. Springer.
2. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. da Silva, A. C. (2001). *Lectures on Symplectic Geometry*. Springer.
