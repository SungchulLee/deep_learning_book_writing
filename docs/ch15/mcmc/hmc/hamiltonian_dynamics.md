# Hamiltonian Dynamics

Hamiltonian dynamics is the mathematical framework from classical mechanics that underlies Hamiltonian Monte Carlo. This section develops the physics foundations: the Hamiltonian formulation of mechanics, symplectic structure, conservation laws, and why these properties make HMC possible.

---

## From Newtonian to Hamiltonian Mechanics

### The Lagrangian Formulation

Classical mechanics began with Newton's second law $\mathbf{F} = m\mathbf{a}$, but the **Lagrangian formulation** provides a more elegant approach. For a system with generalized coordinates $\mathbf{q}$ and velocities $\dot{\mathbf{q}}$, define the Lagrangian:

$$
L(\mathbf{q}, \dot{\mathbf{q}}) = T(\dot{\mathbf{q}}) - U(\mathbf{q})
$$

where $T$ is kinetic energy and $U$ is potential energy. The equations of motion follow from the **principle of least action**: the path taken by the system extremizes the action integral $S = \int L \, dt$.

The Euler-Lagrange equations are:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_i} - \frac{\partial L}{\partial q_i} = 0
$$

### The Legendre Transform

The **Hamiltonian formulation** transforms from velocities to momenta via the Legendre transform. Define the **conjugate momentum**:

$$
p_i = \frac{\partial L}{\partial \dot{q}_i}
$$

For a particle with kinetic energy $T = \frac{1}{2}m|\dot{\mathbf{q}}|^2$, this gives $\mathbf{p} = m\dot{\mathbf{q}}$, the familiar momentum.

The **Hamiltonian** is obtained by the Legendre transform:

$$
H(\mathbf{q}, \mathbf{p}) = \mathbf{p} \cdot \dot{\mathbf{q}} - L(\mathbf{q}, \dot{\mathbf{q}})
$$

where $\dot{\mathbf{q}}$ is expressed in terms of $\mathbf{p}$ using $\mathbf{p} = \partial L / \partial \dot{\mathbf{q}}$.

**Key result**: For conservative systems (time-independent $L$), the Hamiltonian equals the total energy:

$$
H = T + U
$$

### Hamilton's Equations

The equations of motion in Hamiltonian form are:

$$
\frac{d\mathbf{q}}{dt} = \frac{\partial H}{\partial \mathbf{p}}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \mathbf{q}}
$$

These are **first-order** ODEs in $2d$ variables (position and momentum), compared to **second-order** ODEs in $d$ variables for Newtonian mechanics. The symmetric structure—one equation with $+$, one with $-$—is the hallmark of Hamiltonian systems.

**Example**: For a particle in a potential with $H = \frac{|\mathbf{p}|^2}{2m} + U(\mathbf{q})$:

$$
\frac{d\mathbf{q}}{dt} = \frac{\mathbf{p}}{m}, \quad \frac{d\mathbf{p}}{dt} = -\nabla U(\mathbf{q})
$$

The first equation says velocity equals momentum over mass; the second is Newton's law $\dot{\mathbf{p}} = \mathbf{F}$.

---

## The Symplectic Structure

### Hamiltonian Vector Fields

Phase space has an intrinsic geometric structure given by the **symplectic form**:

$$
\omega = \sum_{i=1}^{d} dp_i \wedge dq_i
$$

This 2-form measures "oriented area" in phase space. The symplectic form is **closed** ($d\omega = 0$) and **non-degenerate** (for any nonzero tangent vector $\mathbf{v}$, there exists $\mathbf{w}$ with $\omega(\mathbf{v}, \mathbf{w}) \neq 0$).

Given a Hamiltonian $H$, the **Hamiltonian vector field** $X_H$ is defined implicitly by:

$$
\omega(X_H, \cdot) = dH
$$

In coordinates, this gives:

$$
X_H = \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{\partial}{\partial \mathbf{q}} - \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{\partial}{\partial \mathbf{p}}
$$

The integral curves of $X_H$ are the solutions to Hamilton's equations.

**Darboux's theorem**: All symplectic manifolds of the same dimension are locally equivalent. This is why the standard $(\mathbf{q}, \mathbf{p})$ coordinates exist universally.

### Symplectic Maps

A diffeomorphism $\phi: (\mathbf{q}, \mathbf{p}) \mapsto (\mathbf{Q}, \mathbf{P})$ is **symplectic** (or **canonical**) if it preserves the symplectic form:

$$
\phi^* \omega = \omega
$$

Equivalently, in matrix form, if $\mathbf{J} = \frac{\partial(\mathbf{Q}, \mathbf{P})}{\partial(\mathbf{q}, \mathbf{p})}$ is the Jacobian, then:

$$
\mathbf{J}^T \mathbf{\Omega} \mathbf{J} = \mathbf{\Omega}, \quad \text{where } \mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}
$$

**Properties of symplectic maps**:

1. **Volume preservation**: $|\det \mathbf{J}| = 1$ (follows from $\det(\mathbf{J}^T \mathbf{\Omega} \mathbf{J}) = \det \mathbf{\Omega}$)
2. **Composition**: If $\phi_1$ and $\phi_2$ are symplectic, so is $\phi_1 \circ \phi_2$
3. **Inverse**: If $\phi$ is symplectic, so is $\phi^{-1}$
4. **Hamiltonian flows are symplectic**: The time-$t$ flow map $\phi_t$ of any Hamiltonian system is symplectic

### Generating Functions

Symplectic maps can be characterized by **generating functions**. For a map $(\mathbf{q}, \mathbf{p}) \mapsto (\mathbf{Q}, \mathbf{P})$, if there exists $S(\mathbf{q}, \mathbf{Q})$ such that:

$$
\mathbf{p} = \frac{\partial S}{\partial \mathbf{q}}, \quad \mathbf{P} = -\frac{\partial S}{\partial \mathbf{Q}}
$$

then the map is symplectic. This is the basis for some advanced HMC variants.

---

## Conservation Laws

### Energy Conservation

**Theorem**: Along solutions of Hamilton's equations, the Hamiltonian is constant.

**Proof**: 
$$
\frac{dH}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{d\mathbf{q}}{dt} + \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{d\mathbf{p}}{dt} = \frac{\partial H}{\partial \mathbf{q}} \cdot \frac{\partial H}{\partial \mathbf{p}} - \frac{\partial H}{\partial \mathbf{p}} \cdot \frac{\partial H}{\partial \mathbf{q}} = 0
$$

The cross terms cancel exactly due to the antisymmetric structure of Hamilton's equations.

**Importance for HMC**: Energy conservation is the key property that enables long trajectories without drift. It ensures that proposals far from the starting point still have high acceptance probability.

### Phase Space Volume Preservation (Liouville's Theorem)

**Theorem** (Liouville): Hamiltonian flow preserves phase space volume.

Consider a region $\Omega$ in phase space evolving under Hamilton's equations. Let $\Omega_t$ be its image at time $t$. Then:

$$
\text{Vol}(\Omega_t) = \text{Vol}(\Omega_0) \quad \text{for all } t
$$

**Proof**: The velocity field in phase space is $\mathbf{v} = (\dot{\mathbf{q}}, \dot{\mathbf{p}})$. The divergence is:

$$
\nabla \cdot \mathbf{v} = \sum_i \left( \frac{\partial \dot{q}_i}{\partial q_i} + \frac{\partial \dot{p}_i}{\partial p_i} \right) = \sum_i \left( \frac{\partial^2 H}{\partial q_i \partial p_i} - \frac{\partial^2 H}{\partial p_i \partial q_i} \right) = 0
$$

By the continuity equation, zero divergence implies volume preservation.

**Importance for HMC**: Volume preservation means no Jacobian correction is needed in the Metropolis-Hastings acceptance ratio.

### Poincaré Recurrence

**Theorem** (Poincaré): For a Hamiltonian system on a bounded energy surface, almost every trajectory returns arbitrarily close to its starting point.

This follows from volume preservation: if a region $\Omega$ evolved to disjoint regions forever, the total volume would grow unboundedly, contradicting Liouville's theorem.

**Implication**: Hamiltonian dynamics is fundamentally **recurrent**, not convergent. This is why HMC samples rather than optimizes.

---

## Time Reversibility

### Definition

A dynamical system is **time-reversible** if there exists an involution $R$ (meaning $R^2 = \text{identity}$) such that:

$$
\phi_{-t} = R \circ \phi_t \circ R
$$

where $\phi_t$ is the time-$t$ flow.

### Reversibility in Hamiltonian Systems

For Hamiltonian systems, the momentum-flip map $R: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$ provides reversibility.

**Why it works**: Under $\mathbf{v} \mapsto -\mathbf{v}$:

- The kinetic energy $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T\mathbf{M}^{-1}\mathbf{v}$ is unchanged (even in $\mathbf{v}$)
- Hamilton's equations become $\frac{d\mathbf{x}}{d(-t)} = -\mathbf{M}^{-1}\mathbf{v}$, which equals $\frac{d\mathbf{x}}{dt}$ after flipping $\mathbf{v}$

**Importance for HMC**: Time reversibility ensures the proposal is symmetric in an appropriate sense, simplifying the Metropolis-Hastings acceptance criterion.

---

## Noether's Theorem and Symmetries

### Statement

**Noether's theorem**: Every continuous symmetry of the Hamiltonian corresponds to a conserved quantity.

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy |
| Space translation | Linear momentum |
| Rotation | Angular momentum |

### Implications for Sampling

The only symmetry we use for basic HMC is time translation (energy conservation). However, **rotational symmetry** in the target can cause trajectories to miss modes (conserved angular momentum constrains the orbit), and **discrete symmetries** (e.g., permutation invariance) can affect sampling efficiency.

---

## The Hamiltonian for Sampling

### Standard Form

For sampling from $\pi(\mathbf{x}) \propto \exp(-U(\mathbf{x}))$, we use:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})
$$

where:

- **Position** $\mathbf{x}$ replaces $\mathbf{q}$ (the variable to sample)
- **Momentum** $\mathbf{v}$ replaces $\mathbf{p}$ (auxiliary variable)
- **Potential energy**: $U(\mathbf{x}) = -\log \tilde{\pi}(\mathbf{x})$
- **Kinetic energy**: $K(\mathbf{v}) = \frac{1}{2}\mathbf{v}^T \mathbf{M}^{-1} \mathbf{v}$

The corresponding Hamilton's equations:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = -\nabla U(\mathbf{x}) = \nabla \log \pi(\mathbf{x})
$$

The momentum equation shows that **the score function acts as a force**.

### Separable Hamiltonians

A Hamiltonian is **separable** if $H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + K(\mathbf{v})$ with no cross terms. The standard HMC Hamiltonian is separable.

**Why separability matters**: For separable Hamiltonians, Hamilton's equations decouple—$\frac{d\mathbf{x}}{dt}$ depends only on $\mathbf{v}$ and $\frac{d\mathbf{v}}{dt}$ depends only on $\mathbf{x}$. This structure enables **operator splitting** methods like the leapfrog integrator, where each sub-step can be solved exactly and the composition approximates the full dynamics.

### Non-Separable Hamiltonians

Some advanced HMC variants use non-separable Hamiltonians:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{M}(\mathbf{x})^{-1} \mathbf{v} + \frac{1}{2}\log|\mathbf{M}(\mathbf{x})|
$$

where the mass matrix depends on position. This is **Riemannian HMC** and requires more sophisticated integrators.

---

## Why Hamiltonian Dynamics Enables Sampling

### The Physical Intuition

Consider a particle in a potential well:

1. **Gradient descent** would roll the particle to the bottom and stop
2. **Hamiltonian dynamics** converts potential energy to kinetic energy as the particle falls
3. The particle **overshoots** the minimum and climbs the other side
4. It oscillates indefinitely, exploring the well

The key insight: **conservation of energy prevents convergence to the mode**.

### Mathematical Explanation

For optimization (gradient descent), energy decreases:
$$
\frac{dU}{dt} = \nabla U \cdot \frac{d\mathbf{x}}{dt} = -|\nabla U|^2 \leq 0
$$

For Hamiltonian dynamics, total energy is constant but $U$ and $K$ exchange:
$$
\frac{dU}{dt} = -\frac{dK}{dt}
$$

What matters for sampling is not time spent at each point, but the **stationary distribution** of the dynamics. Energy conservation + volume preservation ensures the Boltzmann distribution $\pi(\mathbf{x}, \mathbf{v}) \propto \exp(-H)$ is stationary.

### Ergodicity and Mixing

For sampling to work, the dynamics must be **ergodic** (time averages equal ensemble averages) and **mixing** (initial conditions are "forgotten").

Pure Hamiltonian dynamics is often not ergodic—trajectories stay on fixed energy surfaces. HMC achieves ergodicity by **resampling momentum** at each iteration: the random momentum injects fresh kinetic energy, allowing exploration of different energy surfaces.

---

## Connection to Statistical Mechanics

### The Boltzmann Distribution

In statistical mechanics, a system in thermal equilibrium at temperature $T$ has state distribution:

$$
\pi(\mathbf{x}, \mathbf{v}) \propto \exp\left(-\frac{H(\mathbf{x}, \mathbf{v})}{k_B T}\right)
$$

where $k_B$ is Boltzmann's constant. For sampling, we set $k_B T = 1$.

### Partition Function and Free Energy

The **partition function** is $Z = \int \exp(-H(\mathbf{x}, \mathbf{v})) \, d\mathbf{x} \, d\mathbf{v}$ and the **free energy** is $F = -\log Z$. Computing $Z$ is typically intractable, which is why MCMC samples rather than integrating directly.

### Microcanonical vs Canonical Ensemble

- **Microcanonical** (constant energy): States uniformly distributed on energy surface $H = E$
- **Canonical** (constant temperature): States distributed as $\exp(-H)$

HMC operates in the canonical ensemble. The momentum resampling step maintains the temperature.

---

## Summary

| Concept | Description | Role in HMC |
|---------|-------------|-------------|
| Hamiltonian | Total energy $H = U + K$ | Defines the dynamics |
| Phase space | Space of $(\mathbf{x}, \mathbf{v})$ | Extended state space |
| Hamilton's equations | $\dot{\mathbf{x}} = \partial_\mathbf{v} H$, $\dot{\mathbf{v}} = -\partial_\mathbf{x} H$ | Govern evolution |
| Energy conservation | $dH/dt = 0$ | High acceptance rates |
| Volume preservation | $\det \mathbf{J} = 1$ | No Jacobian correction |
| Time reversibility | $\phi_{-t} = R \circ \phi_t \circ R$ | Detailed balance |
| Separability | $H = U(\mathbf{x}) + K(\mathbf{v})$ | Enables leapfrog |

The power of HMC comes from these classical mechanics principles: energy conservation prevents convergence to modes, volume preservation eliminates Jacobian corrections, and the symplectic structure enables efficient numerical integration.

---

## Exercises

1. **Verify Hamilton's equations**. For $H = \frac{p^2}{2m} + \frac{1}{2}kx^2$ (harmonic oscillator), derive Hamilton's equations and solve them. Show that the solution is periodic with no convergence to $x = 0$.

2. **Symplectic verification**. Show that the map $(q, p) \mapsto (q + \epsilon p, p)$ is symplectic. Show that $(q, p) \mapsto (q, p - \epsilon \nabla U(q))$ is also symplectic.

3. **Energy surface geometry**. For a 2D Gaussian target $\pi(x) \propto \exp(-\frac{1}{2}x^T \Sigma^{-1} x)$ with $\mathbf{M} = \mathbf{I}$, describe the energy surfaces in the 4D phase space. What shape are they?

4. **Non-separable Hamiltonian**. For $H(x, v) = U(x) + \frac{v^2}{2m(x)}$ with position-dependent mass, derive Hamilton's equations. Why is this more complex than the separable case?

---

## References

1. Arnold, V. I. (1989). *Mathematical Methods of Classical Mechanics*. Springer.
2. Goldstein, H., Poole, C., & Safko, J. (2002). *Classical Mechanics* (3rd ed.). Addison Wesley.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
