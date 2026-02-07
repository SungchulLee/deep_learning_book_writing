# Leapfrog Integrator

The leapfrog integrator is the numerical engine of Hamiltonian Monte Carlo. This section develops the integrator from first principles: operator splitting, the Störmer-Verlet scheme, symplectic structure, error analysis, and practical implementation considerations.

---

## The Numerical Integration Problem

### Why We Need Numerical Integration

Hamilton's equations form a system of ODEs:

$$
\frac{d\mathbf{x}}{dt} = \frac{\partial H}{\partial \mathbf{v}} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = -\frac{\partial H}{\partial \mathbf{x}} = -\nabla U(\mathbf{x})
$$

For most target distributions, these equations have no closed-form solution. We must integrate numerically.

### Requirements for MCMC

Not any numerical integrator will do. For valid MCMC sampling, we need:

| Property | Requirement | Why It Matters |
|----------|-------------|----------------|
| **Symplectic** | Preserve phase space volume | No Jacobian correction in MH ratio |
| **Reversible** | Time-symmetric | Detailed balance |
| **Accurate** | Bounded energy error | High acceptance rate |
| **Efficient** | Few gradient evaluations | Computational cost |

The leapfrog integrator achieves all four.

### What Goes Wrong with Standard Methods

**Euler method**:
$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_n, \quad \mathbf{v}_{n+1} = \mathbf{v}_n - \epsilon \nabla U(\mathbf{x}_n)
$$

Problems:
- Not symplectic (volume-expanding or contracting)
- Not reversible
- Energy drifts systematically over time
- Acceptance rate drops to zero for long trajectories

**Runge-Kutta methods** (RK4, etc.):
- High accuracy per step
- But not symplectic
- Energy drifts, though more slowly
- Unsuitable for long MCMC trajectories

---

## Operator Splitting

### The Splitting Idea

The Hamiltonian $H = U(\mathbf{x}) + K(\mathbf{v})$ generates dynamics via two operators:

**Operator $\mathcal{A}$** (from potential energy $U$):
$$
\frac{d\mathbf{x}}{dt} = 0, \quad \frac{d\mathbf{v}}{dt} = -\nabla U(\mathbf{x})
$$

**Operator $\mathcal{B}$** (from kinetic energy $K$):
$$
\frac{d\mathbf{x}}{dt} = \mathbf{M}^{-1}\mathbf{v}, \quad \frac{d\mathbf{v}}{dt} = 0
$$

Each operator alone is exactly solvable:

$$
e^{\epsilon \mathcal{A}}: \quad \mathbf{v} \to \mathbf{v} - \epsilon \nabla U(\mathbf{x}), \quad \mathbf{x} \to \mathbf{x}
$$

$$
e^{\epsilon \mathcal{B}}: \quad \mathbf{x} \to \mathbf{x} + \epsilon \mathbf{M}^{-1}\mathbf{v}, \quad \mathbf{v} \to \mathbf{v}
$$

The full dynamics $e^{\epsilon(\mathcal{A} + \mathcal{B})}$ mixes both effects and cannot be computed exactly.

### The Baker-Campbell-Hausdorff Formula

For non-commuting operators, the BCH formula gives:

$$
e^{\epsilon \mathcal{A}} e^{\epsilon \mathcal{B}} = \exp\left(\epsilon(\mathcal{A} + \mathcal{B}) + \frac{\epsilon^2}{2}[\mathcal{A}, \mathcal{B}] + O(\epsilon^3)\right)
$$

where $[\mathcal{A}, \mathcal{B}] = \mathcal{A}\mathcal{B} - \mathcal{B}\mathcal{A}$ is the commutator.

**First-order splitting** (Lie-Trotter):
$$
e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\epsilon \mathcal{A}} e^{\epsilon \mathcal{B}} + O(\epsilon^2)
$$

**Second-order splitting** (Strang):
$$
e^{\epsilon(\mathcal{A} + \mathcal{B})} \approx e^{\frac{\epsilon}{2} \mathcal{A}} e^{\epsilon \mathcal{B}} e^{\frac{\epsilon}{2} \mathcal{A}} + O(\epsilon^3)
$$

The symmetric structure of Strang splitting cancels the $O(\epsilon^2)$ error term.

---

## The Leapfrog Algorithm

### Derivation from Strang Splitting

Applying $e^{\frac{\epsilon}{2} \mathcal{A}} e^{\epsilon \mathcal{B}} e^{\frac{\epsilon}{2} \mathcal{A}}$ to state $(\mathbf{x}_n, \mathbf{v}_n)$:

**Step 1** — Half momentum update ($e^{\frac{\epsilon}{2}\mathcal{A}}$):
$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n - \frac{\epsilon}{2} \nabla U(\mathbf{x}_n)
$$

**Step 2** — Full position update ($e^{\epsilon \mathcal{B}}$):
$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1} \mathbf{v}_{n+1/2}
$$

**Step 3** — Half momentum update ($e^{\frac{\epsilon}{2}\mathcal{A}}$):
$$
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} - \frac{\epsilon}{2} \nabla U(\mathbf{x}_{n+1})
$$

### The "Leapfrog" Name

The name comes from the staggered time points:
- Positions are defined at integer times: $\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \ldots$
- Momenta "leap over" positions at half-integer times: $\mathbf{v}_{1/2}, \mathbf{v}_{3/2}, \ldots$

```
Time:     0      ½      1      3/2     2
          |      |      |      |       |
Position: x₀ -------- x₁ -------- x₂ ---
Momentum: v₀ -- v½ -------- v3/2 -------
```

### Equivalent Formulations

**Position-first (PVP)**:
$$
\mathbf{x}_{n+1/2} = \mathbf{x}_n + \frac{\epsilon}{2}\mathbf{M}^{-1}\mathbf{v}_n, \quad
\mathbf{v}_{n+1} = \mathbf{v}_n - \epsilon \nabla U(\mathbf{x}_{n+1/2}), \quad
\mathbf{x}_{n+1} = \mathbf{x}_{n+1/2} + \frac{\epsilon}{2}\mathbf{M}^{-1}\mathbf{v}_{n+1}
$$

**Momentum-first (VPV)** — the standard form:
$$
\mathbf{v}_{n+1/2} = \mathbf{v}_n - \frac{\epsilon}{2}\nabla U(\mathbf{x}_n), \quad
\mathbf{x}_{n+1} = \mathbf{x}_n + \epsilon \mathbf{M}^{-1}\mathbf{v}_{n+1/2}, \quad
\mathbf{v}_{n+1} = \mathbf{v}_{n+1/2} - \frac{\epsilon}{2}\nabla U(\mathbf{x}_{n+1})
$$

Both are equivalent and second-order accurate.

---

## Multiple Leapfrog Steps

### Composing Steps

For $L$ leapfrog steps, we compose the single-step map $L$ times. The half-steps at boundaries combine:

```python
def leapfrog(x, v, epsilon, L, grad_U, M_inv):
    # Initial half-step for momentum
    v = v - (epsilon / 2) * grad_U(x)
    
    # Full steps
    for i in range(L - 1):
        x = x + epsilon * M_inv @ v
        v = v - epsilon * grad_U(x)
    
    # Final position update
    x = x + epsilon * M_inv @ v
    
    # Final half-step for momentum
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

**Gradient evaluations**: $L$ steps require $L + 1$ gradient evaluations (not $2L$), because adjacent half-steps merge.

### Trajectory Length

The total integration time is $T = L\epsilon$. This determines how far the trajectory travels in phase space.

**Trade-off**:
- Larger $T$: Better exploration, but more computation
- Smaller $T$: Cheaper, but more correlated samples

Typical values: $L = 10$–$100$, $\epsilon = 0.01$–$0.1$, giving $T = 0.1$–$10$.

---

## Symplectic Property

### Definition

A map $\phi: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{X}, \mathbf{V})$ is **symplectic** if its Jacobian $\mathbf{J}$ satisfies:

$$
\mathbf{J}^T \mathbf{\Omega} \mathbf{J} = \mathbf{\Omega}, \quad \text{where } \mathbf{\Omega} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}
$$

### Why Leapfrog is Symplectic

Each elementary update is symplectic:

**Momentum update** $\mathbf{v} \to \mathbf{v} - \epsilon \nabla U(\mathbf{x})$:

$$
\mathbf{J}_v = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ -\epsilon \nabla^2 U & \mathbf{I} \end{pmatrix}
$$

Check: $\mathbf{J}_v^T \mathbf{\Omega} \mathbf{J}_v = \mathbf{\Omega}$ ✓ (shear transformation)

**Position update** $\mathbf{x} \to \mathbf{x} + \epsilon \mathbf{M}^{-1}\mathbf{v}$:

$$
\mathbf{J}_x = \begin{pmatrix} \mathbf{I} & \epsilon \mathbf{M}^{-1} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}
$$

Check: $\mathbf{J}_x^T \mathbf{\Omega} \mathbf{J}_x = \mathbf{\Omega}$ ✓ (shear transformation)

**Composition**: The product of symplectic maps is symplectic.

### Consequences

1. **Volume preservation**: $|\det \mathbf{J}| = 1$

2. **No Jacobian in MH ratio**: The proposal density ratio simplifies to just the energy difference

3. **Long-term stability**: Energy stays bounded (no systematic drift)

---

## Time Reversibility

### Definition

A map $\phi$ is **time-reversible** with respect to momentum flip $R: (\mathbf{x}, \mathbf{v}) \mapsto (\mathbf{x}, -\mathbf{v})$ if:

$$
\phi^{-1} = R \circ \phi \circ R
$$

Equivalently: apply $\phi$, flip momentum, apply $\phi$ again, flip momentum → return to start.

### Proof for Leapfrog

The VPV leapfrog can be written as $\phi = A_{1/2} \circ B \circ A_{1/2}$ where:
- $A_{1/2}$: half momentum update
- $B$: full position update

Both $A_{1/2}$ and $B$ are symmetric under $R$-conjugation:
- $R \circ A_{1/2} \circ R = A_{1/2}^{-1}$ (momentum update reverses sign)
- $R \circ B \circ R = B^{-1}$ (position update reverses with flipped momentum)

Therefore:
$$
R \circ \phi \circ R = R \circ A_{1/2} \circ B \circ A_{1/2} \circ R = A_{1/2}^{-1} \circ B^{-1} \circ A_{1/2}^{-1} = \phi^{-1}
$$

### Importance for MCMC

Time reversibility ensures **detailed balance** when combined with the MH acceptance step. The proposal $(\mathbf{x}, \mathbf{v}) \to (\mathbf{x}', \mathbf{v}')$ has the same probability as the reverse $(\mathbf{x}', -\mathbf{v}') \to (\mathbf{x}, -\mathbf{v})$.

---

## Error Analysis

### Local Truncation Error

The leapfrog integrator approximates the exact flow $\phi_\epsilon^{\text{exact}}$ with error:

$$
\phi_\epsilon^{\text{leapfrog}} = \phi_\epsilon^{\text{exact}} + O(\epsilon^3)
$$

The local error is $O(\epsilon^3)$ per step (third-order accurate locally).

### Global Error

After $L$ steps covering time $T = L\epsilon$:

$$
\text{Global error} = O(L \cdot \epsilon^3) = O(T \cdot \epsilon^2)
$$

The global error is $O(\epsilon^2)$ (second-order accurate globally).

### Energy Error

For symplectic integrators, energy oscillates but doesn't drift:

**Backward error analysis**: The leapfrog integrator exactly solves a **modified Hamiltonian**:

$$
\tilde{H}(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, \mathbf{v}) + \epsilon^2 H_2(\mathbf{x}, \mathbf{v}) + O(\epsilon^4)
$$

where $H_2$ is a correction term.

**Consequence**: Energy error $|H(\mathbf{x}_L, \mathbf{v}_L) - H(\mathbf{x}_0, \mathbf{v}_0)| = O(\epsilon^2)$, bounded uniformly in $L$.

This is why symplectic integrators excel for long-time integration: energy error doesn't accumulate.

### Energy Error and Acceptance

The MH acceptance probability depends on energy error:

$$
\alpha = \min(1, \exp(-\Delta H))
$$

where $\Delta H = H(\mathbf{x}', \mathbf{v}') - H(\mathbf{x}, \mathbf{v})$.

| Energy Error $\Delta H$ | Acceptance $\alpha$ |
|------------------------|---------------------|
| 0 | 1.00 |
| 0.1 | 0.90 |
| 0.5 | 0.61 |
| 1.0 | 0.37 |
| 2.0 | 0.14 |

**Target**: $\Delta H \approx 0.1$–$1.0$ for acceptance rates of 60–90%.

---

## Step Size Selection

### The Stability Limit

Leapfrog has a **stability limit**: if $\epsilon$ exceeds a threshold, trajectories diverge exponentially.

For a quadratic potential $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x}$ with $\mathbf{M} = \mathbf{I}$:

$$
\epsilon < \frac{2}{\sqrt{\lambda_{\max}(\mathbf{A})}}
$$

where $\lambda_{\max}$ is the largest eigenvalue of $\mathbf{A}$ (the Hessian of $U$).

**Interpretation**: Step size must be small enough to resolve the fastest oscillation in the system.

### Optimal Step Size

**Too small**: Many steps needed, wasted computation, high autocorrelation.

**Too large**: Energy error grows, acceptance drops, wasted computation (rejected proposals).

**Optimal**: Balance exploration against acceptance. Empirically, target acceptance rate ~65–80%.

**Scaling with dimension**: For well-conditioned problems, optimal $\epsilon \sim d^{-1/4}$.

### Adaptive Step Size

During warmup, adjust $\epsilon$ to achieve target acceptance:

```python
def adapt_step_size(epsilon, accept_rate, target=0.65):
    if accept_rate > target:
        epsilon *= 1.1  # Can afford larger steps
    else:
        epsilon *= 0.9  # Need smaller steps
    return epsilon
```

More sophisticated: dual averaging (Nesterov, 2009) used in Stan.

---

## Numerical Stability

### Gradient Explosions

If $\nabla U(\mathbf{x})$ becomes very large (e.g., near boundaries, in heavy tails), the momentum update can overshoot catastrophically.

**Symptoms**:
- NaN or Inf values
- Extremely large energy changes
- Zero acceptance rate

**Remedies**:
- Reduce step size
- Use gradient clipping (carefully—breaks detailed balance)
- Reparameterize the model
- Use softplus or other transformations for constrained parameters

### Numerical Precision

For very small $\epsilon$ or very long trajectories:

**Accumulation error**: Repeated additions can lose precision. Use compensated summation (Kahan) if needed.

**Hessian condition number**: If $\kappa = \lambda_{\max}/\lambda_{\min}$ is large, different directions need different step sizes. The mass matrix addresses this.

---

## Implementation

### Basic Implementation

```python
import numpy as np

def leapfrog(x, v, epsilon, L, grad_U, M_inv=None):
    """
    Leapfrog integrator for HMC.
    
    Args:
        x: Initial position (d,)
        v: Initial momentum (d,)
        epsilon: Step size
        L: Number of leapfrog steps
        grad_U: Function returning gradient of potential energy
        M_inv: Inverse mass matrix (default: identity)
    
    Returns:
        x_new, v_new: Final position and momentum
    """
    if M_inv is None:
        M_inv = np.eye(len(x))
    
    x = x.copy()
    v = v.copy()
    
    # Half step for momentum
    v = v - (epsilon / 2) * grad_U(x)
    
    # Alternate full steps
    for i in range(L - 1):
        x = x + epsilon * (M_inv @ v)
        v = v - epsilon * grad_U(x)
    
    # Final full step for position
    x = x + epsilon * (M_inv @ v)
    
    # Half step for momentum
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

### With Diagnostics

```python
def leapfrog_with_diagnostics(x, v, epsilon, L, H, grad_U, M_inv=None):
    """Leapfrog with energy tracking for diagnostics."""
    if M_inv is None:
        M_inv = np.eye(len(x))
    
    x = x.copy()
    v = v.copy()
    
    energies = [H(x, v)]
    
    v = v - (epsilon / 2) * grad_U(x)
    
    for i in range(L - 1):
        x = x + epsilon * (M_inv @ v)
        v = v - epsilon * grad_U(x)
        energies.append(H(x, v))
    
    x = x + epsilon * (M_inv @ v)
    v = v - (epsilon / 2) * grad_U(x)
    energies.append(H(x, v))
    
    return x, v, np.array(energies)
```

### Vectorized for Multiple Chains

```python
def leapfrog_vectorized(x, v, epsilon, L, grad_U, M_inv=None):
    """
    Vectorized leapfrog for multiple chains.
    
    Args:
        x: Positions (n_chains, d)
        v: Momenta (n_chains, d)
        ...
    """
    if M_inv is None:
        M_inv = np.eye(x.shape[1])
    
    x = x.copy()
    v = v.copy()
    
    v = v - (epsilon / 2) * grad_U(x)
    
    for i in range(L - 1):
        x = x + epsilon * (v @ M_inv.T)
        v = v - epsilon * grad_U(x)
    
    x = x + epsilon * (v @ M_inv.T)
    v = v - (epsilon / 2) * grad_U(x)
    
    return x, v
```

---

## Higher-Order Integrators

### Fourth-Order Methods

Higher accuracy can be achieved with more stages. A fourth-order symplectic integrator:

$$
\phi_\epsilon^{(4)} = \phi_{c_1\epsilon} \circ \phi_{c_2\epsilon} \circ \phi_{c_3\epsilon} \circ \phi_{c_2\epsilon} \circ \phi_{c_1\epsilon}
$$

with carefully chosen coefficients $c_1, c_2, c_3$ (e.g., Forest-Ruth, Yoshida).

**Trade-off**: More gradient evaluations per step, but larger step sizes possible.

### When Higher Order Helps

- Very smooth targets where gradients are cheap
- High accuracy requirements
- When acceptance rate is already high but samples are correlated

**Usually not worth it**: The standard leapfrog with tuned $\epsilon$ is hard to beat in practice.

---

## Comparison with Other Integrators

| Integrator | Order | Symplectic | Reversible | Gradients/Step |
|------------|-------|------------|------------|----------------|
| Euler | 1 | ✗ | ✗ | 1 |
| Symplectic Euler | 1 | ✓ | ✗ | 1 |
| Leapfrog (Störmer-Verlet) | 2 | ✓ | ✓ | 1 |
| RK4 | 4 | ✗ | ✗ | 4 |
| Yoshida (4th order) | 4 | ✓ | ✓ | 3 |

The leapfrog integrator is the sweet spot: symplectic, reversible, second-order, and only one gradient per step.

---

## Summary

| Property | Leapfrog Integrator |
|----------|-------------------|
| **Order** | 2nd order (global), 3rd order (local) |
| **Symplectic** | Yes — preserves phase space volume |
| **Reversible** | Yes — with momentum flip |
| **Energy error** | $O(\epsilon^2)$, bounded, non-drifting |
| **Gradients per step** | 1 (after combining half-steps) |
| **Stability limit** | $\epsilon < 2/\sqrt{\lambda_{\max}}$ |

The leapfrog integrator's combination of symplecticity, reversibility, and efficiency makes it the standard choice for HMC. Its bounded energy error enables long trajectories with high acceptance rates.

---

## Exercises

1. **Implement and verify**. Implement leapfrog for the 1D harmonic oscillator $U(x) = \frac{1}{2}x^2$. Verify that trajectories are ellipses and energy is approximately conserved.

2. **Stability analysis**. For the harmonic oscillator, analytically compute the eigenvalues of the one-step leapfrog map. Show that $|\lambda| = 1$ (stability) requires $\epsilon < 2$.

3. **Order verification**. Numerically verify that leapfrog is second-order by computing the error for step sizes $\epsilon, \epsilon/2, \epsilon/4$ and checking that error decreases by factor 4.

4. **Energy error statistics**. Run many HMC trajectories and plot the distribution of $\Delta H$. How does the distribution depend on $\epsilon$ and $L$?

5. **Higher-order integrator**. Implement the fourth-order Yoshida integrator and compare energy error with leapfrog for the same computational cost.

---

## References

1. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
2. Hairer, E., Lubich, C., & Wanner, G. (2006). *Geometric Numerical Integration* (2nd ed.). Springer.
3. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
4. Blanes, S., Casas, F., & Sanz-Serna, J. M. (2014). "Numerical Integrators for the Hybrid Monte Carlo Method." *SIAM Journal on Scientific Computing*.
