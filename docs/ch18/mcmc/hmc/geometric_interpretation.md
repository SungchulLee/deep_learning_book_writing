# Geometric Interpretation of HMC

Hamiltonian Monte Carlo admits rich geometric interpretations that illuminate why the algorithm works and suggest directions for improvement. This section explores HMC through the lenses of differential geometry, information geometry, and physical intuition.

---

## The Geometry of Sampling

### Why Geometry Matters

Sampling from a distribution $\pi(\mathbf{x})$ is fundamentally a geometric problem:

- The **typical set** (where most probability mass lies) is a geometric object
- **Exploration** requires navigating this geometry efficiently
- **Curvature** of the log-density affects sampler behavior
- **Distance** in parameter space should reflect statistical distance

HMC's power comes from respecting and exploiting this geometry.

### Three Geometric Perspectives

| Perspective | Key Object | Insight |
|-------------|-----------|---------|
| **Phase space** | Symplectic manifold | Volume preservation, energy conservation |
| **Information geometry** | Fisher-Rao metric | Natural distance on probability distributions |
| **Physical** | Energy landscape | Intuition for dynamics and exploration |

---

## Phase Space Geometry

### Symplectic Manifolds

Phase space $(\mathbf{x}, \mathbf{v}) \in \mathbb{R}^{2d}$ carries a **symplectic structure**—a closed, non-degenerate 2-form:

$$
\omega = \sum_{i=1}^{d} dv_i \wedge dx_i
$$

This structure is preserved by Hamiltonian dynamics:

$$
\phi_t^* \omega = \omega
$$

where $\phi_t$ is the time-$t$ flow.

### What Symplectic Structure Encodes

The symplectic form encodes:

1. **Poisson brackets**: $\{f, g\} = \omega(X_f, X_g)$
2. **Hamilton's equations**: $\dot{\mathbf{z}} = \mathbf{J} \nabla H$ where $\mathbf{J} = \omega^{-1}$
3. **Conservation laws**: Functions with $\{f, H\} = 0$ are conserved
4. **Volume element**: $\omega^d$ gives the Liouville measure

### Geometric Meaning of Conservation

**Energy conservation**: Trajectories lie on level sets $H = E$.

**Volume preservation**: The flow is an isometry of the Liouville measure.

**Consequence**: These properties ensure that if we start with samples from $\pi(\mathbf{x}, \mathbf{v}) \propto e^{-H}$, we remain on the correct distribution.

---

## Energy Landscape Visualization

### The Potential Energy Surface

The potential energy $U(\mathbf{x}) = -\log \tilde{\pi}(\mathbf{x})$ defines a surface over parameter space:

- **Valleys** (low $U$): High probability regions
- **Ridges** (high $U$): Low probability regions
- **Local minima**: Modes of the distribution
- **Saddle points**: Transition regions between modes

### Contour Plots

For a 2D distribution, contours of constant $U$ (equivalently, constant probability density) reveal the geometry:

```
           U increasing
               ↑
        ┌──────────────┐
        │    ╭───╮     │
        │  ╭─┤   ├─╮   │  Nested contours
        │ ╭┤ │ • │ ├╮  │  • = mode (minimum U)
        │  ╰─┤   ├─╯   │
        │    ╰───╯     │
        └──────────────┘
```

HMC trajectories approximately follow these contours (at constant energy), with momentum determining the "speed" along the contour.

### The Typical Set

In high dimensions, the geometry is counterintuitive:

**Mode**: The point of maximum density (minimum $U$).

**Typical set**: A thin shell at intermediate $U$ where volume × density is maximized.

For a $d$-dimensional Gaussian:
- Mode is at the mean
- Typical set is a shell at radius $\approx \sqrt{d}$ from the mean
- Volume of the shell grows exponentially with $d$

**Implication**: Good samplers must explore the typical set, not just find the mode.

---

## Information Geometry

### The Fisher Information Metric

On the space of probability distributions, the **Fisher information** defines a natural metric:

$$
g_{ij}(\boldsymbol{\theta}) = \mathbb{E}_{\pi_\theta}\left[\frac{\partial \log \pi_\theta}{\partial \theta_i} \frac{\partial \log \pi_\theta}{\partial \theta_j}\right] = -\mathbb{E}_{\pi_\theta}\left[\frac{\partial^2 \log \pi_\theta}{\partial \theta_i \partial \theta_j}\right]
$$

This is the **expected Hessian** of the negative log-likelihood.

### Fisher Metric and HMC

For sampling from $\pi(\mathbf{x})$, the "parameters" are the values $\mathbf{x}$ themselves. The Fisher metric becomes:

$$
\mathbf{G}(\mathbf{x}) = -\nabla^2 \log \pi(\mathbf{x}) = \nabla^2 U(\mathbf{x})
$$

the Hessian of the potential energy.

**Connection to mass matrix**: The optimal mass matrix satisfies:

$$
\mathbf{M}^{-1} \approx \mathbb{E}[\mathbf{G}(\mathbf{x})] = \mathbb{E}[\nabla^2 U(\mathbf{x})]
$$

For a Gaussian target, this gives $\mathbf{M}^{-1} = \boldsymbol{\Sigma}^{-1}$, i.e., $\mathbf{M} = \boldsymbol{\Sigma}$.

### Geodesics and Sampling

In Riemannian geometry, **geodesics** are curves of shortest length. In information geometry:

- Geodesics represent the most efficient paths between distributions
- HMC trajectories (with appropriate mass matrix) approximate geodesics
- This explains why HMC explores efficiently

### Riemannian HMC

**Riemannian HMC** uses a position-dependent metric $\mathbf{G}(\mathbf{x})$:

$$
H(\mathbf{x}, \mathbf{v}) = U(\mathbf{x}) + \frac{1}{2}\mathbf{v}^T \mathbf{G}(\mathbf{x})^{-1} \mathbf{v} + \frac{1}{2}\log|\mathbf{G}(\mathbf{x})|
$$

This adapts to local curvature, potentially improving sampling in regions of varying geometry.

---

## Trajectories and Orbits

### Hamiltonian Flow as Rotation

For a quadratic potential $U(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x}$ with $\mathbf{M} = \mathbf{I}$, Hamilton's equations give:

$$
\frac{d}{dt}\begin{pmatrix} \mathbf{x} \\ \mathbf{v} \end{pmatrix} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{A} & \mathbf{0} \end{pmatrix} \begin{pmatrix} \mathbf{x} \\ \mathbf{v} \end{pmatrix}
$$

The solution involves **rotation** in phase space:

$$
\begin{pmatrix} \mathbf{x}(t) \\ \mathbf{v}(t) \end{pmatrix} = \exp\left(t\begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{A} & \mathbf{0} \end{pmatrix}\right) \begin{pmatrix} \mathbf{x}(0) \\ \mathbf{v}(0) \end{pmatrix}
$$

### Orbit Geometry

**1D harmonic oscillator** ($U(x) = \frac{1}{2}\omega^2 x^2$):

Trajectories are ellipses in $(x, v)$ space:

$$
\frac{\omega^2 x^2}{2E} + \frac{v^2}{2E} = 1
$$

**Period**: $T = 2\pi/\omega$. After time $T$, the trajectory returns to its starting point.

**General quadratic**: Trajectories are ellipses in suitably rotated coordinates, with periods determined by eigenvalues of $\mathbf{A}$.

### Non-Quadratic Potentials

For general $U(\mathbf{x})$, trajectories can be:

- **Quasi-periodic**: Fill a torus densely but never exactly repeat
- **Chaotic**: Sensitive dependence on initial conditions (rare for typical posteriors)
- **Complicated**: May have multiple time scales

The U-turn criterion in NUTS detects when the trajectory has explored "enough" without requiring periodicity.

---

## The Shadow Hamiltonian

### Backward Error Analysis

The leapfrog integrator doesn't exactly solve Hamilton's equations for $H$. Instead, it **exactly** solves a **modified Hamiltonian**:

$$
\tilde{H}(\mathbf{x}, \mathbf{v}) = H(\mathbf{x}, \mathbf{v}) + \epsilon^2 H_2(\mathbf{x}, \mathbf{v}) + \epsilon^4 H_4(\mathbf{x}, \mathbf{v}) + \cdots
$$

This $\tilde{H}$ is the **shadow Hamiltonian**.

### Geometric Interpretation

- The numerical trajectory lies exactly on level sets of $\tilde{H}$
- Energy error $|H - \tilde{H}| = O(\epsilon^2)$ is bounded
- The trajectory explores $\tilde{H} = \text{const}$ surfaces, which are $O(\epsilon^2)$-close to $H = \text{const}$ surfaces

This explains why leapfrog has bounded energy error without drift.

### Implications for Sampling

Since $\tilde{H} \approx H$, the leapfrog trajectory explores approximately the correct energy surface. The MH correction accounts for the $O(\epsilon^2)$ discrepancy.

---

## Geometric View of the Mass Matrix

### Metric Interpretation

The mass matrix $\mathbf{M}$ defines a metric on momentum space:

$$
\|\mathbf{v}\|_{\mathbf{M}}^2 = \mathbf{v}^T \mathbf{M} \mathbf{v}
$$

The inverse $\mathbf{M}^{-1}$ defines a metric on velocity space (tangent to position space):

$$
\|\dot{\mathbf{x}}\|_{\mathbf{M}^{-1}}^2 = \dot{\mathbf{x}}^T \mathbf{M}^{-1} \dot{\mathbf{x}}
$$

### Whitening Transformation

With $\mathbf{M} = \boldsymbol{\Sigma}^{-1}$ (optimal for Gaussian target), define:

$$
\tilde{\mathbf{x}} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu}), \quad \tilde{\mathbf{v}} = \boldsymbol{\Sigma}^{1/2}\mathbf{v}
$$

In these coordinates:
- Target becomes $\mathcal{N}(\mathbf{0}, \mathbf{I})$
- Kinetic energy becomes $\frac{1}{2}|\tilde{\mathbf{v}}|^2$
- Dynamics become isotropic

The mass matrix effectively **whitens** the target distribution.

### Condition Number

The **condition number** $\kappa = \lambda_{\max}(\mathbf{M}^{-1}\mathbf{A})/\lambda_{\min}(\mathbf{M}^{-1}\mathbf{A})$ measures how "round" the effective distribution is.

- $\kappa = 1$: Perfectly conditioned (spherical)
- $\kappa \gg 1$: Poorly conditioned (elongated)

Optimal mass matrix achieves $\kappa = 1$.

---

## Curvature and Sampling Difficulty

### Gaussian Curvature

For a 2D distribution, the **Gaussian curvature** of the log-density surface is:

$$
K = \frac{\det(\mathbf{H})}{\left(1 + |\nabla \log \pi|^2\right)^2}
$$

where $\mathbf{H} = \nabla^2 \log \pi$ is the Hessian.

### How Curvature Affects HMC

| Curvature | Geometry | HMC Behavior |
|-----------|----------|--------------|
| Uniform positive | Bowl-shaped | Easy, stable trajectories |
| Varying positive | Varying bowl | May need adaptive methods |
| Mixed sign | Saddle regions | Trajectories may diverge |
| Near-zero | Flat regions | Slow mixing |

### Funnel Geometry

The **funnel** is a pathological geometry where curvature varies dramatically:

$$
y \sim \mathcal{N}(0, \sigma_y^2), \quad x | y \sim \mathcal{N}(0, e^{y})
$$

- At large $y$: Wide, low curvature in $x$
- At small $y$: Narrow, high curvature in $x$

**Problem**: A single step size can't work everywhere. Small $\epsilon$ for the narrow region makes exploration of the wide region very slow.

**Solutions**: Riemannian HMC, reparameterization, or careful tuning.

---

## Visualization Techniques

### Phase Portraits (2D)

For $d = 1$, plot trajectories in $(x, v)$ space:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_phase_portrait(U, grad_U, x_range, v_range, n_traj=10):
    # Energy contours
    x = np.linspace(*x_range, 100)
    v = np.linspace(*v_range, 100)
    X, V = np.meshgrid(x, v)
    H = U(X) + 0.5 * V**2
    
    plt.contour(X, V, H, levels=20, alpha=0.5)
    
    # Sample trajectories
    for _ in range(n_traj):
        x0 = np.random.uniform(*x_range)
        v0 = np.random.uniform(*v_range)
        
        # Simulate trajectory
        xs, vs = [x0], [v0]
        x_curr, v_curr = x0, v0
        
        for _ in range(200):
            # Leapfrog step
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            x_curr = x_curr + 0.1 * v_curr
            v_curr = v_curr - 0.05 * grad_U(x_curr)
            xs.append(x_curr)
            vs.append(v_curr)
        
        plt.plot(xs, vs, 'b-', alpha=0.3, linewidth=0.5)
    
    plt.xlabel('x')
    plt.ylabel('v')
    plt.title('Phase Portrait')
```

### Energy Along Trajectory

Plot $H(t)$ to visualize energy conservation:

```python
def plot_energy_trajectory(x0, v0, U, grad_U, n_steps=100, epsilon=0.1):
    x, v = x0.copy(), v0.copy()
    energies = [U(x) + 0.5 * np.dot(v, v)]
    
    for _ in range(n_steps):
        v = v - (epsilon/2) * grad_U(x)
        x = x + epsilon * v
        v = v - (epsilon/2) * grad_U(x)
        energies.append(U(x) + 0.5 * np.dot(v, v))
    
    plt.plot(energies)
    plt.xlabel('Leapfrog Step')
    plt.ylabel('Hamiltonian H')
    plt.title('Energy Conservation')
    plt.axhline(energies[0], color='r', linestyle='--', label='Initial')
```

### Trajectory in Parameter Space

For $d = 2$, plot the trajectory projected onto $(x_1, x_2)$:

```python
def plot_trajectory_2d(samples, target_log_prob=None):
    plt.plot(samples[:, 0], samples[:, 1], 'b-', alpha=0.5, linewidth=0.5)
    plt.plot(samples[0, 0], samples[0, 1], 'go', markersize=10, label='Start')
    plt.plot(samples[-1, 0], samples[-1, 1], 'ro', markersize=10, label='End')
    
    if target_log_prob is not None:
        x = np.linspace(samples[:, 0].min() - 1, samples[:, 0].max() + 1, 100)
        y = np.linspace(samples[:, 1].min() - 1, samples[:, 1].max() + 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[target_log_prob(np.array([xi, yi])) 
                       for xi, yi in zip(x_row, y_row)]
                      for x_row, y_row in zip(X, Y)])
        plt.contour(X, Y, np.exp(Z), levels=10, alpha=0.5, colors='gray')
    
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.legend()
```

---

## Geometric Perspective on NUTS

### U-Turn as Geometric Criterion

The U-turn condition:

$$
(\mathbf{x}^+ - \mathbf{x}^-) \cdot \mathbf{v}^+ < 0
$$

has a geometric interpretation: the trajectory has started to **curve back** toward its origin.

In the $(x, v)$ plane for 1D:
- Trajectory traces an arc
- U-turn occurs when the arc starts bending backward
- Corresponds to approximately half an "orbit"

### Tree Building as Exploration

The doubling tree in NUTS systematically explores the trajectory:

```
        Depth 0         Depth 1              Depth 2
           •       →    •───•        →    •───•───•───•
                      backward           forward extension
```

Each doubling extends the explored region of phase space, checking for U-turns at multiple scales.

---

## Connections to Other Methods

### Langevin Dynamics

**Overdamped Langevin** (first-order):
$$
d\mathbf{x} = \nabla \log \pi(\mathbf{x}) \, dt + \sqrt{2} \, d\mathbf{W}
$$

**Hamiltonian dynamics** (second-order):
$$
d\mathbf{x} = \mathbf{v} \, dt, \quad d\mathbf{v} = \nabla \log \pi(\mathbf{x}) \, dt
$$

The key difference: HMC has **momentum** that provides coherent motion, while Langevin has **noise** that causes diffusion.

Geometrically: HMC follows deterministic curves in phase space; Langevin follows stochastic paths in position space.

### Natural Gradient Descent

**Natural gradient descent** uses the Fisher metric:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta \mathbf{G}(\boldsymbol{\theta}_t)^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_t)
$$

HMC with optimal mass matrix implements a stochastic version of this, but with energy conservation that prevents convergence to a point.

---

## Summary

| Geometric Concept | Role in HMC |
|------------------|-------------|
| **Symplectic structure** | Ensures volume preservation |
| **Energy surfaces** | Constrain trajectories |
| **Fisher metric** | Motivates optimal mass matrix |
| **Curvature** | Determines sampling difficulty |
| **Shadow Hamiltonian** | Explains bounded energy error |
| **Geodesics** | HMC approximates efficient paths |

The geometric perspective reveals HMC as a natural algorithm: it respects the intrinsic geometry of probability distributions, uses physically motivated dynamics, and achieves efficiency by exploiting conservation laws. Understanding this geometry guides both theoretical analysis and practical improvements.

---

## Exercises

1. **Phase portrait**. Plot the phase portrait for a double-well potential $U(x) = (x^2 - 1)^2$. Identify the separatrix (boundary between oscillation around one well vs. traversing both wells).

2. **Whitening visualization**. For a 2D correlated Gaussian, visualize trajectories before and after the whitening transformation induced by the optimal mass matrix.

3. **Curvature computation**. Compute the Hessian and Gaussian curvature for a 2D mixture of Gaussians. How does curvature vary across the parameter space?

4. **Shadow Hamiltonian**. Numerically estimate the shadow Hamiltonian by fitting $\tilde{H} = H + \epsilon^2 H_2$ to leapfrog trajectories. Verify that $\tilde{H}$ is more nearly conserved than $H$.

5. **Funnel geometry**. Visualize the funnel distribution and run HMC with various step sizes. Identify where divergences occur and relate this to local curvature.

---

## References

1. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
2. Girolami, M., & Calderhead, B. (2011). "Riemann Manifold Langevin and Hamiltonian Monte Carlo Methods." *JRSS-B*.
3. Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
4. Leimkuhler, B., & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
5. Neal, R. M. (2011). "MCMC Using Hamiltonian Dynamics." In *Handbook of Markov Chain Monte Carlo*.
