# Nash Equilibrium

GAN training can be understood through the lens of game theory. The concept of **Nash equilibrium** provides the theoretical foundation for understanding when and how GANs converge.

## Game-Theoretic Framework

### Two-Player Game Formulation

A GAN defines a two-player game where:

- **Player 1 (Generator)**: Chooses strategy $G$ (parameters $\theta_G$)
- **Player 2 (Discriminator)**: Chooses strategy $D$ (parameters $\theta_D$)
- **Payoff function**: $V(D, G)$

This is a **zero-sum game**: D maximizes $V$, G minimizes $V$.

### Nash Equilibrium Definition

A **Nash equilibrium** is a strategy profile $(G^*, D^*)$ where neither player can improve their payoff by unilaterally changing their strategy:

$$V(D, G^*) \leq V(D^*, G^*) \leq V(D^*, G) \quad \forall D, G$$

This means:
- D cannot improve by changing strategy (given G*)
- G cannot improve by changing strategy (given D*)

## Existence of Nash Equilibrium

### Theorem: GAN Equilibrium

For the GAN value function:

$$V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

A Nash equilibrium exists with:

$$p_g^* = p_{\text{data}}$$
$$D^*(x) = \frac{1}{2} \quad \forall x$$

### Proof Sketch

**Step 1**: For any generator G, the optimal discriminator is:

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

**Step 2**: Substituting $D^*_G$ into $V$:

$$C(G) = V(D^*_G, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

**Step 3**: JSD is minimized (equals 0) when $p_g = p_{\text{data}}$.

**Step 4**: At equilibrium:
- $p_g^* = p_{\text{data}}$ minimizes $C(G)$
- $D^*(x) = \frac{p_{\text{data}}(x)}{2 \cdot p_{\text{data}}(x)} = \frac{1}{2}$

## Properties of the GAN Equilibrium

### At Equilibrium

| Quantity | Value |
|----------|-------|
| Generator distribution | $p_g = p_{\text{data}}$ |
| Discriminator output | $D(x) = 0.5$ for all $x$ |
| Value function | $V(D^*, G^*) = -\log 4 \approx -1.386$ |
| Jensen-Shannon divergence | $\text{JSD}(p_{\text{data}} \| p_g) = 0$ |

### Interpretation

At equilibrium:
- The generator produces perfect samples
- The discriminator cannot distinguish real from fake
- The discriminator's best strategy is random guessing (50/50)

## Convergence to Equilibrium

### Ideal Convergence

Under ideal conditions, alternating gradient descent converges to Nash equilibrium:

```python
import torch

def ideal_gan_training(G, D, data_loader, n_iterations):
    """
    Idealized GAN training assuming infinite capacity and 
    optimal updates at each step.
    """
    for i in range(n_iterations):
        # Step 1: Update D to optimality (in practice, approximate)
        for _ in range(k_d):  # k_d discriminator steps
            d_loss = compute_d_loss(D, G, data_loader)
            update_discriminator(D, d_loss)
        
        # Step 2: Update G
        g_loss = compute_g_loss(G, D)
        update_generator(G, g_loss)
    
    # At convergence: p_g ≈ p_data, D(x) ≈ 0.5
    return G, D
```

### Theoretical Guarantees

**Proposition** (Goodfellow et al., 2014): If G and D have enough capacity and at each step the discriminator is allowed to reach its optimum given G, and p_g is updated to improve the criterion:

$$\mathbb{E}_{x \sim p_{\text{data}}}[\log D^*_G(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*_G(x))]$$

then $p_g$ converges to $p_{\text{data}}$.

## Why Convergence is Difficult

### Problem 1: Non-Convex Optimization

The GAN objective is not convex-concave:

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_loss_landscape():
    """The GAN loss landscape is complex, not convex-concave."""
    # In parameter space, the landscape has many saddle points
    # and non-convex regions
    
    theta_g = np.linspace(-2, 2, 100)
    theta_d = np.linspace(-2, 2, 100)
    TG, TD = np.meshgrid(theta_g, theta_d)
    
    # Simplified loss landscape (illustrative)
    # Real GAN landscapes are high-dimensional and more complex
    V = np.sin(TG) * np.cos(TD) + 0.1 * (TG**2 - TD**2)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(TG, TD, V, levels=20, cmap='RdYlBu')
    plt.colorbar(label='V(D, G)')
    plt.xlabel('θ_G')
    plt.ylabel('θ_D')
    plt.title('GAN Loss Landscape (Simplified)')
```

### Problem 2: Oscillation

Instead of converging, G and D may oscillate:

```
Iteration 1: G focuses on mode A, D learns to reject A
Iteration 2: G switches to mode B, D learns to reject B
Iteration 3: G switches back to A (D "forgot"), ...
```

### Problem 3: Finite Capacity

Real networks have finite capacity:
- D cannot represent the true optimal discriminator
- G cannot represent all possible distributions

## Stability Analysis

### Local Stability

Near equilibrium, we can analyze stability using the Jacobian of the gradient dynamics:

$$\frac{d\theta_G}{dt} = -\nabla_{\theta_G} V$$
$$\frac{d\theta_D}{dt} = +\nabla_{\theta_D} V$$

The Jacobian:

$$J = \begin{pmatrix} -\nabla^2_{\theta_G} V & -\nabla_{\theta_G \theta_D} V \\ \nabla_{\theta_D \theta_G} V & \nabla^2_{\theta_D} V \end{pmatrix}$$

### Eigenvalue Analysis

For stability, eigenvalues of J should have negative real parts. However, in GANs:

- The mixed partial derivatives create purely imaginary eigenvalues
- This leads to **rotation** rather than convergence
- Explains oscillatory behavior

```python
def analyze_local_dynamics(V_gg, V_gd, V_dd):
    """
    Analyze local dynamics near equilibrium.
    
    Args:
        V_gg: ∂²V/∂θ_G²
        V_gd: ∂²V/∂θ_G∂θ_D
        V_dd: ∂²V/∂θ_D²
    """
    import numpy as np
    
    # Jacobian of gradient dynamics
    J = np.array([
        [-V_gg, -V_gd],
        [V_gd.T, V_dd]  # Note: V_dg = V_gd.T
    ])
    
    eigenvalues = np.linalg.eigvals(J)
    
    print("Eigenvalues:", eigenvalues)
    print("Real parts:", eigenvalues.real)
    print("Imaginary parts:", eigenvalues.imag)
    
    # Check stability
    if all(eigenvalues.real < 0):
        print("Stable equilibrium")
    elif any(eigenvalues.real > 0):
        print("Unstable equilibrium")
    else:
        print("Marginal stability (oscillation possible)")
```

## Achieving Convergence in Practice

### Strategy 1: Multiple D Steps

Train D more than G to keep D close to optimal:

```python
def train_with_multiple_d_steps(G, D, data, n_d_steps=5):
    """Train with multiple discriminator steps per generator step."""
    
    for _ in range(n_d_steps):
        # Update D
        d_loss = discriminator_loss(D, G, data)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
    
    # Update G once
    g_loss = generator_loss(G, D)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
```

### Strategy 2: Two-Timescale Update Rule (TTUR)

Use different learning rates:

```python
# Discriminator learns faster
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0004)

# Generator learns slower
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
```

### Strategy 3: Spectral Normalization

Constrain discriminator Lipschitz constant:

```python
from torch.nn.utils import spectral_norm

class StableDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Linear(784, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid()
        )
```

### Strategy 4: Gradient Penalty

Add penalty for discriminator gradient magnitude:

```python
def gradient_penalty(D, real_data, fake_data, lambda_gp=10):
    """WGAN-GP gradient penalty for 1-Lipschitz constraint."""
    batch_size = real_data.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Discriminator output
    d_interpolated = D(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True
    )[0]
    
    # Penalty
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

## Monitoring Equilibrium

### Indicators of Convergence

```python
def check_equilibrium_indicators(D, G, real_data, fake_data):
    """Check if training is approaching equilibrium."""
    
    with torch.no_grad():
        d_real = D(real_data).mean().item()
        d_fake = D(fake_data).mean().item()
    
    indicators = {
        'D(real)': d_real,
        'D(fake)': d_fake,
        'D_accuracy': (d_real + (1 - d_fake)) / 2,
    }
    
    # At equilibrium: D(x) ≈ 0.5 for all x
    equilibrium_score = 1 - abs(d_real - 0.5) - abs(d_fake - 0.5)
    indicators['equilibrium_score'] = equilibrium_score
    
    print(f"D(real): {d_real:.4f}")
    print(f"D(fake): {d_fake:.4f}")
    print(f"Equilibrium score: {equilibrium_score:.4f}")
    
    if d_real > 0.9 and d_fake < 0.1:
        print("Warning: D is too strong, G may have vanishing gradients")
    elif abs(d_real - 0.5) < 0.1 and abs(d_fake - 0.5) < 0.1:
        print("Near equilibrium!")
    
    return indicators
```

### Visual Monitoring

```python
def plot_convergence(d_real_history, d_fake_history):
    """Visualize convergence toward equilibrium."""
    plt.figure(figsize=(10, 5))
    
    iterations = range(len(d_real_history))
    
    plt.plot(iterations, d_real_history, label='D(real)', alpha=0.7)
    plt.plot(iterations, d_fake_history, label='D(fake)', alpha=0.7)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Equilibrium')
    
    plt.xlabel('Iteration')
    plt.ylabel('Discriminator Output')
    plt.title('Convergence Toward Nash Equilibrium')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
```

## Beyond Standard Nash Equilibrium

### Mixed Nash Equilibrium

In some cases, the solution is a **mixed equilibrium** where players randomize:

- G samples from a distribution over generators
- D samples from a distribution over discriminators

### Local Nash Equilibrium

Due to non-convexity, GANs typically find **local** Nash equilibria:

- Stable under small perturbations
- May not be globally optimal
- Multiple local equilibria possible

## Summary

| Concept | Description |
|---------|-------------|
| **Nash Equilibrium** | Neither player can improve unilaterally |
| **GAN Equilibrium** | $p_g = p_{\text{data}}$, $D^* = 0.5$ |
| **Convergence** | Theoretically guaranteed under ideal conditions |
| **Practice Issues** | Non-convexity, oscillation, finite capacity |
| **Stabilization** | TTUR, spectral norm, gradient penalty |
| **Monitoring** | Track D(real), D(fake) toward 0.5 |

Understanding Nash equilibrium provides theoretical grounding for GAN training and motivates many practical improvements. While perfect convergence is difficult, these insights guide the development of more stable training algorithms.
