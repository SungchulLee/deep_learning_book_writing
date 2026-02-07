# Wasserstein Loss

The Wasserstein GAN (WGAN) replaces the Jensen-Shannon divergence with the Wasserstein distance (Earth Mover's Distance), providing more stable training and meaningful loss values.

## Motivation

### Problems with JS Divergence

The original GAN minimizes Jensen-Shannon divergence:

$$\text{JSD}(p_{\text{data}} \| p_g) = \frac{1}{2} D_{KL}(p_{\text{data}} \| m) + \frac{1}{2} D_{KL}(p_g \| m)$$

where $m = \frac{1}{2}(p_{\text{data}} + p_g)$.

**Problem**: When $p_{\text{data}}$ and $p_g$ have non-overlapping supports:

$$\text{JSD}(p_{\text{data}} \| p_g) = \log 2 \quad \text{(constant)}$$

This provides **zero gradient** to the generator—it can't learn!

### Wasserstein Distance

The Wasserstein-1 distance (Earth Mover's Distance):

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

where $\Pi(p_{\text{data}}, p_g)$ is the set of all joint distributions with marginals $p_{\text{data}}$ and $p_g$.

**Advantage**: Wasserstein distance provides meaningful gradients even when distributions don't overlap.

## Mathematical Formulation

### Kantorovich-Rubinstein Duality

The Wasserstein distance has a dual formulation:

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \left[ \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)] \right]$$

where the supremum is over all 1-Lipschitz functions $f$.

### WGAN Objective

The discriminator (called **critic** in WGAN) approximates this:

$$\max_D \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

subject to $D$ being 1-Lipschitz.

The generator minimizes:

$$\min_G -\mathbb{E}_{z \sim p_z}[D(G(z))]$$

## Key Differences from Standard GAN

| Aspect | Standard GAN | WGAN |
|--------|--------------|------|
| D output | Probability [0,1] | Unbounded score |
| D activation | Sigmoid | None (linear) |
| D name | Discriminator | Critic |
| Objective | max E[log D(x)] + E[log(1-D(G(z)))] | max E[D(x)] - E[D(G(z))] |
| Divergence | Jensen-Shannon | Wasserstein |
| D constraint | None | 1-Lipschitz |

## Enforcing Lipschitz Constraint

### Method 1: Weight Clipping (Original WGAN)

```python
def clip_weights(model, clip_value=0.01):
    """Clip weights to enforce Lipschitz constraint (crude method)."""
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
```

**Issues with weight clipping**:
- Reduces model capacity
- Can cause vanishing/exploding gradients
- Biases network toward simple functions

## Enforcing Lipschitz via Weight Clipping

The original WGAN used weight clipping, but this has significant drawbacks (capacity reduction, gradient issues). See the [WGAN-GP](wgan_gp.md) section for the preferred gradient penalty approach.

## Theoretical Properties

### Wasserstein Distance Properties

1. **Metric**: $W$ is a true distance metric
2. **Continuous**: Differentiable almost everywhere
3. **Weak convergence**: $W(p_n, p) \to 0$ iff $p_n \to p$ weakly

### Gradient Behavior

Unlike JSD, Wasserstein distance provides useful gradients:

$$\nabla_\theta W(p_{\text{data}}, p_{g_\theta}) = -\mathbb{E}_{z \sim p_z}[\nabla_\theta D^*(G_\theta(z))]$$

where $D^*$ is the optimal critic.

### Meaningful Loss Value

The critic loss approximates the Wasserstein distance, so:
- **Loss decreases** → distributions getting closer
- Loss correlates with sample quality (unlike standard GAN)



## Advantages

### 1. Stable Training

- No mode collapse (empirically)
- Less sensitive to architecture choices
- No need for careful balancing of G and D

### 2. Meaningful Loss

```python
def plot_wgan_loss(wasserstein_distances):
    """WGAN loss correlates with sample quality."""
    plt.figure(figsize=(10, 5))
    plt.plot(wasserstein_distances)
    plt.xlabel('Iteration')
    plt.ylabel('Wasserstein Distance Estimate')
    plt.title('WGAN Training - Loss Correlates with Quality')
    plt.grid(True, alpha=0.3)
```

### 3. No Vanishing Gradients

Even when distributions don't overlap, critic provides gradients.

## Disadvantages

### 1. Slower Training

Requires multiple critic updates per generator update (typically 5).

### 2. Gradient Penalty Overhead

Computing GP requires second-order gradients—computationally expensive.

### 3. Hyperparameter Sensitivity

$\lambda_{GP}$ needs tuning (typically 10 works).



## Summary

| Aspect | Value |
|--------|-------|
| **Divergence** | Wasserstein-1 (Earth Mover's) |
| **Critic Output** | Unbounded score |
| **Lipschitz Constraint** | GP or Spectral Norm |
| **n_critic** | 5 (typical) |
| **λ_GP** | 10 (typical) |
| **Optimizer** | Adam with β₁=0 |
| **Main Advantage** | Stable training, meaningful loss |

WGAN and WGAN-GP represent a significant improvement in GAN stability and have become foundational techniques in modern GAN architectures.
