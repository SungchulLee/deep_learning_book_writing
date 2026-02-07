# Temperature as a Unifying Concept

Temperature appears throughout machine learning under different names and guises—from MCMC sampling to neural network softmax to diffusion models. This section reveals the deep connections between these seemingly disparate techniques, showing how temperature provides a unified framework for understanding exploration-exploitation trade-offs across the field.

---

## The Universal Role of Temperature

### The Boltzmann Principle

The foundation is the **Boltzmann distribution** from statistical mechanics:

$$
p_T(x) = \frac{1}{Z_T} \exp\left(-\frac{E(x)}{T}\right)
$$

This single equation, with temperature $T$ as the control parameter, underlies MCMC sampling, simulated annealing, softmax in neural networks, Langevin dynamics, diffusion models, energy-based models, and reinforcement learning policies.

### The Energy-Entropy Trade-off

Temperature controls the fundamental trade-off between **energy** (preference for low-cost states) and **entropy** (preference for diversity):

$$
F = \langle E \rangle - T \cdot S
$$

| Temperature | Behavior |
|-------------|----------|
| $T \to 0$ | Pure energy minimization (exploitation) |
| $T = 1$ | Balanced trade-off |
| $T \to \infty$ | Pure entropy maximization (exploration) |

This trade-off appears everywhere decisions must balance quality versus diversity.

---

## Temperature in MCMC

### Fixed-Temperature Sampling

Standard MCMC samples from $\pi(x) \propto \exp(-E(x))$, implicitly at $T = 1$:

$$
\pi_T(x) = \frac{1}{Z_T} \exp\left(-\frac{E(x)}{T}\right) = \frac{1}{Z_T} \pi(x)^{1/T}
$$

**Hot chains** ($T > 1$): Flattened distribution, easier exploration, faster mixing between modes, but poor approximation to target.

**Cold chains** ($T < 1$): Sharpened distribution, concentrated on modes, slow mixing, but better local approximation.

### Parallel Tempering

Parallel tempering exploits multiple temperatures simultaneously:

```python
def parallel_tempering_step(chains, temperatures, E):
    """One step of parallel tempering."""
    n_chains = len(chains)
    
    # Local moves at each temperature
    for i in range(n_chains):
        chains[i] = metropolis_step(chains[i], E, temperatures[i])
    
    # Attempt swaps between adjacent temperatures
    for i in range(n_chains - 1):
        delta = (1/temperatures[i] - 1/temperatures[i+1]) * \
                (E(chains[i+1]) - E(chains[i]))
        
        if np.random.rand() < np.exp(delta):
            chains[i], chains[i+1] = chains[i+1], chains[i]
    
    return chains
```

Hot chains explore globally; cold chains refine locally; swaps transfer information between temperature levels.

### Annealed Importance Sampling

Use a temperature path for importance sampling:

$$
w = \prod_{t=1}^{T} \frac{p_{t}(x_{t-1})}{p_{t-1}(x_{t-1})}
$$

where $p_t$ interpolates from an easy distribution to the target via temperature. This reduces variance compared to direct importance sampling from $\pi$.

---

## Temperature in Langevin Dynamics

### The Langevin SDE

The temperature-scaled Langevin equation:

$$
d\mathbf{x}_t = -\nabla E(\mathbf{x}_t) \, dt + \sqrt{2T} \, d\mathbf{W}_t
$$

**Critical insight**: Temperature scales the **noise**, not the drift.

| Component | Role |
|-----------|------|
| $-\nabla E(\mathbf{x}_t) \, dt$ | Deterministic drift toward low energy |
| $\sqrt{2T} \, d\mathbf{W}_t$ | Stochastic exploration |

### Temperature Limits

**$T \to 0$**: Gradient descent (pure optimization), $d\mathbf{x}_t = -\nabla E(\mathbf{x}_t) \, dt$.

**$T \to \infty$**: Brownian motion (pure diffusion), $d\mathbf{x}_t = \sqrt{2T} \, d\mathbf{W}_t$.

**$T = 1$**: Balanced dynamics sampling from $\pi(\mathbf{x}) \propto e^{-E(\mathbf{x})}$.

### Discretization

The Euler-Maruyama discretization with step size $\epsilon$:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \epsilon \nabla E(\mathbf{x}_t) + \sqrt{2\epsilon T} \, \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

This is the **Unadjusted Langevin Algorithm (ULA)** at temperature $T$, connecting the continuous SDE framework to the discrete algorithms covered in the [Langevin Dynamics](../../langevin/fundamentals.md) section.

---

## Temperature in Diffusion Models

### The Forward Process

Diffusion models add noise progressively:

$$
d\mathbf{x}_t = -\frac{1}{2}\beta(t) \mathbf{x}_t \, dt + \sqrt{\beta(t)} \, d\mathbf{W}_t
$$

The noise schedule $\beta(t)$ plays the role of (inverse) temperature: early times have low noise (low temperature) preserving data structure, while late times have high noise (high temperature) approaching pure noise.

### The Reverse Process

Generation reverses the diffusion:

$$
d\mathbf{x}_t = \left[-\frac{1}{2}\beta(t) \mathbf{x}_t - \beta(t) \nabla_\mathbf{x} \log p_t(\mathbf{x})\right] dt + \sqrt{\beta(t)} \, d\bar{\mathbf{W}}_t
$$

This is precisely **Langevin dynamics** with time-varying temperature.

### The Annealing Analogy

| Simulated Annealing | Diffusion Models |
|---------------------|------------------|
| Start: high $T$ (uniform-like) | Start: high noise (Gaussian) |
| End: low $T$ (concentrated) | End: low noise (data distribution) |
| Decrease $T$ over iterations | Decrease noise over steps |
| Goal: find optimum | Goal: generate sample |
| Energy function $E(x)$ | Score function $\nabla \log p(x)$ |

Both methods use temperature/noise schedules to interpolate between easy and hard problems.

### Temperature Scaling at Inference

At inference time, we can modify the noise scale:

$$
\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t, t) + \sigma_t^{1/T} \boldsymbol{\epsilon}
$$

$T > 1$ gives more stochastic, diverse samples; $T < 1$ gives more deterministic, higher quality samples; $T = 0$ yields deterministic (DDIM-style) generation.

---

## Temperature in Neural Networks

### Softmax Temperature

The temperature-scaled softmax:

$$
\text{softmax}_T(z_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**Low temperature** ($T \to 0$): $\text{softmax}_T(z) \to \text{one-hot}(\arg\max z)$—sharp, deterministic selection.

**High temperature** ($T \to \infty$): $\text{softmax}_T(z) \to \text{uniform}$—soft, uniform distribution.

### Applications in Deep Learning

**Knowledge distillation** (Hinton et al., 2015): Teacher produces soft targets at high $T$, student learns from softened distribution, transferring "dark knowledge" about class relationships.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """Knowledge distillation with temperature."""
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    
    # Distillation loss (scaled by T^2 as per Hinton)
    distill_loss = F.kl_div(soft_student, soft_targets, 
                            reduction='batchmean') * T * T
    
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * distill_loss + (1 - alpha) * hard_loss
```

**Temperature in attention**: The $\sqrt{d_k}$ factor in scaled dot-product attention is a temperature that prevents saturation as dimension grows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

**Calibration via temperature scaling**: Post-hoc calibration by fitting $T$ on validation data: $p(y | x) = \text{softmax}(f(x) / T)$.

---

## Temperature in Reinforcement Learning

### Entropy-Regularized RL

The soft Bellman equation:

$$
V^*(s) = \max_a \left[ Q^*(s, a) - T \log \pi(a|s) \right]
$$

Temperature controls exploration: high $T$ encourages exploring diverse actions, low $T$ exploits best-known actions.

### Soft Actor-Critic (SAC)

The maximum entropy objective:

$$
J(\pi) = \sum_t \mathbb{E}\left[ r(s_t, a_t) + T \cdot H[\pi(\cdot | s_t)] \right]
$$

Temperature $T$ (often called $\alpha$ in SAC) is the entropy coefficient, and can be automatically tuned to maintain a target entropy level.

---

## Temperature in Optimization

### Entropic Regularization

Add entropy to the objective:

$$
\min_\mathbf{x} \left[ E(\mathbf{x}) - T \cdot H[\mathbf{x}] \right]
$$

For probability vectors, this gives $\mathbf{x}^* \propto \exp(-E(\mathbf{x}) / T)$.

### Optimal Transport

Entropic optimal transport with regularization $\epsilon$:

$$
\min_{\gamma} \langle C, \gamma \rangle - \epsilon H(\gamma)
$$

The regularization $\epsilon$ is temperature—larger $\epsilon$ gives smoother transport plans, enabling the efficient Sinkhorn algorithm.

---

## The Unified View

### Common Mathematical Structure

All temperature-based methods share:

$$
p_T(x) \propto f(x)^{1/T}
$$

where $f(x)$ is a "fitness" or "unnormalized probability" function. Equivalently, with energy $E(x) = -\log f(x)$:

$$
p_T(x) \propto \exp(-E(x)/T)
$$

### Cross-Domain Correspondences

| Domain | "Temperature" | "Energy" | High-T Behavior |
|--------|--------------|----------|-----------------|
| Statistical Mechanics | $T$ | $E(x)$ | Disorder |
| MCMC | $1/\beta$ | $-\log \pi(x)$ | Flat chain |
| Simulated Annealing | $T(t) \to 0$ | Cost function | Exploration |
| Langevin | Noise scale | $-\log p(x)$ | Diffusion |
| Diffusion Models | $\sigma(t)$ | Score function | Gaussian |
| Softmax | $T$ | $-z_i$ | Uniform |
| RL | $T$ in SAC | $-Q(s,a)$ | Random policy |
| Optimal Transport | $\epsilon$ | Cost matrix | Smooth plan |

### Temperature Tuning Heuristics

| Goal | Temperature Strategy |
|------|---------------------|
| Find global optimum | Anneal: high → low |
| Sample accurately | Fixed $T = 1$ |
| Increase diversity | Raise $T$ |
| Increase quality | Lower $T$ |
| Transfer knowledge | Use elevated $T$ |
| Calibrate predictions | Fit $T$ on held-out data |

### Common Pitfalls

**Confusing temperature effects**: Temperature affects the *distribution*, not the *dynamics* directly. In Langevin, $T$ scales noise, not gradient. In softmax, $T$ scales logits, not probabilities.

**Wrong scaling**:

```python
# WRONG: Temperature on probability
p_wrong = softmax(z) ** (1/T)

# RIGHT: Temperature on logits
p_right = softmax(z / T)
```

**Ignoring normalization**: Temperature changes the partition function and may need to be accounted for in comparisons across temperatures.

---

## Quantitative Finance Perspective

Temperature-based methods appear naturally in finance:

- **Portfolio optimization**: Entropic regularization smooths the efficient frontier, and simulated annealing handles discrete constraints (cardinality, integer lots)
- **Option pricing**: Importance sampling with temperature-tilted measures reduces variance in Monte Carlo pricing of deep out-of-the-money options
- **Risk management**: Parallel tempering improves exploration of tail risk scenarios in complex portfolio models
- **Model calibration**: SA navigates the multimodal likelihood surfaces common in stochastic volatility and jump-diffusion models

---

## Historical Perspective

| Year | Development |
|------|-------------|
| 1870s | Boltzmann distribution in physics |
| 1953 | Metropolis algorithm (sampling at $T$) |
| 1983 | Simulated annealing (optimization) |
| 1986 | Boltzmann machines |
| 1990s | Parallel tempering |
| 2010s | Temperature in deep learning (distillation, SAC) |
| 2020s | Diffusion models as annealing |

---

## Summary

| Method | Temperature Role | Key Insight |
|--------|-----------------|-------------|
| **MCMC** | Flattens/sharpens target | Parallel tempering exploits multiple $T$ |
| **Simulated Annealing** | Annealing schedule | Guarantees global optimum (if slow enough) |
| **Langevin** | Noise scale | $T$ controls diffusion strength |
| **Diffusion Models** | Noise schedule | Generation is reverse annealing |
| **Softmax** | Sharpness control | Distillation, calibration, attention |
| **RL** | Exploration bonus | Entropy regularization |
| **EBMs** | Training/sampling | Score matching at multiple $T$ |

**The unifying principle**: Temperature controls the trade-off between energy minimization (quality, exploitation) and entropy maximization (diversity, exploration). This fundamental trade-off appears throughout machine learning, and temperature provides the mathematical knob to control it.
