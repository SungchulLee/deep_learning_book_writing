# Learned Optimizers: Meta-Learning for Optimization

## Introduction

While traditional optimizers (SGD, Adam, RMSprop) use fixed, hand-designed update rules, learned optimizers represent a paradigm where the optimization process itself is learned through meta-learning. By training on a distribution of optimization tasks, learned optimizers develop adaptive, problem-aware update strategies that often surpass fixed optimizers in speed and final performance.

For quantitative finance, learned optimizers offer particular advantages: market dynamics change continuously, requiring constant model retraining. A learned optimizer that adapts to specific asset classes or market regimes could substantially accelerate model fitting while improving generalization across diverse financial instruments.

## Key Concepts

- **Learned Update Rule**: Replace fixed optimizer with learned function
- **Optimization Tasks**: Training losses as meta-learning tasks
- **Meta-Optimizer**: Algorithm learning the optimizer
- **State Representation**: Features encoding optimization history
- **Generalization Across Tasks**: Optimizer works for diverse loss landscapes

## Mathematical Framework

### Fixed Optimizer Baseline

Standard optimizers apply identical update rule:

$$\theta_{t+1} = \theta_t - \alpha_t \mathbf{g}_t$$

where $\mathbf{g}_t$ is gradient and $\alpha_t$ is learning rate.

### Learned Optimizer Framework

Replace fixed rule with learned function:

$$\theta_{t+1} = \theta_t + f_{\Phi}(\mathbf{g}_t, \mathbf{m}_t, \mathbf{s}_t)$$

where:
- $f_{\Phi}$ is learned update function
- $\mathbf{m}_t$ is momentum/historical state
- $\mathbf{s}_t$ encodes problem structure

### Optimization Loss

Meta-training minimizes loss over training trajectory:

$$\mathcal{L}_{\text{meta}} = \sum_{\mathcal{L} \sim \mathcal{D}} \sum_{t=1}^{T} \mathcal{L}(\theta_t)$$

where meta-loss is sum over target losses, optimized through learned optimizer.

## Architectures for Learned Optimizers

### Recurrent Optimizer

Use LSTM to maintain hidden state representing optimization memory:

$$\mathbf{h}_t = \text{LSTM}(\mathbf{g}_t, \mathbf{h}_{t-1})$$
$$\Delta \theta_t = \text{Linear}(\mathbf{h}_t)$$

LSTM learns when to accelerate (momentum-like), when to reset (adaptive learning rate), and which directions to explore.

### Convolutional Optimizer

Process gradient maps (for image models) with convolutions:

$$\Delta \theta_t = \text{Conv}(\mathbf{g}_t, \text{conv history})$$

Captures spatial structure in gradients, learning how different layer types should update.

### Transformer-Based Optimizer

Attend to gradient history and problem structure:

$$\Delta \theta_t = \text{Attention}(\mathbf{g}_t, \text{past gradients})$$

Enables long-range dependency modeling across optimization trajectory.

## State Representation

What features should learned optimizer input?

### Gradient Information

**Raw Gradients**: $\mathbf{g}_t$ directly (high-dimensional but informative)

**Normalized Gradients**: $\mathbf{g}_t / (\|\mathbf{g}_t\| + \epsilon)$ (removes scale, focuses on direction)

**Log-Absolute Gradients**: $\log(|\mathbf{g}_t| + \epsilon)$ (handles wide range of magnitudes)

### Momentum Information

$$\mathbf{m}_t = \beta \mathbf{m}_{t-1} + (1-\beta) \mathbf{g}_t$$

Encode recent history of updates and momentum state.

### Second-Order Information

**Gradient Variance**: $\text{Var}(\mathbf{g}_{t-k:t})$ for recent window

**Loss Curvature Estimate**: From Hessian diagonal approximations

### Problem-Specific Features

!!! tip "Domain Knowledge Integration"
    Include domain-specific features for better adaptation:
    
    - Asset class (equity, bond, derivative)
    - Market liquidity regime
    - Historical volatility
    - Time since last update

## Training Procedures

### Meta-Training

Collect diverse optimization tasks (different losses, datasets, architectures):

```
for each meta-iteration:
    Sample task T ~ P(tasks)
    Initialize θ from N(0, I)
    for t = 1 to T_steps:
        g ← compute gradients
        Δθ ← optimizer(g, history)
        θ ← θ + Δθ
        Accumulate meta-loss L(θ)
    Compute meta-gradient: ∂L_meta/∂Φ
    Update optimizer: Φ ← Φ - α∇Φ L_meta
```

### Challenges in Meta-Training

**Non-Stationarity**: Optimization landscape changes during training, making learned optimizer unstable.

**Generalization Gap**: Optimizer trained on specific problem distributions may not transfer to different problem types.

**Computational Overhead**: Meta-training requires optimizing the optimizer, expensive nested loops.

## Addressing Generalization

### Task Distribution Design

Meta-train on diverse problem distributions:

- Different loss functions (classification, regression, ranking)
- Different architectures (CNNs, RNNs, Transformers)
- Different dataset sizes
- Different condition numbers (easy vs. hard optimization landscapes)

### Regularization of Learned Optimizers

Add regularization to prevent overfitting to specific problems:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{meta}} + \lambda \Omega(\Phi)$$

where $\Omega(\Phi)$ encourages smooth, interpretable update rules.

### Hybrid Approaches

Combine learned and fixed components:

$$\Delta \theta_t = \alpha_{\text{learned}}(\mathbf{g}_t) \cdot \mathbf{g}_t + \beta \cdot \text{momentum}$$

Balance adaptivity with stability by learning scalar multiplier while keeping base structure fixed.

## Efficiency Considerations

### Computational Cost

Learned optimizers add overhead:

- Extra neural network evaluations per optimization step
- Maintaining hidden state (memory cost)
- More iterations needed to converge (generalization-convergence trade-off)

**Net Effect**: Can be faster or slower depending on problem

### Acceleration for Repeated Optimization

Learned optimizers shine when solving similar optimization problems repeatedly:

**Financial Application**: Retrain model daily on new data

**Speedup**: 50-200% faster convergence than Adam on same task distribution

## Comparison with Fixed Optimizers

| Aspect | Adam | SGD+Momentum | Learned |
|--------|------|-------------|---------|
| **Tuning** | Low | Low | Very High |
| **Generalization** | Broad | Narrow | Specific |
| **Speed** | Fast | Medium | Fast/Slow |
| **Interpretability** | High | High | Low |
| **Computation** | Low | Low | Medium |

## Applications in Quantitative Finance

!!! warning "Financial Optimization"
    
    Learn asset-class-specific optimizers:
    - **Equity Models**: Fast convergence on noisy daily data
    - **Bond Models**: Capture term structure with fewer updates
    - **FX Models**: Adapt to regime-specific volatility
    - **Portfolio Optimization**: Rapidly rebalance with learned updates

## Research Directions

- Improving generalization across diverse problem classes
- Theoretically understanding learned optimizer convergence
- Combining meta-learned and evolutionary optimization
- Application to non-convex, non-smooth optimization

## Related Topics

- Meta-Learning Overview (Chapter 11.1)
- Task Distribution Design (Chapter 11.3)
- Optimization Theory
- Neural Architecture Search (uses learned optimization)
