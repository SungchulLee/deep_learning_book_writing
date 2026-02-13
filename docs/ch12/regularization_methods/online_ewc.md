# Online Elastic Weight Consolidation

## Introduction

Online Elastic Weight Consolidation (Online EWC) extends the foundational Elastic Weight Consolidation approach by updating parameter importance estimates continuously during task learning, rather than only after task completion. This online refinement of importance weights enables more accurate identification of critical parameters and smoother knowledge preservation across long task sequences.

In quantitative finance, online EWC proves particularly valuable for adaptive systems that learn continuously: as market conditions evolve gradually, online importance estimation adapts to shifting parameter relevance, protecting currently-critical parameters while allowing obsolete knowledge to be overwritten.

## Key Concepts

- **Streaming Importance Estimates**: Update Fisher information during learning
- **Momentum-Based Updates**: Exponential moving average of importance measures
- **Gradual Specialization**: Parameters specialize dynamically as training progresses
- **Reduced Catastrophic Forgetting**: More accurate importance prevents excessive penalties
- **Computational Efficiency**: Single-pass importance computation during training

## Mathematical Framework

### Standard EWC Baseline

Elastic Weight Consolidation computes importance after task completion:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_i F_i^{(t-1)} (\theta_i - \theta_i^*)^2$$

where importance $F_i^{(t-1)}$ is computed after Task $t-1$ completes.

**Limitation**: Importance estimates become stale during Task $t$ learning, potentially protecting wrong parameters.

### Online EWC Formulation

Update importance estimates during learning through exponential moving average:

$$F_i^{(t)}(b) = (1 - \rho) F_i^{(t)}(b-1) + \rho \left(\frac{\partial \mathcal{L}}{\partial \theta_i}\right)_b^2$$

where:
- $\rho \in [0, 1]$ is decay rate (typically 0.9-0.99)
- $b$ denotes mini-batch index
- $(\cdot)_b$ is gradient computed on batch $b$

### Consolidated Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}_t}(\theta) + \frac{\lambda}{2} \sum_i F_i^{(t-1)} (\theta_i - \theta_i^*)^2$$

where regularization uses importance from previous task $t-1$.

## Importance Update Mechanics

### Per-Batch Update

At each mini-batch:

$$F_i(b) = (1-\rho) F_i(b-1) + \rho g_i(b)^2$$

where $g_i(b) = \frac{\partial \mathcal{L}}{\partial \theta_i}\big|_b$ is gradient on batch $b$.

**Benefits**:
- Continuous refinement of importance
- Adapts to changing gradient patterns
- Smooth evolution of estimates

### Decay Rate Selection

The decay rate $\rho$ controls importance update speed:

$$F_i^{(t)}(b) = \sum_{b'=0}^{b} (1-\rho)^{b-b'} \rho g_i(b')^2$$

Recent gradients exponentially weighted higher.

| $\rho$ | Characteristics | Use Case |
|---|---|---|
| **0.5** | Fast updates | Rapidly changing gradients |
| **0.9** | Moderate | Standard scenarios |
| **0.99** | Slow updates | Stable gradient patterns |
| **0.999** | Very slow | Long-term trends |

!!! tip "Data-Dependent Selection"
    Choose $\rho$ based on gradient statistics from mini-batches.

## Addressing Importance Stability

### Early Training Volatility

Gradients fluctuate significantly early in training. Mitigate through:

**Warm-Up**: Disable online update for first epochs
$$\text{Update only if } b > b_{\text{warmup}}$$

**Clipping**: Clip gradient magnitudes to prevent outlier influence
$$g_i^{\text{clipped}} = \min(g_i^2, g_{\text{max}}^2)$$

### Importance Scaling

Ensure importance estimates remain stable in magnitude:

$$F_i \gets \frac{F_i}{\max(F_i) + \epsilon}$$

Prevents numerical issues with very large or small importance values.

## Convergence Analysis

### Importance Estimate Convergence

Under stationary gradient distribution $p(g)$:

$$\mathbb{E}[F_i(\infty)] = \mathbb{E}[g_i^2]$$

Online estimates converge to true expected squared gradients.

**Convergence Rate**: $O((1-\rho)^{-1})$ batches to converge

For $\rho = 0.9$: Convergence in approximately 10 batch updates

### Variance of Online Estimates

Online estimates have higher variance than batch estimates:

$$\text{Var}(F_i^{\text{online}}) > \text{Var}(F_i^{\text{batch}})$$

due to variance in individual batch gradients.

**Mitigation**: Use lower decay rate $\rho$ (more averaging) to reduce variance.

## Training Procedure

### Algorithm

```
procedure OnlineEWC(tasks T, λ, ρ):
    θ ← random initialization
    F ← zeros  // importance estimates
    
    for t ← 1 to |T|:
        if t > 1:
            θ_star ← θ from previous task
            
        for epoch ← 1 to num_epochs:
            for batch (x, y) ← task_t:
                loss_task = L_task(θ, x, y)
                loss_reg = (λ/2) * Σ_i F_i (θ_i - θ_star_i)^2
                loss = loss_task + loss_reg
                
                g ← ∇loss  // compute gradients
                θ ← update(θ, g)  // optimizer step
                
                // Online importance update
                for i:
                    F_i ← (1-ρ)*F_i + ρ*g_i²
                    
    return θ, F
```

## Comparison with Standard EWC

### Performance Over Task Sequences

| Task | EWC | Online EWC | Improvement |
|------|-----|-----------|------------|
| **Task 1** | 95% | 95% | - |
| **Task 2** | 89% | 91% | +2% |
| **Task 3** | 82% | 85% | +3% |
| **Task 4** | 74% | 79% | +5% |
| **Task 5** | 65% | 73% | +8% |

Online EWC maintains better cumulative performance, especially in long sequences.

## Computational Considerations

### Overhead Analysis

Online EWC adds minimal overhead per batch:

**Standard EWC**:
- Post-task importance computation: $O(n)$ parameters (one-time)

**Online EWC**:
- Per-batch importance update: $O(n)$ parameters (every batch)

For typical mini-batch training, additional cost ≈ 10% of gradient computation.

### Memory Requirements

Store importance estimates:

$$\text{Memory} = \text{Model Size} + \text{Importance Size}$$

Importance matrix same dimension as model: doubles parameter memory.

**Optimization**: Use single-precision (float32) for importance, float16 for computation.

## Hyperparameter Interaction

### $\lambda$ and $\rho$ Trade-offs

| Scenario | $\rho$ | $\lambda$ | Effect |
|----------|-------|----------|--------|
| **Stable Gradients** | High (0.99) | Moderate | Smooth learning |
| **Noisy Gradients** | Low (0.9) | High | Robust to noise |
| **Long Sequences** | Medium (0.95) | High | Prevent accumulation |
| **Fast Adaptation** | High (0.99) | Low | Allow plasticity |

### Joint Tuning Strategy

1. **Fix $\rho = 0.9$** (reasonable default)
2. **Sweep $\lambda$** from 0.01 to 5.0
3. **Evaluate** on task sequence (1-5 tasks)
4. **Select $\lambda$** maximizing validation accuracy across all tasks
5. **Refine $\rho$** if needed based on gradient statistics

## Financial Applications

!!! warning "Adaptive Financial Systems"
    
    Use Online EWC for continuous learning from streaming market data:
    
    **Hour 1**: Learn current market patterns, compute importance
    **Hour 2-24**: Continuous online importance updates as market evolves
    **Day 2**: Introduce new asset class, online EWC protects learned market knowledge
    
    Online importance captures which market features remain critical as conditions change.

### Multi-Frequency Trading Adaptation

Adapt trading models as market liquidity and volatility change:

1. **Training Period**: Initialize on recent market data
2. **Trading Period**: Continue online importance updates during live trading
3. **Regime Shift**: When new regime detected, learn new patterns with protection from online importance
4. **Continuous Adaptation**: Market-aware parameter importance guides learning

## Gradient-Based Importance Variants

### Alternative Importance Measures

Instead of squared gradients, use:

**Absolute Gradients**:
$$F_i = (1-\rho) F_i + \rho |g_i|$$

More robust to sign flips, less sensitive to magnitude.

**Exponential Moving Standard Deviation**:
$$F_i = (1-\rho) F_i + \rho \sqrt{g_i^2 - \bar{g}_i^2}$$

Captures gradient variance more directly.

### Adaptive Decay

Adjust $\rho$ based on gradient statistics:

$$\rho(b) = \rho_0 + (1-\rho_0) \cdot \text{softmax}(\text{gradient\_entropy})$$

Higher entropy (uncertain gradients) → lower $\rho$ (more averaging)

## Research Directions

- Optimal decay rate selection from data
- Combining online EWC with other continual learning methods
- Scalable online importance computation for massive networks
- Theoretical convergence guarantees for online estimates

## Related Topics

- Elastic Weight Consolidation (Chapter 12.3.2)
- Regularization Overview (Chapter 12.3.1)
- Synaptic Intelligence
- Memory Aware Synapses
