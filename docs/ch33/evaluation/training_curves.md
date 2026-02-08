# 33.6.1 Training Curves

## What to Monitor

### Primary Metrics
- **Episode return**: Total undiscounted reward per episode. The main performance indicator
- **Rolling average return**: Smoothed return (e.g., 100-episode window) to identify trends
- **Evaluation return**: Periodic greedy evaluation (ε=0) on separate episodes

### Diagnostic Metrics
- **Loss**: TD loss; should generally decrease but may be noisy
- **Q-value estimates**: Mean and max Q-values; should grow but not diverge
- **Gradient norms**: Detect exploding gradients (should be bounded)
- **Epsilon**: Verify the exploration schedule is working
- **Buffer utilization**: Fraction of buffer filled

## Interpreting Training Curves

### Healthy Training
- Episode returns trend upward with decreasing variance
- Loss decreases then stabilizes
- Q-values grow proportionally to actual returns
- Gradient norms are bounded

### Common Pathologies

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Return plateaus early | Insufficient exploration | Increase ε decay period |
| Return collapses after improvement | Target net instability | Increase C or decrease τ |
| Q-values diverge to ±∞ | Overestimation cascade | Use Double DQN, reduce LR |
| Loss increases over time | Bootstrapping instability | Gradient clipping, Huber loss |
| High variance returns throughout | Function approximation error | Increase network capacity |
| Sudden performance drops | Catastrophic forgetting | Larger replay buffer |

## Evaluation Best Practices

1. **Separate evaluation episodes**: Never evaluate on training episodes
2. **Greedy policy**: Use ε=0 during evaluation
3. **Multiple seeds**: Run 3–10 seeds per configuration
4. **Sufficient episodes**: Evaluate over 20+ episodes per evaluation point
5. **Track both mean and variance**: Report mean ± std or confidence intervals

## Smoothing Techniques

- **Simple moving average**: Window of 50–100 episodes
- **Exponential moving average**: $\bar{R}_t = \beta \bar{R}_{t-1} + (1-\beta) R_t$, with $\beta = 0.99$
- **Median instead of mean**: More robust to outliers
- **Interquartile range**: Shows 25th–75th percentile band
