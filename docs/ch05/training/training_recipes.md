# Practical Training Recipes

## Introduction

Individual training techniques — learning rate schedules, data augmentation, regularization, normalization — each provide incremental gains. However, their true power emerges when combined systematically. This section presents practical "recipes" for training deep learning models effectively, drawing from empirical findings that show how combining simple tricks can yield substantial cumulative improvements.

The philosophy is straightforward: rather than searching for a single breakthrough technique, stack multiple well-understood improvements that are individually modest but collectively significant.

## Recipe 1: Learning Rate Scheduling

The learning rate schedule is often the single most impactful hyperparameter choice beyond the model architecture itself.

### Linear Warmup + Cosine Decay

For most training scenarios, a warmup phase followed by cosine annealing provides robust performance:

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \cdot \pi)) & \text{otherwise}
\end{cases}
$$

**Why warmup matters**: At initialization, gradients can be large and noisy. A warmup phase prevents early divergence, especially important with large batch sizes where gradient noise is lower but each step has outsized impact.

**Practical defaults**:
- Warmup: 5-10% of total training steps
- Peak learning rate: scale linearly with batch size ($\eta = \eta_{\text{base}} \times \frac{B}{B_{\text{ref}}}$)
- Minimum learning rate: $\eta_{\max} / 100$ or $1 \times 10^{-6}$

### Large Batch Training

When scaling to large batches for faster training:

1. **Linear scaling rule**: Multiply the learning rate by $k$ when the batch size is multiplied by $k$
2. **Gradual warmup**: Use $T_{\text{warmup}} = 5$ epochs to ramp up from a small LR to the scaled LR
3. **Layer-wise adaptive rates**: Optimizers like LAMB automatically adjust per-layer learning rates, enabling batch sizes up to 32K without accuracy loss

## Recipe 2: Regularization Stack

Regularization techniques can be combined, but care is needed to avoid over-regularization.

### Recommended Combinations

**For small datasets** (common in quant finance — limited historical data):

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Weight decay | 1e-4 to 1e-2 | Prevent large weights |
| Dropout | 0.1-0.3 | Ensemble approximation |
| Label smoothing ($\epsilon = 0.1$) | Soften targets | Prevent overconfident predictions |
| Data augmentation | Domain-specific | Increase effective dataset size |

**For large datasets** (high-frequency tick data, large alternative data):

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Weight decay | 1e-5 to 1e-4 | Light regularization |
| Stochastic depth | 0.1-0.2 drop rate | Regularize deep networks |
| Mixup ($\alpha = 0.2$) | Interpolate samples | Smooth decision boundaries |

### Label Smoothing

Instead of training with hard one-hot targets, use soft targets:

$$
y_i^{\text{smooth}} = (1 - \epsilon) \cdot y_i + \frac{\epsilon}{K}
$$

where $K$ is the number of classes and $\epsilon$ is typically 0.1. This prevents the model from becoming overconfident and improves calibration — critical for quant models where calibrated probabilities inform position sizing.

### Mixup Training

Mixup creates virtual training examples by linearly interpolating between pairs:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$. For time-series data in quant finance, mixup can be applied to feature vectors after embedding, smoothing the learned decision boundaries and reducing overfitting to specific market regimes.

## Recipe 3: Weight Decay Best Practices

Not all parameters should be regularized equally.

### No Bias Decay

A simple but effective trick: apply weight decay only to weight matrices, not to biases or normalization parameters:

```python
def get_parameter_groups(model, weight_decay=1e-2):
    """Separate parameters into decay and no-decay groups."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
```

**Rationale**: Biases and normalization parameters have different roles than weight matrices. Regularizing them constrains the model's ability to shift activations, often hurting performance. This trick alone can yield 0.2-0.5% accuracy improvement.

## Recipe 4: Initialization for Residual Networks

For networks with residual connections (common in modern architectures):

### Zero-Initialize Final BatchNorm

Initialize the $\gamma$ parameter of the last BatchNorm layer in each residual block to zero. This makes each residual block behave like an identity mapping at initialization, effectively starting training with a shallower network that gradually deepens:

$$
\text{output} = x + \gamma \cdot F(x) \quad \xrightarrow{\gamma=0} \quad \text{output} = x
$$

This improves training stability, especially for very deep networks (100+ layers), and typically improves final accuracy by 0.1-0.3%.

## Recipe 5: Combining Tricks — A Complete Training Configuration

Here is a practical configuration that combines the above recipes. While originally validated for image classification, the principles transfer to any supervised deep learning task:

```python
training_config = {
    # Optimizer
    'optimizer': 'AdamW',
    'base_lr': 1e-3,
    'weight_decay': 0.01,
    'no_bias_decay': True,
    
    # Schedule
    'warmup_epochs': 5,
    'total_epochs': 100,
    'scheduler': 'cosine',
    'min_lr': 1e-6,
    
    # Regularization
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'dropout': 0.1,
    
    # Initialization
    'zero_init_residual': True,
    
    # Training
    'gradient_clip_norm': 1.0,
    'ema_decay': 0.999,  # Exponential moving average of weights
}
```

### Cumulative Impact

Each technique provides a modest individual gain, but they compound:

| Technique Added | Incremental Gain | Cumulative Gain |
|----------------|-----------------|-----------------|
| Baseline (SGD, step LR) | — | — |
| + Cosine LR schedule | +0.5% | +0.5% |
| + LR warmup | +0.3% | +0.8% |
| + Label smoothing | +0.3% | +1.1% |
| + No bias decay | +0.2% | +1.3% |
| + Mixup ($\alpha=0.2$) | +0.5% | +1.8% |
| + Zero $\gamma$ init | +0.2% | +2.0% |
| + Knowledge distillation | +1.0% | +3.0% |

These numbers are representative of image classification benchmarks. For financial time-series tasks, the absolute magnitudes differ but the principle of cumulative stacking holds.

## Quant Finance Considerations

When applying these recipes to financial models:

- **Label smoothing** improves probability calibration, which directly affects Kelly criterion-based position sizing and risk management.
- **Mixup** acts as a data augmentation for regime interpolation — the model learns smoother transitions between market states rather than sharp boundaries.
- **Large batch training** is essential when processing tick-level data across thousands of securities. Linear LR scaling ensures convergence.
- **No bias decay** is especially important when model features have heterogeneous scales (price levels vs. volume ratios vs. sentiment scores).
- **EMA weights** provide more stable predictions for production deployment, reducing turnover from noisy daily rebalancing signals.

## Key Takeaways

1. **Stack incrementally**: Add one technique at a time and validate the improvement before adding the next.
2. **Warmup is non-negotiable**: Especially with adaptive optimizers (Adam, AdamW) and large batches.
3. **Regularize selectively**: Not all parameters benefit from weight decay.
4. **Calibration matters**: In quant finance, well-calibrated probabilities are often more valuable than raw accuracy.
5. **Reproducibility**: Fix all random seeds and document the exact recipe for audit compliance and strategy replication.

## References

1. He, T., et al. (2019). "Bag of Tricks for Image Classification with Convolutional Neural Networks." CVPR.
2. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv.
3. You, Y., et al. (2020). "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes." ICLR.
4. Zhang, H., et al. (2018). "mixup: Beyond Empirical Risk Minimization." ICLR.
5. Muller, R., et al. (2019). "When Does Label Smoothing Help?" NeurIPS.
