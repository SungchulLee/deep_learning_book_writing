# Warmup and Scheduling

## Linear Warmup

$\text{lr}(t) = \text{lr}_{\text{peak}} \cdot t / T_{\text{warmup}}$. Prevents large, poorly-directed gradient updates when attention patterns are random early in training.

## Noam Schedule

From Vaswani et al. (2017): $\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot T_w^{-1.5})$. Combines linear warmup with inverse square root decay.

## Cosine with Warmup

The most common modern schedule:

```python
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=2000)
cosine = CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[2000])
```

## Duration Guidelines

Small models (<100M params): 500-2000 warmup steps. Medium (100M-1B): 2000-5000. Large (>1B): 5000-10000. For fine-tuning: 6-10% of total steps.

## Why Warmup Matters

Adam's second moment estimates $v_t$ are biased toward zero in early steps, leading to excessively large effective learning rates. The bias correction helps but is insufficient for deep transformers where early attention patterns are random and loss landscapes are sharp.
