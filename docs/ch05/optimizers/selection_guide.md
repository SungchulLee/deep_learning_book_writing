# Optimizer Selection Guide

## Overview

This section provides practical guidance for choosing an optimizer based on the task, architecture, and constraints.

## Decision Framework

**Start with AdamW** ($\eta = 0.001$, $\lambda = 0.01$). It works well across most architectures and requires minimal tuning.

**Switch to SGD + Nesterov momentum** if:

- Training a CNN for image classification where final accuracy matters.
- Willing to invest time tuning learning rate and schedule.
- Model shows the Adam generalization gap.

**Consider specialized optimizers** for:

- Very large batch training → LAMB
- Unstable early training → RAdam
- Small models with full-batch gradients → L-BFGS
- Sparse features (NLP, recommendations) → Adam or Adagrad

## By Architecture

| Architecture | Recommended | Notes |
|---|---|---|
| ResNet, VGG, CNNs | SGD + Nesterov or AdamW | SGD often generalizes better |
| Transformers (NLP) | AdamW | Standard choice, with warmup |
| Vision Transformers | AdamW | Layer-wise LR decay helps |
| RNNs, LSTMs | Adam or RMSprop | Adaptive methods handle exploding gradients better |
| GANs | Adam ($\beta_1=0.0$) | Low momentum stabilizes adversarial training |
| Fine-tuning pretrained | AdamW (low LR) | Differential LR for backbone vs. head |

## By Constraints

**Limited compute**: AdamW converges faster, reducing total training time.

**Limited memory**: SGD uses no optimizer state. Consider SGD if Adam's 2× state overhead is prohibitive.

**Large-scale distributed**: LAMB for large batch training. AdamW with linear scaling for moderate parallelism.

## Quantitative Finance Recommendations

- **Neural network calibration**: AdamW with cosine annealing. The loss surface is often smooth enough for standard adaptive methods.
- **Surrogate pricing models**: Start with AdamW; switch to L-BFGS for final refinement if the model is small enough.
- **Hedging strategies**: Adam with conservative learning rates. The training signal is noisy and non-stationary.
- **Time series forecasting**: AdamW with warmup. Financial time series exhibit regime changes that benefit from adaptive methods.

## Key Takeaways

- AdamW is the safe default for rapid iteration across most tasks.
- SGD + Nesterov can outperform in final generalization for CNNs.
- Match the optimizer to the architecture, task, and compute budget.
- In quantitative finance, adaptive optimizers (AdamW, Adam) are generally preferred due to noisy, non-stationary loss landscapes.
