# Optimizer Comparison

## Overview

This section provides a systematic comparison of optimizers across key dimensions: convergence speed, generalization, memory overhead, and sensitivity to hyperparameters.

## Summary Table

| Optimizer | Memory (per param) | Hyperparams | Convergence Speed | Generalization | Best For |
|-----------|-------------------|-------------|-------------------|----------------|----------|
| SGD | 0 | $\eta$ | Slow | Excellent | CNNs, final tuning |
| SGD+Momentum | 1 float | $\eta, \mu$ | Moderate | Excellent | Computer vision |
| Nesterov | 1 float | $\eta, \mu$ | Moderate+ | Excellent | SGD improvement |
| Adagrad | 1 float | $\eta$ | Fast (early) | Good | Sparse features |
| RMSprop | 1 float | $\eta, \alpha$ | Fast | Good | RNNs, RL |
| Adam | 2 floats | $\eta, \beta_1, \beta_2$ | Fast | Good | General default |
| AdamW | 2 floats | $\eta, \beta_1, \beta_2, \lambda$ | Fast | Very good | Transformers, default |
| RAdam | 2 floats | $\eta, \beta_1, \beta_2$ | Fast | Good | No warmup needed |
| LAMB | 2 floats | $\eta, \beta_1, \beta_2, \lambda$ | Fast | Good | Large-batch training |
| L-BFGS | $O(m \cdot d)$ | $m$, lr | Very fast | N/A | Small models, PINNs |

## Convergence Speed vs. Generalization

A consistent empirical finding: adaptive methods (Adam, AdamW) converge faster but can generalize worse than well-tuned SGD with momentum. The generalization gap is most pronounced in image classification tasks and diminishes for NLP and other domains.

For practitioners: start with AdamW for rapid iteration. If final performance matters and compute allows, try SGD with momentum and cosine annealing for the final model.

## Memory Overhead

Each optimizer state variable adds one float per parameter:

- SGD: No additional state.
- SGD + Momentum: Stores velocity $v$ (1× parameter memory).
- Adam/AdamW: Stores first moment $m$ and second moment $v$ (2× parameter memory).

For a 100M parameter model in FP32 (400 MB), Adam adds 800 MB of optimizer state.

## Optimizer Dynamics Visualization

Consider a 2D loss surface with an elongated valley. The trajectory of different optimizers reveals their character:

- **SGD**: Oscillates across the valley, slow progress along it.
- **Momentum**: Dampens oscillations, accelerates along the valley.
- **Adam**: Quickly adapts to the curvature, finding a direct path.

For ill-conditioned problems (high condition number), adaptive methods significantly outperform non-adaptive methods.

## Key Takeaways

- No single optimizer dominates all tasks.
- AdamW is the best general-purpose default.
- SGD + momentum can achieve better generalization with careful tuning.
- Memory overhead scales linearly with optimizer state: Adam uses 3× the parameter memory (parameters + 2 state tensors).
