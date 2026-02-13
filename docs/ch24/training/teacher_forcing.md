# Teacher Forcing

## Overview

Teacher forcing is the standard training strategy for autoregressive models. During training, the model receives the ground-truth previous tokens as input rather than its own predictions, enabling efficient parallel training.

## Mechanism

At training time:
$$\text{Input: } (x_1, x_2, \ldots, x_{T-1}) \rightarrow \text{Output: } (x_2, x_3, \ldots, x_T)$$

The model predicts $x_t$ given the true prefix $x_{1:t-1}$, not the model's own predictions. This allows the entire sequence to be processed in a single forward pass.

## Advantages

1. **Parallel computation**: all positions can be computed simultaneously (no sequential dependency during training)
2. **Stable gradients**: the model always conditions on correct inputs, avoiding error accumulation
3. **Fast convergence**: direct supervision at every position

## Exposure Bias

The mismatch between training (conditions on ground truth) and inference (conditions on own predictions) is called **exposure bias**. During generation, errors in early predictions propagate and compound because the model has never learned to recover from its own mistakes.

## Mitigations

- **Scheduled sampling**: gradually replace ground-truth inputs with model predictions during training
- **Sequence-level training**: use REINFORCE or other RL methods to optimize sequence-level metrics
- **Data augmentation**: inject noise into teacher-forced inputs to simulate prediction errors
- **Beam search**: at inference time, maintain multiple hypotheses to reduce the impact of individual errors

## Implementation

```python
def teacher_forced_loss(model, x):
    # x: (batch, seq_len) ground truth tokens
    logits = model(x[:, :-1])  # input: all but last
    targets = x[:, 1:]          # target: all but first
    return F.cross_entropy(logits.reshape(-1, vocab_size), 
                          targets.reshape(-1))
```
