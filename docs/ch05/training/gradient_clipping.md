# Gradient Clipping

## Overview

Gradient clipping bounds the magnitude of gradients during training, preventing the exploding gradient problem that can destabilize optimization—particularly in recurrent networks and deep architectures. It is applied after `loss.backward()` and before `optimizer.step()`.

## Gradient Norm Clipping

The most common approach clips the global gradient norm. Given parameter gradients $g_1, \ldots, g_n$, the total norm is:

$$\|g\| = \sqrt{\sum_{i=1}^n \|g_i\|^2}$$

If $\|g\| > \text{max\_norm}$, all gradients are rescaled:

$$g_i \leftarrow g_i \cdot \frac{\text{max\_norm}}{\|g\|}$$

This preserves the direction of the gradient while limiting its magnitude.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Usage in the training loop:

```python
for x, y in train_loader:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()

    # Clip before optimizer step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

The function returns the total gradient norm before clipping, which is useful for monitoring.

## Gradient Value Clipping

An alternative clips each gradient element independently to $[-\text{clip\_value}, \text{clip\_value}]$:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Value clipping changes the gradient direction (unlike norm clipping) and is less commonly used.

## Choosing the Clip Threshold

The max norm should be set based on typical gradient magnitudes for the model:

1. Run a few training steps without clipping and log gradient norms.
2. Set `max_norm` to the 90th–95th percentile of observed norms.

Typical values: 1.0 for transformers, 5.0 for RNNs, 0.5–1.0 for LSTMs. Start conservative and relax if clipping occurs too frequently (which slows convergence).

## Monitoring Clipping Frequency

```python
max_norm = 1.0
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

if grad_norm > max_norm:
    clip_count += 1

# Log clipping ratio
clip_ratio = clip_count / total_steps
```

If clipping occurs on more than 20–30% of steps, the learning rate may be too high or the model may have architectural issues.

## Key Takeaways

- Gradient norm clipping rescales the gradient vector to bound its magnitude while preserving direction.
- Apply clipping after `backward()` and before `step()`.
- Set the clip threshold based on observed gradient norm statistics.
- Monitor clipping frequency—excessive clipping signals optimization problems.
