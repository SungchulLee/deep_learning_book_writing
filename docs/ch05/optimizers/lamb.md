# LAMB

Training large models like BERT with very large batch sizes (e.g., 32K or 65K samples) can dramatically reduce wall-clock time, but standard [Adam](adam.md) often diverges in this regime.  The root cause is that different layers of a deep network can have vastly different gradient magnitudes, and a single global learning rate cannot accommodate all of them simultaneously when the batch is large enough to reduce gradient noise.  **LAMB** (Layer-wise Adaptive Moments optimizer for Batch training; You et al., 2020) addresses this by adding a per-layer **trust ratio** on top of Adam's per-parameter adaptive step, enabling stable training at batch sizes up to 65,536 and reducing BERT pre-training from days to just 76 minutes.

## Algorithm

LAMB builds on [Adam](adam.md).  At each time step $t$, let $g_t = \nabla_\theta L(\theta_t)$ denote the gradient.  For each layer $l$ with parameter tensor $\theta_t^{(l)}$, LAMB performs the following steps.  All element-wise operations are applied within each layer.

**Step 1.** Update the biased first and second moment estimates (identical to Adam):

$$
m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t
$$

$$
v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2
$$

**Step 2.** Compute bias-corrected moments:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Step 3.** Compute the layer-wise update direction $r_t^{(l)}$, which combines the Adam step with decoupled weight decay (coefficient $\lambda$):

$$
r_t^{(l)} = \frac{\hat{m}_t^{(l)}}{\sqrt{\hat{v}_t^{(l)}} + \epsilon} + \lambda \, \theta_t^{(l)}
$$

**Step 4.** Compute the **trust ratio** $\phi^{(l)}$ using $\ell_2$ norms:

$$
\phi^{(l)} = \frac{\|\theta_t^{(l)}\|_2}{\|r_t^{(l)}\|_2}
$$

If $\|\theta_t^{(l)}\|_2 = 0$ or $\|r_t^{(l)}\|_2 = 0$, the trust ratio is set to $1$.

**Step 5.** Apply the layer-wise scaled update:

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \, \phi^{(l)} \, r_t^{(l)}
$$

Here $\eta$ is the global learning rate, $\beta_1$ and $\beta_2$ are the moment decay rates (typically $0.9$ and $0.999$), and $\epsilon$ is a small constant for numerical stability.

## Trust Ratio Intuition

The trust ratio $\phi^{(l)} = \|\theta_t^{(l)}\|_2 / \|r_t^{(l)}\|_2$ rescales the update for each layer so that the relative change $\|\Delta\theta^{(l)}\|_2 / \|\theta^{(l)}\|_2$ is controlled by the learning rate $\eta$ alone, regardless of the raw gradient magnitude.  Layers with small parameters receive proportionally small updates, and layers with large parameters receive proportionally large updates, preventing any single layer from dominating the optimization step.

!!! note "Relationship to LARS"
    LAMB can be viewed as a combination of Adam and LARS (Layer-wise Adaptive Rate Scaling).  LARS applies the same trust ratio idea to SGD with momentum.  LAMB extends it to Adam, inheriting Adam's per-parameter adaptivity while adding LARS's per-layer scaling.

## PyTorch Example

```python
"""LAMB optimizer usage example for large-batch training."""

# === LAMB Optimizer Setup ===

# LAMB is not in core PyTorch; install via:
#   pip install torch-optimizer

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    try:
        from torch_optimizer import Lamb

        model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        optimizer = Lamb(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Simulated training step
        x = torch.randn(64, 768)
        y = torch.randn(64, 768)
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")

    except ImportError:
        print("Install torch-optimizer: pip install torch-optimizer")
```

## When to Use LAMB

LAMB is designed for a specific regime:

- **Large-batch distributed training**: batch sizes of 8K--65K across many GPUs/TPUs, where standard Adam or [AdamW](adamw.md) becomes unstable.
- **Pre-training large transformers**: the original paper demonstrated its effectiveness for BERT pre-training at scale.
- **Linear scaling of batch size**: LAMB allows increasing the batch size with near-linear speedup, maintaining convergence quality.

For standard single-GPU training with batch sizes under 1024, [AdamW](adamw.md) is simpler and sufficient.  The trust ratio adds per-layer overhead and complexity that provides no benefit at small batch scales.

## Reference

- You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., & Hsieh, C.-J. (2020). Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes. *ICLR 2020*.
