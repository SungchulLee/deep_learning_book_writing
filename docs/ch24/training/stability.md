# Training Stability

## Overview

Training autoregressive models at scale requires careful attention to numerical stability, gradient flow, and optimization dynamics. This section covers practical techniques for stable training.

## Common Instabilities

### Loss Spikes
Sudden increases in loss, often caused by rare data or numerical overflow. Mitigation: gradient clipping, learning rate warmup, and loss spike detection with automatic recovery.

### Gradient Vanishing in Deep Models
Very deep autoregressive models (50+ layers) can suffer from vanishing gradients. Mitigation: residual connections, careful initialization, and normalization layers.

### Softmax Overflow
For large vocabulary sizes, logits can overflow before softmax. Mitigation: subtract the maximum logit before softmax (numerically stable softmax).

## Best Practices

### Learning Rate
- Use warmup (1000–5000 steps for small models, up to 10000 for large)
- Cosine decay to near-zero by end of training
- Peak LR scales roughly as $1/\sqrt{d_{\text{model}}}$

### Gradient Clipping
Clip gradient norm to 1.0:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Weight Initialization
- Embed layers: $\mathcal{N}(0, 0.02)$
- Output projection: $\mathcal{N}(0, 0.02 / \sqrt{2L})$ where $L$ is the number of layers
- This scaling ensures the residual stream magnitude stays controlled

### Mixed Precision
Use BF16 for forward/backward, FP32 for optimizer states. Avoid FP16 for AR models due to attention logit overflow risk.

## Monitoring

Track during training: gradient norm (should be stable), loss variance across batches, attention entropy per layer (should not collapse to 0 or max), and activation magnitudes in residual stream.
