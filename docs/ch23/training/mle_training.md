# MLE Training

## Overview

Normalizing flows are trained by maximum likelihood estimation (MLE), directly maximizing the log-probability of the training data under the model. This is possible because flows provide exact, tractable density evaluation.

## Objective

$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x^{(i)})$$

Using the change of variables formula:

$$\log p_\theta(x) = \log p_0(f_\theta^{-1}(x)) + \log \left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|$$

where $p_0$ is the base distribution (typically standard Gaussian) and $f_\theta^{-1}$ maps data to the latent space.

## Training Loop

```python
def train_flow(model, optimizer, dataloader, epochs):
    for epoch in range(epochs):
        for x in dataloader:
            z, log_det = model.inverse(x)  # data -> latent
            log_prob = model.base_dist.log_prob(z).sum(-1) + log_det
            loss = -log_prob.mean()  # negative log-likelihood
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

## Dequantization

For discrete data (images with integer pixel values), dequantization adds uniform noise to avoid degenerate solutions:

$$\tilde{x} = x + u, \quad u \sim \text{Uniform}(0, 1)$$

Variational dequantization (learned noise distribution) further improves results.

## Training Stability

Common issues and solutions:

| Issue | Symptom | Solution |
|-------|---------|----------|
| Exploding log-det | NaN loss | Gradient clipping, reduce LR |
| Mode collapse | Low sample diversity | Increase model capacity |
| Slow convergence | Stagnant loss | Multi-scale architecture, warmup |
| Numerical instability | Inf values | Clamp activations, use residual flows |

## Bits Per Dimension

The standard metric for density models on images:

$$\text{BPD} = \frac{-\log_2 p(x)}{d}$$

where $d$ is the dimensionality. Lower is better. State-of-the-art flows achieve ~3.0 BPD on CIFAR-10 (vs ~2.5 for autoregressive models).
