# Gradient Penalty

Gradient penalty enforces Lipschitz constraints by penalizing discriminator gradients that deviate from a target norm.

## WGAN-GP

The most common gradient penalty, introduced with Wasserstein GAN:

### Formulation

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

where $\hat{x}$ is sampled uniformly along lines between real and fake samples.

### Implementation

```python
def gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=10):
    """Compute WGAN-GP gradient penalty."""
    batch_size = real_samples.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Get discriminator output
    d_output = D(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_output),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty: (||grad|| - 1)^2
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

## R1 Regularization

Simpler penalty on real samples only (used in StyleGAN):

$$\mathcal{L}_{R1} = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{data}}[\|\nabla_x D(x)\|^2]$$

```python
def r1_regularization(D, real_samples, gamma=10):
    """R1 gradient penalty on real samples."""
    real_samples.requires_grad_(True)
    
    d_output = D(real_samples)
    
    gradients = torch.autograd.grad(
        outputs=d_output.sum(),
        inputs=real_samples,
        create_graph=True
    )[0]
    
    penalty = gamma / 2 * gradients.pow(2).sum([1, 2, 3]).mean()
    return penalty
```

## R2 Regularization

Penalty on fake samples:

```python
def r2_regularization(D, fake_samples, gamma=10):
    """R2 gradient penalty on fake samples."""
    fake_samples.requires_grad_(True)
    
    d_output = D(fake_samples)
    
    gradients = torch.autograd.grad(
        outputs=d_output.sum(),
        inputs=fake_samples,
        create_graph=True
    )[0]
    
    penalty = gamma / 2 * gradients.pow(2).sum([1, 2, 3]).mean()
    return penalty
```

## Comparison

| Method | Where Applied | Target |
|--------|---------------|--------|
| WGAN-GP | Interpolated | ||grad|| = 1 |
| R1 | Real samples | ||grad|| → 0 |
| R2 | Fake samples | ||grad|| → 0 |

## Lazy Regularization

Apply penalty every N steps for efficiency:

```python
def train_step(D, G, real, z, step, reg_interval=16, lambda_gp=10):
    # Standard loss every step
    fake = G(z)
    d_loss = d_loss_fn(D(real), D(fake))
    
    # Gradient penalty every reg_interval steps
    if step % reg_interval == 0:
        gp = gradient_penalty(D, real, fake) * reg_interval
        d_loss += gp
    
    return d_loss
```

## Summary

| Aspect | WGAN-GP | R1 |
|--------|---------|-----|
| Samples | Interpolated | Real only |
| Target | ||grad|| = 1 | ||grad|| → 0 |
| Used in | WGAN | StyleGAN |
| λ typical | 10 | 10 |
