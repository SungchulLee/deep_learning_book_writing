# KL Annealing

Gradually introducing the KL divergence penalty to prevent posterior collapse.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain why KL annealing helps prevent posterior collapse
- Implement linear, cyclical, and sigmoid annealing schedules
- Choose appropriate annealing hyperparameters
- Understand the trade-offs of different schedules

---

## Why KL Annealing?

At the start of training, the decoder is randomly initialized and produces poor reconstructions regardless of the latent code. The KL term provides a strong, clear gradient pushing $q_\phi(z|x)$ toward $\mathcal{N}(0, I)$. If the KL penalty is fully active from the beginning, the encoder collapses to the prior before the decoder can learn to use latent information.

**KL annealing** addresses this by starting with a low KL weight and gradually increasing it, giving the encoder and decoder time to establish a useful latent representation.

---

## The Annealed Objective

$$\mathcal{L}_{\text{annealed}} = \mathbb{E}_q[\log p(x|z)] - \beta(t) \cdot D_{KL}(q(z|x) \| p(z))$$

where $\beta(t)$ is a schedule function that increases from 0 (or a small value) to the target weight (typically 1) over training.

---

## Annealing Schedules

### Linear Annealing

The simplest approach — linearly increase $\beta$ from 0 to target:

```python
def linear_annealing(step, warmup_steps, max_beta=1.0):
    """Linear KL annealing: β increases linearly during warmup."""
    if step < warmup_steps:
        return max_beta * step / warmup_steps
    return max_beta
```

### Sigmoid Annealing

Provides a gentler start and end with faster increase in the middle:

```python
import numpy as np

def sigmoid_annealing(step, warmup_steps, max_beta=1.0, k=10):
    """Sigmoid KL annealing: S-shaped curve."""
    x = (step - warmup_steps / 2) / (warmup_steps / k)
    return max_beta / (1 + np.exp(-x))
```

### Cyclical Annealing

Repeats the annealing process multiple times (Fu et al., 2019):

```python
def cyclical_annealing(step, cycle_length=10000, ratio=0.5, max_beta=1.0):
    """
    Cyclical KL annealing: repeat warmup cycles.
    
    Args:
        step: Current training step
        cycle_length: Steps per cycle
        ratio: Fraction of cycle spent warming up
        max_beta: Maximum β value
    """
    cycle_pos = step % cycle_length
    warmup_steps = int(cycle_length * ratio)
    
    if cycle_pos < warmup_steps:
        return max_beta * cycle_pos / warmup_steps
    return max_beta
```

### Comparison

| Schedule | Behavior | Best For |
|----------|----------|----------|
| **Linear** | Steady increase | Simple, reliable default |
| **Sigmoid** | Slow→fast→slow | Smoother transitions |
| **Cyclical** | Repeated warmups | Strong collapse tendency |
| **None** | $\beta = 1$ always | When collapse isn't an issue |

---

## Choosing Warmup Duration

The warmup period should be long enough for the decoder to learn meaningful reconstructions from latent codes, but not so long that the model trains as a pure autoencoder for too many epochs.

Rules of thumb:

- **MNIST/simple data:** 5–10 epochs
- **CIFAR/complex images:** 20–50 epochs
- **Sequential data (text):** 10–30 epochs

Monitor the reconstruction loss: once it begins to plateau, the decoder has learned to use latent codes effectively and annealing can proceed.

---

## Implementation in Training Loop

```python
def train_with_annealing(model, loader, optimizer, device, 
                         epoch, total_epochs, schedule='linear'):
    """Training epoch with KL annealing."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(loader):
        # Compute global step
        step = epoch * len(loader) + batch_idx
        total_steps = total_epochs * len(loader)
        warmup_steps = int(0.2 * total_steps)  # 20% warmup
        
        # Get annealing weight
        if schedule == 'linear':
            beta = linear_annealing(step, warmup_steps)
        elif schedule == 'cyclical':
            beta = cyclical_annealing(step, cycle_length=warmup_steps)
        else:
            beta = 1.0
        
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        
        recon, mu, logvar = model(data)
        recon_loss = F.binary_cross_entropy(recon, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader.dataset)
```

---

## Summary

| Schedule | Formula | Key Property |
|----------|---------|-------------|
| **Linear** | $\beta(t) = \min(1, t/T_w)$ | Simple, predictable |
| **Sigmoid** | $\beta(t) = 1/(1 + e^{-k(t - T_w/2)})$ | Smooth transitions |
| **Cyclical** | Periodic linear | Repeated warm-ups help escape collapse |

---

## Exercises

### Exercise 1: Schedule Comparison

Train the same VAE with linear, sigmoid, and cyclical annealing. Compare final ELBO, reconstruction quality, and number of active latent dimensions.

### Exercise 2: Warmup Duration

Sweep warmup duration from 1 to 50 epochs. Plot active dimensions and final loss vs warmup length.

---

## What's Next

The next section covers [Free Bits](free_bits.md), an alternative approach to preventing posterior collapse.
