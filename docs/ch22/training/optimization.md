# VAE Optimization

Practical strategies for training Variational Autoencoders effectively.

---

## Learning Objectives

By the end of this section, you will be able to:

- Configure optimizers and learning rate schedules for VAE training
- Implement a complete training loop with monitoring
- Understand the unique optimization challenges of VAEs
- Apply gradient clipping and other stabilization techniques

---

## Optimization Challenges in VAEs

VAE training presents unique challenges compared to standard neural networks. The loss function combines two competing terms (reconstruction and KL), the reparameterization trick introduces stochasticity into the gradient estimates, the KL term can dominate early training causing posterior collapse, and the reconstruction objective varies significantly with data type and loss normalization.

---

## Optimizer Selection

### Adam (Default Choice)

Adam is the standard optimizer for VAE training due to its adaptive learning rates per parameter:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

### AdamW (With Weight Decay)

Weight decay helps prevent overfitting, particularly with larger models:

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

### Practical Guidelines

| Hyperparameter | Recommended Range | Notes |
|---------------|------------------|-------|
| Learning rate | 1e-4 to 1e-3 | Start with 1e-3, reduce if unstable |
| β₁ | 0.9 | Standard setting |
| β₂ | 0.999 | Standard setting |
| Weight decay | 0 to 1e-5 | Optional, helps with large models |
| Gradient clip | 1.0 to 5.0 | Recommended for stability |

---

## Complete Training Loop

```python
import torch
import torch.nn.functional as F

def train_vae(model, train_loader, test_loader, device,
              num_epochs=50, lr=1e-3, beta=1.0,
              kl_annealing=True, warmup_epochs=10):
    """
    Complete VAE training with monitoring.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {'train_loss': [], 'train_recon': [], 'train_kl': [],
               'test_loss': [], 'test_recon': [], 'test_kl': []}
    
    for epoch in range(1, num_epochs + 1):
        # KL annealing weight
        if kl_annealing:
            kl_weight = min(1.0, epoch / warmup_epochs) * beta
        else:
            kl_weight = beta
        
        # Train
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            
            recon_loss = F.binary_cross_entropy(recon, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        n = len(train_loader.dataset)
        history['train_loss'].append(epoch_loss / n)
        history['train_recon'].append(epoch_recon / n)
        history['train_kl'].append(epoch_kl / n)
        
        # Evaluate
        model.eval()
        test_loss, test_recon, test_kl = 0, 0, 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                recon, mu, logvar = model(data)
                
                r = F.binary_cross_entropy(recon, data, reduction='sum')
                k = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                test_loss += (r + beta * k).item()
                test_recon += r.item()
                test_kl += k.item()
        
        n_test = len(test_loader.dataset)
        history['test_loss'].append(test_loss / n_test)
        history['test_recon'].append(test_recon / n_test)
        history['test_kl'].append(test_kl / n_test)
        
        scheduler.step(test_loss / n_test)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train: {epoch_loss/n:.2f} "
                  f"(R:{epoch_recon/n:.2f} K:{epoch_kl/n:.2f}) | "
                  f"Test: {test_loss/n_test:.2f} | β_eff: {kl_weight:.3f}")
    
    return history
```

---

## Loss Normalization

### Per-Sample vs Per-Batch

The choice between `reduction='sum'` and `reduction='mean'` affects the balance between reconstruction and KL terms:

| Reduction | Reconstruction Scale | KL Scale | Effect |
|-----------|---------------------|----------|--------|
| `sum` over batch | $\sum_i \sum_d \ell_{i,d}$ | $\sum_i \sum_j \text{kl}_{i,j}$ | Scales with batch size |
| `mean` over batch | $\frac{1}{B}\sum_i \sum_d \ell_{i,d}$ | $\frac{1}{B}\sum_i \sum_j \text{kl}_{i,j}$ | Invariant to batch size |

With `sum` reduction, the effective KL weight changes with batch size. Using `mean` is generally more stable across different batch sizes.

### Per-Dimension Normalization

For fair comparison across models with different data and latent dimensions, normalize by the respective dimensions:

```python
recon_per_dim = recon_loss / data_dim  # e.g., 784 for MNIST
kl_per_dim = kl_loss / latent_dim     # e.g., 32
```

---

## Monitoring and Diagnostics

Track these metrics during training:

1. **Total loss:** Should decrease steadily
2. **Reconstruction loss:** Should decrease, indicating better reconstruction
3. **KL divergence:** Should increase initially (encoder learning), then stabilize
4. **Active dimensions:** Number of latent dimensions with KL > threshold
5. **Gradient norms:** Monitor for exploding gradients

```python
def compute_gradient_norm(model):
    """Compute total gradient norm for monitoring."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
```

---

## Summary

| Strategy | Purpose |
|----------|---------|
| **Adam optimizer** | Adaptive per-parameter learning rates |
| **Learning rate scheduling** | Reduce LR when loss plateaus |
| **Gradient clipping** | Prevent exploding gradients |
| **KL annealing** | Prevent posterior collapse |
| **Loss monitoring** | Track recon/KL balance |

---

## Exercises

### Exercise 1: Optimizer Comparison

Compare Adam, AdamW, and SGD with momentum for VAE training. Plot convergence curves.

### Exercise 2: Learning Rate Sweep

Train VAEs with LR ∈ {1e-2, 1e-3, 1e-4, 1e-5}. Which is optimal for MNIST?

---

## What's Next

The next section covers [KL Annealing](kl_annealing.md), a critical technique for preventing posterior collapse.
