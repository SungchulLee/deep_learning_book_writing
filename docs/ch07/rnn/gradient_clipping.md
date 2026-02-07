# Gradient Clipping

## Introduction

Gradient clipping is the standard technique for preventing exploding gradients during RNN training. By limiting gradient magnitudes before the optimizer step, clipping provides a simple, effective safeguard that preserves training stability while allowing the model to learn from large but finite gradient signals.

## Clipping Strategies

### Gradient Norm Clipping (Recommended)

The standard approach scales all gradients uniformly if their combined $L_2$ norm exceeds a threshold $\tau$:

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\ \tau \cdot \dfrac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \tau \end{cases}$$

This preserves gradient **direction** while limiting magnitude—a critical property because the direction contains information about which way to update parameters, even when the magnitude is unreliable.

```python
import torch
import torch.nn as nn
import numpy as np


def train_step_with_clipping(model, optimizer, loss, max_norm=1.0):
    """Standard training step with gradient norm clipping."""
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients by global norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=max_norm
    )
    
    optimizer.step()
    return total_norm  # Useful for monitoring
```

The PyTorch function `clip_grad_norm_` returns the original (pre-clipping) gradient norm, which is valuable for monitoring gradient health.

### Manual Implementation

```python
def clip_grad_norm_manual(parameters, max_norm):
    """Manual gradient norm clipping for understanding."""
    parameters = [p for p in parameters if p.grad is not None]
    
    # Compute total L2 norm across all parameters
    total_norm = 0
    for p in parameters:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Scale if exceeding threshold
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
```

### Gradient Value Clipping

An alternative that clips each gradient element independently to $[-\tau, \tau]$:

```python
# PyTorch built-in
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Manual implementation
def clip_grad_value_manual(parameters, clip_value):
    """Clip each gradient element to [-clip_value, clip_value]."""
    for p in parameters:
        if p.grad is not None:
            p.grad.data.clamp_(-clip_value, clip_value)
```

Value clipping can **alter gradient direction** because it clips each element independently. If one component is large and others are small, the resulting vector points in a different direction than the original gradient. For this reason, norm clipping is generally preferred for RNN training.

### Adaptive Clipping

Rather than using a fixed threshold, adaptive clipping adjusts based on gradient history:

```python
class AdaptiveGradientClipper:
    """Adaptive clipping threshold based on gradient statistics."""
    
    def __init__(self, percentile=90, warmup_steps=100):
        self.percentile = percentile
        self.warmup_steps = warmup_steps
        self.gradient_history = []
        self.step = 0
    
    def clip(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_history.append(total_norm)
        self.step += 1
        
        if self.step < self.warmup_steps:
            max_norm = 1.0  # Fixed during warmup
        else:
            max_norm = np.percentile(self.gradient_history, self.percentile)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return total_norm, max_norm
```

## Choosing the Clipping Threshold

### Empirical Guidelines

| Model Type | Typical `max_norm` |
|------------|-------------------|
| Vanilla RNN | 1.0–5.0 |
| LSTM / GRU | 1.0–10.0 |
| Deep RNN (3+ layers) | 0.5–2.0 |
| Seq2Seq | 1.0–5.0 |
| Transformer | 1.0 |

A common starting point is `max_norm=1.0`. The threshold should be tuned based on monitoring: too low clips useful gradient signal, too high fails to prevent instability.

### Empirical Threshold Discovery

Run a few batches without clipping to characterize the gradient distribution:

```python
def find_clipping_threshold(model, dataloader, criterion, device, 
                            num_batches=100):
    """Analyze gradient norms to determine appropriate clipping threshold."""
    gradient_norms = []
    
    model.train()
    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        gradient_norms.append(total_norm ** 0.5)
    
    gradient_norms = np.array(gradient_norms)
    
    print(f"Mean: {gradient_norms.mean():.4f}")
    print(f"Median: {np.median(gradient_norms):.4f}")
    print(f"95th percentile: {np.percentile(gradient_norms, 95):.4f}")
    print(f"Max: {gradient_norms.max():.4f}")
    
    suggested = np.percentile(gradient_norms, 90)
    print(f"\nSuggested max_norm: {suggested:.2f}")
    
    return gradient_norms, suggested
```

## Monitoring Gradient Health

Tracking gradient statistics during training reveals whether clipping is working effectively:

```python
class GradientMonitor:
    """Monitor gradient statistics during training."""
    
    def __init__(self):
        self.norms = []
        self.clipped_count = 0
        self.total_count = 0
    
    def log(self, model, max_norm):
        """Log gradient norm before clipping."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.norms.append(total_norm)
        self.total_count += 1
        if total_norm > max_norm:
            self.clipped_count += 1
        
        return total_norm
    
    def summary(self):
        """Print summary statistics."""
        norms = np.array(self.norms)
        clip_rate = self.clipped_count / max(self.total_count, 1)
        
        print(f"Gradient Monitoring Summary:")
        print(f"  Total steps: {self.total_count}")
        print(f"  Clipping rate: {clip_rate:.1%}")
        print(f"  Mean norm: {norms.mean():.4f}")
        print(f"  Max norm: {norms.max():.4f}")
        
        if clip_rate > 0.5:
            print("  → High clipping rate: consider lowering learning rate")
        elif clip_rate < 0.01:
            print("  → Low clipping rate: threshold may be unnecessarily high")
    
    def plot(self):
        """Plot gradient norm history."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(self.norms, alpha=0.7)
        plt.axhline(y=np.mean(self.norms), color='r', linestyle='--', label='Mean')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm (pre-clip)')
        plt.title('Gradient Norm During Training')
        plt.legend()
        plt.show()
```

### Interpreting the Clipping Rate

The fraction of steps where clipping activates reveals the health of training:

| Clipping Rate | Interpretation | Action |
|---------------|---------------|--------|
| < 1% | Clipping rarely needed | Threshold may be too conservative |
| 5–25% | Healthy range | Training is stable |
| 25–50% | Frequent clipping | Consider reducing learning rate |
| > 50% | Excessive clipping | Reduce learning rate or check for bugs |

## Complete Training Loop

```python
def train_rnn_with_clipping(model, train_loader, val_loader,
                            epochs, lr, max_norm, device):
    """Complete training loop with gradient clipping and monitoring."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    monitor = GradientMonitor()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            
            # Monitor before clipping
            pre_clip_norm = monitor.log(model, max_norm)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Detect anomalies
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at batch {batch_idx}")
                print(f"Pre-clip gradient norm: {pre_clip_norm:.4f}")
                break
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")
    
    monitor.summary()
    return model
```

## When to Use Gradient Clipping

**Always recommended for**: vanilla RNNs, deep networks (many layers), long sequences, and early training phases where gradients are unpredictable.

**Usually helpful for**: LSTM and GRU networks, seq2seq models, and any recurrent architecture.

**May not be necessary for**: well-initialized shallow networks, very short sequences, and fine-tuning pre-trained models.

In practice, the computational overhead of gradient clipping is negligible (a single norm computation and optional scaling), so there is little reason not to include it as a default safety measure in any RNN training pipeline.

## Summary

Gradient clipping prevents exploding gradients by scaling the gradient vector when its norm exceeds a threshold. Norm clipping (the recommended approach) preserves gradient direction while limiting magnitude; value clipping operates element-wise but can distort direction. Best practice is to start with `max_norm=1.0`, monitor the clipping rate during training (targeting 5–25% of steps), and adjust based on observed gradient statistics. Combined with proper initialization and gated architectures, gradient clipping provides a robust foundation for stable RNN training.
