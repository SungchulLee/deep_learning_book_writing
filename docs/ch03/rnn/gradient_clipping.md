# Gradient Clipping

## Introduction

Gradient clipping is a technique to prevent exploding gradients by limiting gradient magnitudes during training. When gradients become excessively large, they can cause unstable parameter updates that derail training entirely. Clipping provides a simple, effective safeguard.

## The Exploding Gradient Problem

During backpropagation through time, gradients are multiplied repeatedly:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

When $\|W_{hh}\| > 1$ or gradient terms exceed 1, this product grows exponentially, causing:
- Parameter updates that overshoot optimal values
- Numerical overflow (NaN/Inf values)
- Complete training failure

## Clipping Strategies

### Gradient Norm Clipping (Recommended)

Scale all gradients if their combined norm exceeds a threshold:

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq \tau \\ \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \text{if } \|\mathbf{g}\| > \tau \end{cases}$$

This preserves gradient direction while limiting magnitude.

```python
import torch
import torch.nn as nn

# PyTorch built-in implementation
def train_step_with_clipping(model, optimizer, loss, max_norm=1.0):
    """Training step with gradient norm clipping."""
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients by global norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=max_norm
    )
    
    optimizer.step()
    
    return total_norm  # Useful for monitoring

# Manual implementation
def clip_grad_norm_manual(parameters, max_norm):
    """Manual gradient norm clipping."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    # Compute total norm
    total_norm = 0
    for p in parameters:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm
```

### Gradient Value Clipping

Clip each gradient element independently to $[-\tau, \tau]$:

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

**Drawback**: Can significantly alter gradient direction, potentially harming optimization.

### Adaptive Clipping

Adjust clipping threshold based on gradient history:

```python
class AdaptiveGradientClipper:
    """Adaptive clipping based on gradient statistics."""
    
    def __init__(self, percentile=90, warmup_steps=100):
        self.percentile = percentile
        self.warmup_steps = warmup_steps
        self.gradient_history = []
        self.step = 0
    
    def clip(self, model):
        # Compute current gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_history.append(total_norm)
        self.step += 1
        
        # During warmup, use fixed threshold
        if self.step < self.warmup_steps:
            max_norm = 1.0
        else:
            # Use percentile of historical gradients
            max_norm = np.percentile(self.gradient_history, self.percentile)
        
        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        return total_norm, max_norm
```

## Choosing the Clipping Threshold

### Empirical Guidelines

| Model Type | Typical max_norm |
|------------|------------------|
| Vanilla RNN | 1.0 - 5.0 |
| LSTM/GRU | 1.0 - 10.0 |
| Deep RNN (3+ layers) | 0.5 - 2.0 |
| Seq2Seq | 1.0 - 5.0 |
| Transformer | 1.0 |

### Finding Optimal Threshold

```python
def find_clipping_threshold(model, dataloader, criterion, device, 
                            num_batches=100):
    """
    Analyze gradient norms to determine appropriate clipping threshold.
    """
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
        
        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        gradient_norms.append(total_norm)
    
    gradient_norms = np.array(gradient_norms)
    
    print("Gradient Norm Statistics:")
    print(f"  Mean: {gradient_norms.mean():.4f}")
    print(f"  Std:  {gradient_norms.std():.4f}")
    print(f"  Min:  {gradient_norms.min():.4f}")
    print(f"  Max:  {gradient_norms.max():.4f}")
    print(f"  Median: {np.median(gradient_norms):.4f}")
    print(f"  95th percentile: {np.percentile(gradient_norms, 95):.4f}")
    
    # Suggested threshold: median or 90th percentile
    suggested = np.percentile(gradient_norms, 90)
    print(f"\nSuggested max_norm: {suggested:.2f}")
    
    return gradient_norms, suggested
```

## Monitoring Gradient Health

```python
class GradientMonitor:
    """Monitor gradient statistics during training."""
    
    def __init__(self):
        self.norms = []
        self.clipped_count = 0
        self.total_count = 0
    
    def log(self, model, max_norm):
        """Log gradient statistics before clipping."""
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
            print("  ⚠️ High clipping rate - consider lower learning rate")
        elif clip_rate < 0.01:
            print("  ℹ️ Low clipping rate - threshold may be too high")
    
    def plot(self):
        """Plot gradient norm history."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.norms, alpha=0.7)
        plt.axhline(y=np.mean(self.norms), color='r', linestyle='--', label='Mean')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm During Training')
        plt.legend()
        plt.show()
```

## Complete Training Loop with Clipping

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
                print(f"⚠️ NaN/Inf loss at batch {batch_idx}")
                print(f"   Pre-clip gradient norm: {pre_clip_norm:.4f}")
                break
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss:.4f}")
    
    monitor.summary()
    return model
```

## When to Use Gradient Clipping

**Always recommended for**:
- Vanilla RNNs
- Deep networks (many layers)
- Long sequences
- Early training phases

**Usually helpful for**:
- LSTM and GRU networks
- Seq2Seq models
- Any recurrent architecture

**May not be necessary for**:
- Well-initialized, shallow networks
- Very short sequences
- Pre-trained models (fine-tuning)

## Summary

Gradient clipping prevents exploding gradients through:

1. **Norm clipping**: Scale gradients to maximum norm (preserves direction)
2. **Value clipping**: Clamp individual values (may alter direction)

Best practices:
- Use norm clipping (not value clipping) for RNNs
- Start with `max_norm=1.0` and adjust based on monitoring
- Monitor clipping frequency—aim for 5-25% of steps
- Combine with proper initialization and LSTM/GRU for robust training
