# Training Optimization for Transformers

## Overview

Training Transformers effectively requires specific optimization strategies that address their unique characteristics: deep residual networks, attention-based computation with quadratic memory scaling, and sensitivity to hyperparameters. This section covers learning rate schedules, regularization techniques, and strategies for managing memory and computational costs.

## Learning Rate Schedules

### The Warmup Problem

Transformers are sensitive to the learning rate during early training. Large initial learning rates cause unstable updates because the randomly initialized attention weights produce high-variance gradients. The original Transformer paper introduced a warmup schedule to address this.

### Original Transformer Schedule

The "Attention Is All You Need" paper uses a schedule that increases linearly during warmup, then decays proportionally to the inverse square root of the step number:

$$
lr = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

This schedule has two phases:

1. **Warmup phase** (step $\leq$ warmup\_steps): Learning rate increases linearly from 0 to the peak value $d_{\text{model}}^{-0.5} \cdot \text{warmup\_steps}^{-0.5}$.
2. **Decay phase** (step $>$ warmup\_steps): Learning rate decays as $\text{step}^{-0.5}$.

```python
import torch
import math


class TransformerLRScheduler:
    """
    Original Transformer learning rate schedule with warmup.
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        """Update learning rate and advance step counter."""
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _compute_lr(self) -> float:
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
```

### Cosine Annealing with Warmup

A widely used alternative combines linear warmup with cosine decay:

$$
lr(t) = \begin{cases}
lr_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\[6pt]
lr_{\min} + \frac{lr_{\max} - lr_{\min}}{2}\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right) & t > T_{\text{warmup}}
\end{cases}
$$

```python
class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 1e-4,
        lr_min: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _compute_lr(self) -> float:
        if self.step_num <= self.warmup_steps:
            # Linear warmup
            return self.lr_max * self.step_num / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )
```

### Schedule Comparison

| Schedule | Peak LR | Warmup | Decay | Used By |
|----------|---------|--------|-------|---------|
| Inverse sqrt | $\propto d_{\text{model}}^{-0.5}$ | Linear | $t^{-0.5}$ | Original Transformer |
| Cosine warmup | Tunable | Linear | Cosine | GPT-3, LLaMA |
| Linear warmup + linear decay | Tunable | Linear | Linear | BERT |
| Constant with warmup | Tunable | Linear | None | Fine-tuning |

## Regularization Techniques

### Gradient Clipping

Gradient clipping prevents exploding gradients by capping the gradient norm before parameter updates:

$$
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\tau \cdot \frac{g}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}
$$

where $\tau$ is the clipping threshold (typically 1.0 for Transformers).

```python
import torch.nn as nn


def train_step_with_clipping(model, optimizer, criterion, src, tgt, max_norm=1.0):
    """Training step with gradient clipping."""
    model.train()
    optimizer.zero_grad()
    
    output = model(src, tgt)
    loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    loss.backward()
    
    # Clip gradients by global norm
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    
    optimizer.step()
    return loss.item(), grad_norm.item()
```

### Dropout

The Transformer applies dropout at several points in the architecture:

1. **After positional encoding addition**: Drops random elements of the combined embedding
2. **After attention weights** (within multi-head attention): Drops random attention connections
3. **After each sub-layer** (before the residual addition): Drops sub-layer outputs
4. **Within the feed-forward network**: Drops intermediate activations

Typical dropout rate: 0.1 for large models, 0.3 for smaller models.

```python
class TransformerBlockWithDropout(nn.Module):
    """Shows all dropout application points."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,           # Dropout on attention weights
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),       # Dropout in FFN
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention sub-layer
        self.dropout2 = nn.Dropout(dropout)  # Dropout after FFN sub-layer
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        
        return x
```

### Weight Decay

Weight decay (L2 regularization) adds a penalty proportional to the squared magnitude of weights, discouraging overly large parameter values:

$$
\theta_{t+1} = \theta_t - \eta \left(\nabla_\theta \mathcal{L} + \lambda \theta_t\right)
$$

With AdamW (the standard optimizer for Transformers), weight decay is applied directly to the parameters rather than through the gradient, which is important for correct interaction with adaptive learning rates:

$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

```python
import torch.optim as optim


def configure_optimizer(model, lr=1e-4, weight_decay=0.01):
    """
    Configure AdamW with weight decay applied only to weight matrices,
    not to biases or layer normalization parameters.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases and LayerNorm parameters from weight decay
        if 'bias' in name or 'norm' in name or 'layernorm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    return optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)
```

## Memory and Computational Challenges

### Quadratic Attention Cost

Self-attention has $O(n^2 \cdot d)$ time and $O(n^2)$ memory complexity, where $n$ is the sequence length. For a model with $d_{\text{model}} = 1024$ and $n = 2048$:

$$
\text{Attention matrix size per head} = n^2 = 2048^2 = 4{,}194{,}304 \text{ elements}
$$

For 16 heads in float32, a single attention layer stores approximately 256 MB of attention matrices during training.

### Gradient Checkpointing

Gradient checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them:

- **Without checkpointing**: Store all intermediate activations → $O(N \cdot L)$ memory, where $N$ is sequence length and $L$ is the number of layers
- **With checkpointing**: Store only layer inputs, recompute activations → $O(N + L)$ memory, but ~33% slower training

```python
from torch.utils.checkpoint import checkpoint


class CheckpointedEncoder(nn.Module):
    """Encoder with gradient checkpointing for reduced memory."""
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithDropout(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.use_checkpoint = True
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Recompute activations during backward pass
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        return x
```

### Mixed Precision Training

Mixed precision uses float16 for most computations while maintaining float32 for numerically sensitive operations (loss scaling, normalization, softmax):

```python
from torch.amp import autocast, GradScaler


def train_step_mixed_precision(
    model, optimizer, criterion, src, tgt, scaler, device
):
    """Training step with automatic mixed precision."""
    model.train()
    optimizer.zero_grad()
    
    with autocast(device_type=device.type, dtype=torch.float16):
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
    
    # Scale loss to prevent underflow in float16 gradients
    scaler.scale(loss).backward()
    
    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


# Usage
scaler = GradScaler()
loss = train_step_mixed_precision(model, optimizer, criterion, src, tgt, scaler, device)
```

**Memory savings with mixed precision**:

| Component | float32 | float16 | Savings |
|-----------|---------|---------|---------|
| Model parameters | 4 bytes/param | 2 bytes/param | 50% |
| Activations | 4 bytes/elem | 2 bytes/elem | 50% |
| Optimizer states (Adam) | 12 bytes/param | 8 bytes/param | 33% |

### Efficient Attention Mechanisms

For long sequences, several alternatives reduce the $O(n^2)$ attention cost:

| Method | Complexity | Approach |
|--------|-----------|----------|
| Flash Attention | $O(n^2)$ time, $O(n)$ memory | IO-aware tiling |
| Sparse Attention | $O(n \sqrt{n})$ | Attend to subset of positions |
| Linear Attention | $O(n)$ | Kernel approximation |
| Sliding Window | $O(n \cdot w)$ | Local attention window |

See the dedicated pages on [Flash Attention](flash_attention.md) and [Sparse Attention Patterns](sparse_attention.md) for details.

## Complete Optimized Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    d_model: int = 512,
    warmup_steps: int = 4000,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    use_amp: bool = True,
    device: str = "cuda"
):
    """
    Complete training pipeline with standard Transformer optimizations.
    
    Includes: AdamW, warmup schedule, gradient clipping, mixed precision,
    label smoothing, and validation evaluation.
    """
    device = torch.device(device)
    model = model.to(device)
    
    # Optimizer with separated weight decay
    optimizer = configure_optimizer(model, lr=1e-4, weight_decay=weight_decay)
    
    # Learning rate scheduler
    total_steps = num_epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps,
        total_steps=total_steps, lr_max=1e-4
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,                                # Ignore padding
        label_smoothing=label_smoothing
    )
    
    # Mixed precision
    scaler = GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=use_amp):
                tgt_mask = torch.triu(
                    torch.ones(tgt_input.size(1), tgt_input.size(1), device=device),
                    diagonal=1
                ).bool()
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_labels.reshape(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_labels = tgt[:, 1:]
                
                tgt_mask = torch.triu(
                    torch.ones(tgt_input.size(1), tgt_input.size(1), device=device),
                    diagonal=1
                ).bool()
                
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_labels.reshape(-1)
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Checkpoint best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_transformer.pt")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )
```

## Summary

Effective Transformer training combines several key strategies:

1. **Learning rate warmup** prevents training instability from random initialization; the original inverse-sqrt schedule and cosine warmup are both widely used.
2. **Gradient clipping** (typically norm ≤ 1.0) guards against gradient explosions.
3. **Weight decay** (with AdamW) provides regularization while correctly interacting with adaptive learning rates. Biases and normalization parameters are excluded.
4. **Dropout** is applied at multiple points throughout the architecture.
5. **Label smoothing** prevents overconfidence and improves generalization.
6. **Gradient checkpointing** and **mixed precision** are essential for training large models within memory constraints.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR. (AdamW)
3. Micikevicius, P., et al. (2018). "Mixed Precision Training." ICLR.
4. Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost." (Gradient checkpointing)
5. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
