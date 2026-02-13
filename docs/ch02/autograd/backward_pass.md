# Backward Pass

## Overview

The backward pass is the core mechanism by which PyTorch propagates gradients from the loss function back through the computational graph to all trainable parameters. This section covers the mechanics of `.backward()`, the structure of the training loop that orchestrates forward and backward passes, gradient clipping, monitoring, and profiling tools for diagnosing performance bottlenecks.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the mechanics of `.backward()` and how it traverses the computational graph
2. Implement correct training loops with proper gradient management
3. Apply gradient clipping to prevent exploding gradients
4. Monitor gradient statistics for debugging
5. Profile forward and backward passes to identify performance bottlenecks

## Mechanics of `.backward()`

### How the Backward Pass Works

When `loss.backward()` is called on a scalar tensor, PyTorch:

1. **Initializes** the upstream gradient as $\bar{L} = 1$ (since $\partial L / \partial L = 1$)
2. **Traverses** the computational graph in reverse topological order
3. **Applies** each node's `grad_fn` to compute the local vector-Jacobian product
4. **Accumulates** gradients into the `.grad` attribute of each leaf tensor
5. **Frees** the computational graph (unless `retain_graph=True`)

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward: build graph
y = x ** 2          # PowBackward0
z = y.sum()         # SumBackward0

# Backward: traverse graph in reverse
z.backward()

# dz/dx = 2x
print(f"x.grad: {x.grad}")  # tensor([4., 6.])
```

### The `gradient` Argument

For scalar outputs, the implicit upstream gradient is $1$. For non-scalar outputs, you must provide it explicitly:

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2   # Non-scalar

# Must supply upstream gradient (same shape as y)
y.backward(gradient=torch.tensor([1.0, 1.0]))
print(f"x.grad: {x.grad}")  # tensor([2., 4.])

# Weighted gradient
x.grad = None
y = x ** 2
y.backward(gradient=torch.tensor([0.5, 2.0]))
print(f"x.grad: {x.grad}")  # tensor([1., 8.])
```

### `retain_graph` and `create_graph`

Two important keyword arguments control backward behavior:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `retain_graph` | `False` | If `True`, preserves the graph for additional backward passes |
| `create_graph` | `False` | If `True`, builds a graph of the backward pass itself (for higher-order derivatives) |

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First backward — retain graph for second pass
y.backward(retain_graph=True)
print(f"After 1st backward: x.grad = {x.grad}")  # 12.0

x.grad = None

# Second backward works because graph was retained
y.backward()
print(f"After 2nd backward: x.grad = {x.grad}")  # 12.0
```

## The Training Loop

### Fundamental Cycle

Every training iteration follows the same four-step pattern:

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x, y = torch.randn(32, 10), torch.randn(32, 1)

for epoch in range(100):
    # 1. Forward pass — build computational graph
    predictions = model(x)
    
    # 2. Compute loss
    loss = criterion(predictions, y)
    
    # 3. Backward pass — compute gradients
    optimizer.zero_grad()    # Clear previous gradients FIRST
    loss.backward()          # Populate .grad for all parameters
    
    # 4. Parameter update
    optimizer.step()         # Applies gradients (uses no_grad internally)
```

### Why Order Matters

The `zero_grad → backward → step` ordering is critical:

```python
# ✅ CORRECT: zero → backward → step
optimizer.zero_grad()   # Clear old gradients
loss.backward()         # Compute fresh gradients
optimizer.step()        # Apply them

# ❌ WRONG: gradients accumulate without zeroing
loss.backward()         # Adds to old gradients!
optimizer.step()        # Update based on accumulated (incorrect) gradients
```

### Complete Training and Validation Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """Run validation (no gradient computation)."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X, y = torch.randn(1000, 10), torch.randn(1000, 1)
train_loader = DataLoader(TensorDataset(X[:800], y[:800]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[800:], y[800:]), batch_size=32)

for epoch in range(50):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
```

### `model.train()` vs `model.eval()`

These methods affect layers with mode-dependent behavior:

| Mode | Dropout | BatchNorm |
|------|---------|-----------|
| `model.train()` | Active (random masking) | Uses batch statistics |
| `model.eval()` | Disabled (no masking) | Uses running statistics |

Note: `model.eval()` does **not** disable gradient computation — that requires `torch.no_grad()`.

## Gradient Clipping

### Preventing Exploding Gradients

Gradient clipping bounds gradient magnitudes before the parameter update. Two strategies are available:

**Clip by norm** (preferred) — scales the entire gradient vector to have a maximum norm:

```python
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    
    # Clip gradient norm to max_norm=1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

The returned `total_norm` is the original (pre-clipping) gradient norm — useful for monitoring.

**Clip by value** — clamps each gradient element independently:

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### Mathematical Detail: Norm Clipping

Given gradient vector $g$ and threshold $\tau$, norm clipping computes:

$$g_{\text{clipped}} = \begin{cases} g & \text{if } \|g\| \leq \tau \\ \frac{\tau}{\|g\|} \cdot g & \text{if } \|g\| > \tau \end{cases}$$

This preserves gradient direction while bounding magnitude.

## Monitoring Gradients

### Gradient Statistics

Logging gradient statistics helps diagnose vanishing or exploding gradients:

```python
def log_gradient_stats(model):
    """Print gradient statistics for all parameters."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            print(f"{name}: mean={g.mean():.2e}, "
                  f"std={g.std():.2e}, max={g.abs().max():.2e}")

# In training loop
loss.backward()
log_gradient_stats(model)
optimizer.step()
```

### Gradient Norm Tracking

```python
def compute_grad_norm(model, norm_type=2):
    """Compute total gradient norm across all parameters."""
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad, norm_type) for p in params]),
        norm_type
    )
    return total_norm.item()

# Track over training
grad_norms = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss.backward()
    grad_norms.append(compute_grad_norm(model))
    optimizer.step()
```

## Profiling the Backward Pass

### The `torch.autograd.profiler` API

The legacy profiler records operation-level timing for both forward and backward passes:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
x = torch.randn(64, 1024)

with torch.autograd.profiler.profile(use_cuda=False) as prof:
    y = model(x)
    loss = y.sum()
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### The Modern `torch.profiler` API

PyTorch 1.8+ provides `torch.profiler` with richer functionality including memory profiling, TensorBoard integration, and scheduling:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    y = model(x)
    loss = y.sum()
    loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

### Profiling Training Loops with Scheduling

```python
import torch.profiler

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(
        wait=1,      # Skip first step (warmup)
        warmup=1,    # Warmup (included but not measured)
        active=3,    # Profile 3 steps
        repeat=1
    ),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(5):
        x = torch.randn(32, 784)
        target = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        prof.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
```

### Custom Labels with `record_function`

Label code sections for fine-grained profiling:

```python
from torch.profiler import record_function

with torch.profiler.profile() as prof:
    with record_function("data_preprocessing"):
        x = torch.randn(64, 1024)
        x = (x - x.mean()) / x.std()
    
    with record_function("forward_pass"):
        y = model(x)
    
    with record_function("loss_computation"):
        loss = y.sum()
    
    with record_function("backward_pass"):
        loss.backward()

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Forward vs Backward Cost Comparison

```python
model = nn.Sequential(
    nn.Linear(1024, 2048), nn.ReLU(),
    nn.Linear(2048, 1024), nn.ReLU(),
    nn.Linear(1024, 10)
)
x = torch.randn(64, 1024)

# Profile forward only
with torch.autograd.profiler.profile() as prof_fwd:
    y = model(x)
print("=== Forward Pass ===")
print(prof_fwd.key_averages().table(sort_by="cpu_time_total", row_limit=5))

# Profile backward only
y = model(x)
loss = y.sum()
with torch.autograd.profiler.profile() as prof_bwd:
    loss.backward()
print("=== Backward Pass ===")
print(prof_bwd.key_averages().table(sort_by="cpu_time_total", row_limit=5))
```

### TensorBoard Integration

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(5):
        # ... training step ...
        prof.step()

# View with: tensorboard --logdir=./log/profiler
```

### Interpreting Profiler Output

Key columns in the profiler table:

| Column | Meaning |
|--------|---------|
| `Name` | Operation name (e.g., `aten::addmm`, `aten::relu`) |
| `Self CPU` | CPU time in this op only (excluding children) |
| `CPU total` | Total CPU time (including children) |
| `# of Calls` | Number of times executed |
| `Self CUDA` | GPU kernel time (with `use_cuda=True`) |

### Common Bottlenecks and Optimizations

| Finding | Optimization |
|---------|-------------|
| Large matrix multiplications dominate | Use mixed precision (`torch.cuda.amp`) |
| Expensive memory copies (`aten::copy_`) | Ensure contiguous tensors; use `pin_memory` |
| Backward pass ≫ 2× forward | Consider gradient checkpointing |
| Data loading is slow | Increase `num_workers`; use `pin_memory=True` |
| Many small operations | Use `torch.compile()` to fuse operations |

## Summary

| Step | Action | Purpose |
|------|--------|---------|
| `optimizer.zero_grad()` | Clear gradients | Prevent accumulation between steps |
| `loss.backward()` | Reverse-traverse graph | Compute gradients via VJP chain |
| `clip_grad_norm_()` | Bound gradient magnitude | Prevent exploding gradients |
| `optimizer.step()` | Update parameters | Apply learning rate × gradient |
| `model.train()` | Set training mode | Enable dropout, BN training stats |
| `model.eval()` | Set eval mode | Disable dropout, use BN running stats |
| `torch.no_grad()` | Disable tracking | Save memory during inference |

## Common Pitfalls

1. **Forgetting `optimizer.zero_grad()`** — gradients from previous iterations accumulate
2. **Calling `model.eval()` without `torch.no_grad()`** — eval mode does not disable gradient tracking
3. **Not calling `model.train()` after validation** — dropout and BN remain in eval mode
4. **Ignoring gradient clipping** for recurrent networks — RNNs are prone to exploding gradients

## References

- PyTorch Autograd Mechanics: [https://pytorch.org/docs/stable/notes/autograd.html](https://pytorch.org/docs/stable/notes/autograd.html)
- PyTorch Profiler: [https://pytorch.org/docs/stable/profiler.html](https://pytorch.org/docs/stable/profiler.html)
- PyTorch Profiler Tutorial: [https://pytorch.org/tutorials/beginner/profiler.html](https://pytorch.org/tutorials/beginner/profiler.html)
- PyTorch Profiler with TensorBoard: [https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
