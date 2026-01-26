# Training Loops and Gradient Management

## Overview

The training loop is the core algorithmic structure in deep learning. It orchestrates the forward pass, loss computation, backward pass, and parameter updates. This section provides comprehensive coverage of training loop patterns.

## Learning Objectives

1. Implement the fundamental training cycle: forward → loss → backward → update
2. Properly manage gradients across training iterations
3. Build training loops with gradient accumulation
4. Monitor training progress

## The Fundamental Training Cycle

### Core Pattern

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x, y = torch.randn(32, 10), torch.randn(32, 1)

for epoch in range(100):
    # 1. Forward pass
    predictions = model(x)
    
    # 2. Compute loss
    loss = criterion(predictions, y)
    
    # 3. Backward pass (MUST zero_grad first!)
    optimizer.zero_grad()
    loss.backward()
    
    # 4. Update parameters
    optimizer.step()
```

### Why Order Matters

```python
# ✅ CORRECT order
optimizer.zero_grad()  # 1. Clear old gradients
loss.backward()        # 2. Compute new gradients
optimizer.step()       # 3. Apply gradients

# ❌ WRONG: Gradients accumulate without zeroing
loss.backward()
optimizer.step()  # Uses accumulated gradients!
```

## Complete Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
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
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
```

## Gradient Accumulation

Simulate larger batch sizes with limited memory:

```python
def train_with_accumulation(model, loader, criterion, optimizer, 
                            accumulation_steps, device):
    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        
        output = model(x)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()  # Gradients accumulate
        
        accumulated_loss += loss.item() * accumulation_steps
        
        # Update every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Handle remaining batches
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return accumulated_loss / len(loader)
```

## Gradient Clipping

Prevent exploding gradients:

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = model(x, y)
    loss.backward()
    
    # Clip gradients by norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Or clip by value
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    
    optimizer.step()
```

## Monitoring Gradients

```python
def log_gradient_stats(model):
    """Log gradient statistics for debugging."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"{name}: mean={grad.mean():.2e}, "
                  f"std={grad.std():.2e}, max={grad.abs().max():.2e}")

# In training loop
loss.backward()
log_gradient_stats(model)
optimizer.step()
```

## Key Takeaways

| Step | Action | Why |
|------|--------|-----|
| `zero_grad()` | Clear gradients | Prevent accumulation |
| `backward()` | Compute gradients | Chain rule through graph |
| `step()` | Update parameters | Apply learning |
| `model.train()` | Training mode | Enable dropout, BN training |
| `model.eval()` | Eval mode | Disable dropout, BN inference |
| `torch.no_grad()` | Disable tracking | Save memory in inference |
