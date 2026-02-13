# PyTorch Loss & Optimizer Quick Reference Guide

## 🔥 Essential Imports
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
```

---

## 📊 Loss Functions

### Regression Losses

```python
# Mean Squared Error (L2 Loss)
criterion = nn.MSELoss()
loss = criterion(predictions, targets)

# Mean Absolute Error (L1 Loss)
criterion = nn.L1Loss()
loss = criterion(predictions, targets)

# Smooth L1 Loss (Huber Loss)
criterion = nn.SmoothL1Loss()
loss = criterion(predictions, targets)
```

### Classification Losses

```python
# Binary Cross-Entropy (for binary classification)
# Input: probabilities (after sigmoid)
criterion = nn.BCELoss()
loss = criterion(predictions, targets)

# BCE with Logits (more stable - use this!)
# Input: raw logits (before sigmoid)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)

# Cross-Entropy (for multi-class)
# Input: raw logits, Targets: class indices
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
```

### Loss Parameters

```python
# Reduction: how to aggregate losses
criterion = nn.MSELoss(reduction='mean')  # Default: average
criterion = nn.MSELoss(reduction='sum')   # Sum all losses
criterion = nn.MSELoss(reduction='none')  # Per-sample losses

# Class weights (for imbalanced data)
weights = torch.tensor([1.0, 2.0, 3.0])  # Higher = more important
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## ⚙️ Optimizers

### Creating Optimizers

```python
# SGD (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam with weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### Optimizer Parameters

```python
# Learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Weight decay (L2 regularization)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Momentum (for SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Betas (for Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

### Using Optimizers

```python
# Basic training step
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute new gradients
optimizer.step()       # Update parameters

# Get current learning rate
current_lr = optimizer.param_groups[0]['lr']
```

---

## 📉 Learning Rate Schedulers

### Creating Schedulers

```python
# Step LR: decay every N epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential LR: decay every epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Reduce on Plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)
```

### Using Schedulers

```python
# For time-based schedulers (Step, Exponential, Cosine)
for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # Call at end of epoch

# For ReduceLROnPlateau
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Pass validation metric
```

---

## 🔄 Complete Training Loop

```python
# Setup
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for data, targets in train_loader:
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for data, targets in val_loader:
            outputs = model(data)
            val_loss = criterion(outputs, targets)
    
    # Update learning rate
    scheduler.step()
```

---

## 💾 Model Saving/Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pt')

# Save complete checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pt'))

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 🎯 Common Patterns

### Gradient Clipping

```python
# Clip gradients by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip gradients by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Usage in training loop
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, targets in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(data)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Custom Loss Function

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # Your custom loss calculation
        loss = torch.mean((predictions - targets) ** 2)
        return loss

# Usage
criterion = CustomLoss()
loss = criterion(predictions, targets)
```

---

## 🔍 Debugging Tips

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")

# Check for NaN/Inf
if torch.isnan(loss) or torch.isinf(loss):
    print("Loss is NaN or Inf!")

# Monitor learning rate
print(f"Current LR: {optimizer.param_groups[0]['lr']}")

# Check model outputs
print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
print(f"Output mean: {outputs.mean():.3f}")
```

---

## 📋 Recommended Defaults

### For Quick Prototyping
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # for classification
# or
criterion = nn.MSELoss()  # for regression
```

### For Production Training
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### For Transformers/NLP
```python
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## 🚨 Common Mistakes to Avoid

```python
# ❌ DON'T: Forget to zero gradients
loss.backward()
optimizer.step()

# ✅ DO: Always zero gradients first
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ❌ DON'T: Apply sigmoid before BCEWithLogitsLoss
probs = torch.sigmoid(logits)
loss = criterion(probs, targets)

# ✅ DO: Pass raw logits
loss = criterion(logits, targets)

# ❌ DON'T: Use .item() during training
loss_value = loss.item()
loss_value.backward()  # Error!

# ✅ DO: Call .item() only for logging
loss.backward()
print(f"Loss: {loss.item()}")

# ❌ DON'T: Forget to set model.train() / model.eval()
model(data)  # May behave unexpectedly

# ✅ DO: Set appropriate mode
model.train()  # For training
model.eval()   # For evaluation
```

---

## 🎓 Learning Rate Selection Guide

| Optimizer | Typical LR | Notes |
|-----------|-----------|-------|
| SGD | 0.01 - 0.1 | Needs momentum |
| Adam | 0.001 (1e-3) | Good default |
| AdamW | 0.0001 - 0.001 | For transformers |
| RMSprop | 0.001 | For RNNs |

**Signs LR is wrong:**
- Too high: Loss oscillates or diverges (NaN)
- Too low: Loss decreases very slowly
- Just right: Steady, consistent decrease

---

## 📚 Further Reading

- [PyTorch Loss Functions Docs](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [PyTorch Optimizers Docs](https://pytorch.org/docs/stable/optim.html)
- [PyTorch LR Scheduler Docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

**💡 Pro Tip**: Start simple (Adam + CrossEntropyLoss), then optimize!
