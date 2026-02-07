# Training Loop Basics

## Overview

Training a neural network consists of repeatedly applying a **training loop** that updates model parameters to minimize a loss function. The loop follows a fixed pattern that, once internalized, serves as the scaffold for every deep learning experiment.

## The Standard Training Loop

A canonical training step in PyTorch:

```python
for x, y in dataloader:
    optimizer.zero_grad()       # 1. Clear accumulated gradients
    pred = model(x)             # 2. Forward pass
    loss = loss_fn(pred, y)     # 3. Compute loss
    loss.backward()             # 4. Backward pass (compute gradients)
    optimizer.step()            # 5. Update parameters
```

This five-step sequence performs one gradient-based update per batch. Each step is essential:

1. **`optimizer.zero_grad()`** — Gradients accumulate by default in PyTorch. Without zeroing, gradients from the previous batch contaminate the current update.
2. **Forward pass** — The model maps inputs to predictions using current parameters.
3. **Loss computation** — A scalar loss quantifies the discrepancy between predictions and targets.
4. **`loss.backward()`** — Autograd computes $\partial \mathcal{L} / \partial \theta$ for every parameter $\theta$ with `requires_grad=True`.
5. **`optimizer.step()`** — The optimizer applies the update rule (e.g., SGD, Adam) using the computed gradients.

## Epochs and Batches

- **Batch**: a subset of data used for one parameter update.
- **Epoch**: one full pass over the entire dataset.

```python
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
```

The number of parameter updates per epoch equals $\lceil N / B \rceil$ where $N$ is the dataset size and $B$ is the batch size.

## Training vs. Evaluation Mode

Some modules behave differently during training and evaluation:

```python
model.train()   # Enable training behavior
model.eval()    # Enable evaluation behavior
```

Modules affected include:

- **Dropout**: Active during `train()`, disabled during `eval()`.
- **Batch Normalization**: Uses batch statistics during `train()`, running statistics during `eval()`.

Forgetting to switch modes is a common source of subtle bugs—models that perform well during training but poorly at test time (Dropout still active) or vice versa.

## Complete Training Loop

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# Full training run
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                            optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}")
```

## Monitoring Training

Typical quantities to monitor during training:

- **Training loss**: Should decrease steadily. Sudden spikes indicate learning rate issues or data problems.
- **Validation loss**: Divergence from training loss signals overfitting.
- **Gradient norms**: Exploding or vanishing gradients indicate optimization issues.

```python
# Extract scalar loss value
current_loss = loss.item()

# Monitor gradient norms
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
```

## Quantitative Finance Application

Training loops are used throughout quantitative finance for:

- **Surrogate pricing models**: Fitting neural networks to approximate computationally expensive pricing functions (e.g., Monte Carlo pricing of exotic derivatives).
- **Neural calibration**: Calibrating stochastic volatility models (Heston, SABR) to market-observed option surfaces by minimizing calibration error.
- **Learned hedging strategies**: Training networks to output hedge ratios that minimize hedging P&L variance, framed as a custom loss minimization problem.

These applications often require custom loss functions and careful monitoring of domain-specific metrics alongside standard training loss.

## Key Takeaways

- The training loop follows a fixed five-step pattern: zero gradients, forward pass, compute loss, backward pass, optimizer step.
- Always switch between `model.train()` and `model.eval()` appropriately.
- Monitor training loss, validation loss, and gradient norms to detect common failure modes.
- Monitoring prevents silent failures—a decreasing training loss alone does not guarantee a useful model.
