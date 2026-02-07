# Validation Loop

## Overview

The validation loop evaluates model performance on held-out data after each training epoch (or at regular intervals). Unlike the training loop, the validation loop does not compute gradients or update parameters—its sole purpose is to estimate generalization performance.

## Standard Validation Loop

```python
@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()  # Critical: disable dropout, use running BN stats
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
```

Two critical elements distinguish the validation loop from training:

1. **`model.eval()`** — Switches layers like Dropout and BatchNorm to evaluation mode.
2. **`torch.no_grad()`** — Disables gradient computation, reducing memory usage and accelerating inference.

## Integration with Training Loop

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                            optimizer, loss_fn, device)
    val_loss, val_acc = validate(model, val_loader, loss_fn, device)

    print(f"Epoch {epoch+1}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Track best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

    scheduler.step(val_loss)  # For ReduceLROnPlateau
```

## Validation Frequency

Validating every epoch is standard, but for large datasets or expensive models, less frequent validation may be appropriate:

```python
for epoch in range(num_epochs):
    train_one_epoch(...)

    if (epoch + 1) % val_every == 0:
        val_loss, val_acc = validate(...)
```

Alternatively, validate every $k$ batches within an epoch for more granular feedback during long epochs.

## Regression Validation

For regression tasks, track different metrics:

```python
@torch.no_grad()
def validate_regression(model, loader, device):
    model.eval()
    predictions, targets = [], []

    for x, y in loader:
        x = x.to(device)
        pred = model(x)
        predictions.append(pred.cpu())
        targets.append(y)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    return {'mse': mse, 'mae': mae}
```

## Common Pitfalls

**Forgetting `model.eval()`**: Dropout remains active, artificially degrading validation performance. BatchNorm uses batch statistics instead of accumulated running statistics, introducing batch-dependent variance.

**Forgetting `torch.no_grad()`**: Gradients are computed and stored unnecessarily, wasting memory and computation. For large models this can cause out-of-memory errors during validation.

**Data leakage**: If any part of the validation data was used during training (e.g., computing normalization statistics), validation metrics are biased. Normalization statistics should be computed from the training set only.

## Key Takeaways

- The validation loop evaluates generalization without updating parameters.
- Always use `model.eval()` and `torch.no_grad()` during validation.
- Track validation loss alongside training loss to detect overfitting.
- Save the model state when validation performance improves (early stopping criterion).
