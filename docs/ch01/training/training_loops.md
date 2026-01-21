# Training Loops

Training a neural network consists of repeatedly applying a **training loop** that updates model parameters to minimize a loss function.

---

## The standard

A canonical training step in PyTorch:

```python
for x, y in dataloader:
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

This loop performs one gradient-based update per batch.

---

## Epochs and batches

- **Batch**: a subset of data used for one update
- **Epoch**: one full pass over the dataset

```python
for epoch in range(num_epochs):
    train_one_epoch()
```

---

## Training vs

Some modules behave differently during training:

```python
model.train()
model.eval()
```

Examples:
- Dropout
- Batch Normalization

---

## Monitoring training

Typical quantities to monitor:
- training loss,
- validation loss,
- gradient norms.

```python
loss.item()
```

---

## Financial

Training loops are used for:
- fitting surrogate pricing models,
- calibrating neural networks to market data,
- learning hedging strategies.

---

## Key takeaways

- Training loops follow a fixed pattern.
- Separate training and evaluation modes.
- Monitoring prevents silent failures.
