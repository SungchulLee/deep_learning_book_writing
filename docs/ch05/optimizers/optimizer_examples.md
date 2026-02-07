# Practical Examples

## Overview

This section provides complete, runnable examples demonstrating optimizer usage in common scenarios.

## Example 1: Image Classification with SGD

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4,
                            nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

## Example 2: Transformer Training with AdamW

```python
model = TransformerModel(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                              betas=(0.9, 0.98), weight_decay=0.01)

# Warmup + cosine decay
warmup_steps = 4000
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    anneal_strategy='cos'
)

for step, (x, y) in enumerate(train_loader):
    optimizer.zero_grad()
    loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

## Example 3: Differential Learning Rates for Fine-Tuning

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)

optimizer = torch.optim.AdamW([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=0.01)

# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False
for param in model.layer3.parameters():
    param.requires_grad = False
```

## Example 4: Neural Calibration with L-BFGS

```python
model = HestonNeuralCalibrator(...)
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0,
                               max_iter=20, history_size=50,
                               line_search_fn='strong_wolfe')

# Full-batch training (small calibration dataset)
market_prices = torch.tensor(observed_prices)

for epoch in range(100):
    def closure():
        optimizer.zero_grad()
        model_prices = model(strikes, maturities, spots)
        loss = nn.MSELoss()(model_prices, market_prices)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    if loss.item() < 1e-8:
        break
```

## Example 5: Gradient Accumulation with Adam

```python
accumulation_steps = 8
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
for i, (x, y) in enumerate(train_loader):
    loss = loss_fn(model(x), y) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Key Takeaways

- Match the optimizer-scheduler pair to the architecture: SGD + cosine for CNNs, AdamW + warmup for transformers.
- Differential learning rates are essential for fine-tuning pretrained models.
- L-BFGS is effective for small-scale optimization problems like neural calibration.
- Gradient accumulation enables large effective batch sizes with any optimizer.
