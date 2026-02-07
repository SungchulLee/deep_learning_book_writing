# Training Strategies for Deep Ensembles

## Overview

Effective ensemble training requires diversity among members while maintaining individual model quality. This section covers training strategies that promote both objectives.

## Independent Training

The simplest and most effective approach: train each member from scratch with different random initialization.

```python
import torch
import torch.nn as nn
from typing import List, Dict


def train_ensemble_member(
    model: nn.Module,
    train_loader,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = None,
    device: str = 'cpu'
) -> Dict[str, List[float]]:
    if seed is not None:
        torch.manual_seed(seed)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        
        scheduler.step()
        history['loss'].append(total_loss / total)
        history['accuracy'].append(correct / total)
    
    return history
```

## Negative Log-Likelihood Loss for Regression

Lakshminarayanan et al. (2017) recommend training with proper scoring rules. For regression, each member predicts mean and variance:

$$\mathcal{L}_m = \frac{1}{N}\sum_{i=1}^N \left[\frac{(y_i - \mu_m(\mathbf{x}_i))^2}{2\sigma_m^2(\mathbf{x}_i)} + \frac{1}{2}\log \sigma_m^2(\mathbf{x}_i)\right]$$

```python
class GaussianNLLMember(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.features = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, 1)
        self.logvar_head = nn.Linear(prev, 1)
    
    def forward(self, x):
        h = self.features(x)
        return self.mean_head(h), self.logvar_head(h)
    
    def loss(self, x, y):
        mean, log_var = self.forward(x)
        var = torch.exp(log_var)
        return (0.5 * (y - mean)**2 / var + 0.5 * log_var).mean()
```

## Snapshot Ensembles

Collect models along the SGD trajectory using cyclic learning rates, eliminating the need to train M models from scratch:

```python
import numpy as np

def train_snapshot_ensemble(
    model, train_loader, n_snapshots=5, epochs_per_cycle=50,
    lr_max=0.1, lr_min=1e-4
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    snapshots = []
    total_epochs = n_snapshots * epochs_per_cycle
    
    for epoch in range(total_epochs):
        cycle_epoch = epoch % epochs_per_cycle
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1 + np.cos(np.pi * cycle_epoch / epochs_per_cycle))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
        
        if (epoch + 1) % epochs_per_cycle == 0:
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            snapshots.append(snapshot)
    
    return snapshots
```

## Bootstrapping

Each member trains on a bootstrap sample (sampling with replacement):

```python
from torch.utils.data import Subset

def create_bootstrap_loaders(dataset, n_members, batch_size=64):
    n = len(dataset)
    loaders = []
    for _ in range(n_members):
        indices = np.random.choice(n, size=n, replace=True)
        subset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders
```

## Key Takeaways

!!! success "Summary"
    1. **Independent training** with different seeds is the most robust approach
    2. **NLL loss** enables each member to estimate its own aleatoric uncertainty
    3. **Snapshot ensembles** reduce cost to a single training run
    4. **Bootstrapping** adds data-level diversity

## References

- Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
- Huang, G., et al. (2017). "Snapshot Ensembles: Train 1, Get M for Free." ICLR.
