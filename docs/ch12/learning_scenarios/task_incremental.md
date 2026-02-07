# Task-Incremental Learning

Task-incremental learning (Task-IL) is the simplest continual learning scenario. The model receives a task identifier at both training and test time, enabling it to use task-specific output heads.

## Problem Setting

The model learns a sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, ..., \mathcal{T}_T$. At test time, the task identity is known, so the model selects the appropriate output head:

$$\hat{y} = \arg\max_{c \in \mathcal{C}_t} f_t(x; \theta_{\text{shared}}, \theta_t)$$

where $\theta_{\text{shared}}$ are shared backbone parameters and $\theta_t$ are task-specific head parameters.

## Multi-Head Architecture

```python
import torch
import torch.nn as nn


class TaskIncrementalModel(nn.Module):
    """Multi-head model for task-incremental learning."""
    
    def __init__(self, backbone, feature_dim=512):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.heads = nn.ModuleDict()
    
    def add_task(self, task_id, num_classes):
        """Add a new task-specific head."""
        self.heads[str(task_id)] = nn.Linear(self.feature_dim, num_classes)
    
    def forward(self, x, task_id):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.heads[str(task_id)](features)
    
    def predict(self, x, task_id):
        logits = self.forward(x, task_id)
        return logits.argmax(dim=1)
```

## Training Protocol

```python
def train_task_incremental(model, task_id, train_loader, num_classes,
                           epochs=10, lr=0.001, device='cuda',
                           regularizer=None):
    """Train on a new task while optionally regularising against forgetting."""
    model.add_task(task_id, num_classes)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task_id)
            loss = criterion(logits, y)
            
            if regularizer is not None:
                loss += regularizer.penalty()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Why Task-IL Is Easier

Task-IL avoids the hardest aspect of continual learning: distinguishing between classes from different tasks. Since the task ID is given, each head only discriminates within its own task's classes.

| Scenario | Task ID at test | Output space | Difficulty |
|----------|----------------|-------------|-----------|
| Task-IL | ✅ Known | Per-task heads | Easiest |
| Domain-IL | ✅ Known | Shared head | Moderate |
| Class-IL | ❌ Unknown | Unified over all classes | Hardest |

## References

1. van de Ven, G.M., & Tolias, A.S. (2019). "Three Scenarios for Continual Learning." arXiv.
