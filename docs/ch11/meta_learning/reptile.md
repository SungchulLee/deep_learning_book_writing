# Reptile

Reptile (Nichol et al., 2018) is a first-order meta-learning algorithm that provides a simpler alternative to MAML. While MAML computes second-order derivatives through the inner optimisation, Reptile achieves competitive results using only standard gradient descent.

## Algorithm

Reptile performs the following steps:

1. Sample a task $\mathcal{T}$
2. Run $k$ steps of SGD on $\mathcal{T}$ starting from parameters $\theta$, obtaining $\tilde{\theta}$
3. Update: $\theta \leftarrow \theta + \epsilon(\tilde{\theta} - \theta)$

The meta-gradient is simply the direction from the current parameters to the task-adapted parameters.

$$\theta \leftarrow \theta + \epsilon \cdot \frac{1}{n} \sum_{i=1}^{n} (\tilde{\theta}_i - \theta)$$

## Why It Works

Reptile's update direction approximates a combination of:

1. **Task-specific gradient**: The direction that improves performance on each task
2. **Inter-task gradient alignment**: The direction where task gradients agree

This second component is what enables meta-learning—Reptile finds initialisations where a few gradient steps can quickly specialise to any task.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def reptile_train(model, task_sampler, meta_lr=1.0, inner_lr=0.01,
                  inner_steps=5, num_iterations=10000, device='cuda'):
    """
    Reptile meta-learning algorithm.
    
    Args:
        model: Base model
        task_sampler: Yields (support_x, support_y, query_x, query_y) per task
        meta_lr: Outer learning rate (epsilon)
        inner_lr: Inner learning rate for task adaptation
        inner_steps: Number of SGD steps per task
        num_iterations: Total meta-training iterations
    """
    model = model.to(device)
    
    for iteration in range(num_iterations):
        # Save original parameters
        original_params = {name: param.clone()
                          for name, param in model.named_parameters()}
        
        # Sample a task
        support_x, support_y, query_x, query_y = next(task_sampler)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        
        # Inner loop: k steps of SGD on the task
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
        
        for step in range(inner_steps):
            logits = model(support_x)
            loss = F.cross_entropy(logits, support_y)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        # Outer update: move toward task-adapted parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_params[name] + meta_lr * (param.data - original_params[name])
        
        # Logging
        if iteration % 500 == 0:
            # Evaluate on query set
            model.eval()
            with torch.no_grad():
                query_x = query_x.to(device)
                query_y = query_y.to(device)
                preds = model(query_x).argmax(1)
                acc = (preds == query_y).float().mean().item()
            model.train()
            print(f"Iteration {iteration}: Query acc = {acc:.4f}")
    
    return model


def reptile_adapt(model, support_x, support_y, inner_lr=0.01, inner_steps=10):
    """Adapt model to a new task at test time."""
    adapted_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
    
    adapted_model.train()
    for _ in range(inner_steps):
        logits = adapted_model(support_x)
        loss = F.cross_entropy(logits, support_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    adapted_model.eval()
    return adapted_model
```

## Reptile vs MAML

| Aspect | MAML | Reptile |
|--------|------|---------|
| Gradient order | Second-order (or first-order approx) | First-order only |
| Implementation | Complex (inner/outer loop) | Simple (SGD + interpolation) |
| Memory | High (computation graph) | Low |
| Performance | Slightly higher | Competitive |
| Inner optimiser | Typically 1-5 steps | Typically 5-50 steps |

## References

1. Nichol, A., Achiam, J., & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms." arXiv.
