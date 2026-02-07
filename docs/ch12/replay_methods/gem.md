# Gradient Episodic Memory (GEM)

GEM (Lopez-Paz & Ranzato, 2017) uses stored examples not for replay, but as constraints on gradient updates. The gradient on the current task is projected so that it does not increase the loss on any previous task.

## Key Idea

When updating on task $t$, GEM ensures:

$$\langle g_t, g_k \rangle \geq 0, \quad \forall k < t$$

where $g_t$ is the gradient on the current task and $g_k$ is the gradient computed on stored examples from task $k$. If a proposed gradient would increase any previous task's loss, it is projected to the closest gradient satisfying all constraints.

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize


class GEM:
    """Gradient Episodic Memory for continual learning."""
    
    def __init__(self, model, memory_per_task=256):
        self.model = model
        self.memory_per_task = memory_per_task
        self.memories = {}  # task_id -> (x, y)
        self.task_grads = {}
    
    def store_memory(self, task_id, dataloader, device='cuda'):
        """Store examples for gradient constraints."""
        xs, ys = [], []
        for x, y in dataloader:
            xs.append(x)
            ys.append(y)
            if sum(len(x_) for x_ in xs) >= self.memory_per_task:
                break
        
        xs = torch.cat(xs)[:self.memory_per_task]
        ys = torch.cat(ys)[:self.memory_per_task]
        self.memories[task_id] = (xs.to(device), ys.to(device))
    
    def compute_task_gradient(self, task_id, device='cuda'):
        """Compute gradient on stored examples from a previous task."""
        x, y = self.memories[task_id]
        self.model.zero_grad()
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        grad = torch.cat([p.grad.flatten() for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None])
        return grad
    
    def project_gradient(self, current_grad, device='cuda'):
        """Project gradient to satisfy non-forgetting constraints."""
        if not self.memories:
            return current_grad
        
        # Compute gradients on all previous tasks
        ref_grads = []
        for task_id in sorted(self.memories.keys()):
            ref_grad = self.compute_task_gradient(task_id, device)
            ref_grads.append(ref_grad)
        
        ref_grads = torch.stack(ref_grads)  # (num_tasks, d)
        
        # Check if any constraint is violated
        dots = torch.mv(ref_grads, current_grad)
        
        if (dots >= 0).all():
            return current_grad  # No projection needed
        
        # Solve QP to find closest gradient satisfying constraints
        # min ||g - g_current||^2 s.t. g^T g_k >= 0 for all k
        projected = self._solve_qp(current_grad, ref_grads)
        
        return projected
    
    def _solve_qp(self, g, G):
        """Solve the QP for gradient projection."""
        # Simplified: project onto intersection of half-spaces
        g_np = g.cpu().numpy()
        G_np = G.cpu().numpy()
        
        n_tasks = G_np.shape[0]
        
        def objective(v):
            return 0.5 * np.sum((v - g_np) ** 2)
        
        def grad_objective(v):
            return v - g_np
        
        constraints = [{'type': 'ineq', 'fun': lambda v, i=i: np.dot(v, G_np[i])}
                      for i in range(n_tasks)]
        
        result = minimize(objective, g_np, jac=grad_objective,
                         constraints=constraints, method='SLSQP')
        
        return torch.tensor(result.x, dtype=g.dtype, device=g.device)
```

## A-GEM: Averaged GEM

A-GEM (Chaudhry et al., 2019) simplifies GEM by using a single averaged gradient constraint instead of per-task constraints, making it much more efficient:

```python
def agem_project(current_grad, ref_grad):
    """A-GEM: project if dot product is negative."""
    dot = torch.dot(current_grad, ref_grad)
    if dot >= 0:
        return current_grad
    else:
        return current_grad - (dot / (torch.dot(ref_grad, ref_grad) + 1e-8)) * ref_grad
```

## References

1. Lopez-Paz, D., & Ranzato, M. (2017). "Gradient Episodic Memory for Continual Learning." *NeurIPS*.
2. Chaudhry, A., et al. (2019). "Efficient Lifelong Learning with A-GEM." *ICLR*.
