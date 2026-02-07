# Memory Aware Synapses (MAS)

Memory Aware Synapses (Aljundi et al., 2018) estimates parameter importance based on the sensitivity of the model's output to parameter changes, without requiring task labels.

## Key Idea

MAS measures importance as the magnitude of the gradient of the model's output with respect to each parameter, averaged over the data:

$$\Omega_k = \frac{1}{N} \sum_{i=1}^{N} \left\| \frac{\partial \|f(x_i; \theta)\|_2^2}{\partial \theta_k} \right\|$$

This is task-agnostic: it measures how sensitive the learned function is to each parameter, regardless of the specific loss or labels.

## Regularisation Loss

$$\mathcal{L}_{\text{MAS}} = \mathcal{L}_{\text{task}} + \lambda \sum_k \Omega_k (\theta_k - \theta_k^*)^2$$

## Implementation

```python
import torch
import torch.nn as nn


class MAS:
    """Memory Aware Synapses for continual learning."""
    
    def __init__(self, model, lambda_reg=1.0):
        self.model = model
        self.lambda_reg = lambda_reg
        self.omega = {n: torch.zeros_like(p)
                     for n, p in model.named_parameters() if p.requires_grad}
        self.prev_params = {n: p.data.clone()
                           for n, p in model.named_parameters() if p.requires_grad}
    
    def estimate_importance(self, dataloader, device='cuda'):
        """Estimate parameter importance from unlabeled data."""
        self.model.eval()
        
        importance = {n: torch.zeros_like(p)
                     for n, p in self.model.named_parameters() if p.requires_grad}
        
        num_samples = 0
        for x, _ in dataloader:
            x = x.to(device)
            self.model.zero_grad()
            
            output = self.model(x)
            # Use L2 norm of output as importance signal
            loss = output.norm(2, dim=1).mean()
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad.abs()
            
            num_samples += x.size(0)
        
        # Normalise
        for n in importance:
            importance[n] /= num_samples
        
        # Accumulate with previous tasks
        for n in self.omega:
            self.omega[n] += importance[n]
        
        # Store current parameters
        self.prev_params = {n: p.data.clone()
                           for n, p in self.model.named_parameters() if p.requires_grad}
    
    def penalty(self):
        """Compute MAS regularisation penalty."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            loss += (self.omega[n] * (p - self.prev_params[n]) ** 2).sum()
        return self.lambda_reg * loss
```

## Advantages over EWC

1. **Task-agnostic**: No need for task labels or loss function
2. **Unsupervised importance**: Works with unlabeled data
3. **Simpler computation**: Uses output gradient magnitude, not Fisher information

## References

1. Aljundi, R., et al. (2018). "Memory Aware Synapses: Learning What (Not) to Forget." *ECCV*.
