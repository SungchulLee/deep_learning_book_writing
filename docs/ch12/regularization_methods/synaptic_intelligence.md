# Synaptic Intelligence (SI)

Synaptic Intelligence (Zenke et al., 2017) protects important parameters by tracking their contribution to loss reduction during training, providing an online alternative to EWC's Fisher information estimation.

## Key Idea

While EWC estimates parameter importance after training (via the Fisher matrix), SI accumulates importance **during training** by measuring how much each parameter contributed to the loss decrease:

$$\omega_k = \sum_{\nu} \frac{(\Delta_k^\nu)^2}{\left(\Delta \theta_k^\nu\right)^2 + \xi}$$

where $\Delta_k^\nu$ is the contribution of parameter $k$ to the loss change during training on task $\nu$, and $\xi$ is a damping term.

## Regularisation Loss

$$\mathcal{L}_{\text{SI}} = \mathcal{L}_{\text{task}} + c \sum_k \omega_k (\theta_k - \theta_k^*)^2$$

where $\theta_k^*$ are the parameters after the previous task and $c$ is the regularisation strength.

## Implementation

```python
import torch
import torch.nn as nn


class SynapticIntelligence:
    """
    Synaptic Intelligence for continual learning.
    
    Tracks online parameter importance during training.
    """
    
    def __init__(self, model, c=1.0, xi=0.1):
        self.model = model
        self.c = c
        self.xi = xi
        
        # Importance weights
        self.omega = {n: torch.zeros_like(p)
                     for n, p in model.named_parameters() if p.requires_grad}
        
        # Running sum of per-parameter path integral
        self.running_sum = {n: torch.zeros_like(p)
                           for n, p in model.named_parameters() if p.requires_grad}
        
        # Parameters at start of current task
        self.prev_params = {n: p.data.clone()
                           for n, p in model.named_parameters() if p.requires_grad}
        
        # Parameters at previous step (for computing deltas)
        self.step_params = {n: p.data.clone()
                           for n, p in model.named_parameters() if p.requires_grad}
    
    def update_running_sum(self):
        """Call after each optimiser step to track parameter contributions."""
        for n, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            
            delta = p.data - self.step_params[n]
            self.running_sum[n] += -p.grad.data * delta
            self.step_params[n] = p.data.clone()
    
    def update_omega(self):
        """Call at end of task to consolidate importance weights."""
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            
            delta_theta = p.data - self.prev_params[n]
            self.omega[n] += self.running_sum[n] / (delta_theta ** 2 + self.xi)
            
            # Reset for next task
            self.running_sum[n].zero_()
            self.prev_params[n] = p.data.clone()
            self.step_params[n] = p.data.clone()
    
    def penalty(self):
        """Compute SI regularisation penalty."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            loss += (self.omega[n] * (p - self.prev_params[n]) ** 2).sum()
        return self.c * loss
```

## SI vs EWC

| Aspect | EWC | SI |
|--------|-----|-----|
| Importance estimation | Fisher matrix (post-task) | Path integral (online) |
| Computation | Requires separate pass | Accumulated during training |
| Memory | Stores Fisher diagonal | Stores running sums |
| Approximation | Second-order (Fisher) | First-order (gradient × displacement) |

## References

1. Zenke, F., Poole, B., & Ganguli, S. (2017). "Continual Learning Through Synaptic Intelligence." *ICML*.
