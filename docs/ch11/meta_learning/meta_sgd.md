# Meta-SGD

Meta-SGD (Li et al., 2017) extends MAML by learning not only the model initialisation but also the **per-parameter learning rates** and **update directions** for task adaptation.

## Motivation

MAML uses a single scalar learning rate $\alpha$ for all parameters during the inner loop. Meta-SGD recognises that different parameters may need different adaptation rates:

- Parameters encoding general features should adapt slowly
- Parameters near the output should adapt quickly
- Some parameters may need larger or smaller steps

## Algorithm

Meta-SGD jointly meta-learns:

1. **Initial parameters** $\theta_0$ (like MAML)
2. **Per-parameter learning rates** $\alpha = \{\alpha_1, \alpha_2, ..., \alpha_d\}$

Inner loop update:

$$\theta_i' = \theta_i - \alpha_i \odot \nabla_{\theta_i} \mathcal{L}_{\mathcal{T}}(\theta)$$

where $\odot$ denotes element-wise multiplication.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaSGD(nn.Module):
    """
    Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.
    
    Learns per-parameter learning rates alongside model initialisation.
    """
    
    def __init__(self, base_model, init_lr=0.01):
        super().__init__()
        self.model = base_model
        
        # Create learnable per-parameter learning rates
        self.task_lr = nn.ParameterDict()
        for name, param in self.model.named_parameters():
            # Initialise learning rates (same shape as parameters)
            lr_param = nn.Parameter(torch.ones_like(param) * init_lr)
            # Use sanitised name for ParameterDict
            safe_name = name.replace('.', '_')
            self.task_lr[safe_name] = lr_param
    
    def adapt(self, support_x, support_y):
        """
        Single-step adaptation using learned per-parameter LRs.
        
        Returns adapted model parameters (as a dict).
        """
        logits = self.model(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        adapted_params = {}
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            safe_name = name.replace('.', '_')
            lr = self.task_lr[safe_name]
            adapted_params[name] = param - lr * grad
        
        return adapted_params
    
    def forward_with_params(self, x, params):
        """Forward pass using adapted parameters (functional forward)."""
        # This requires a functional implementation of the model
        # For simplicity, here's the concept:
        raise NotImplementedError("Requires functional model implementation")
    
    def meta_train_step(self, support_x, support_y, query_x, query_y):
        """
        One meta-training step.
        
        1. Adapt to support set using learned LRs
        2. Evaluate on query set
        3. Return meta-loss
        """
        adapted_params = self.adapt(support_x, support_y)
        
        # Evaluate with adapted parameters on query set
        # (simplified - full version uses functional forward)
        query_logits = self.model(query_x)  # Placeholder
        meta_loss = F.cross_entropy(query_logits, query_y)
        
        return meta_loss


def train_meta_sgd(model, task_sampler, meta_lr=1e-3, num_iterations=10000, device='cuda'):
    """Train Meta-SGD."""
    meta_sgd = MetaSGD(model).to(device)
    
    # Meta-optimiser updates both model params and learning rates
    meta_optimizer = torch.optim.Adam(meta_sgd.parameters(), lr=meta_lr)
    
    for iteration in range(num_iterations):
        support_x, support_y, query_x, query_y = next(task_sampler)
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        meta_loss = meta_sgd.meta_train_step(support_x, support_y, query_x, query_y)
        
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        if iteration % 500 == 0:
            print(f"Iteration {iteration}: Meta-loss = {meta_loss.item():.4f}")
    
    return meta_sgd
```

## Comparison

| Method | Learned | Inner loop | Complexity |
|--------|---------|-----------|------------|
| MAML | Initialisation $\theta_0$ | Fixed scalar $\alpha$ | High |
| Meta-SGD | $\theta_0$ + per-param $\alpha_i$ | Learned $\alpha_i$ | Higher |
| Reptile | Initialisation $\theta_0$ | Fixed scalar $\alpha$ | Low |

Meta-SGD typically outperforms MAML slightly but at the cost of more meta-parameters (doubles the number of learnable values).

## References

1. Li, Z., et al. (2017). "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning." arXiv.
