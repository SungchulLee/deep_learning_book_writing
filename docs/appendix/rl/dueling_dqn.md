# Dueling DQN

Dueling DQN was introduced in the 2016 paper "Dueling Network Architectures for Deep Reinforcement Learning." - Learn V(s) and A(s,a) separately, then combine to Q(s,a)   - Helps when many actions have similar value.

This implementation provides a concise, educational reference for Dueling DQN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Dueling DQN - Separate value and advantage streams
Paper: "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
Authors: Ziyu Wang et al.
Key idea:
  - Learn V(s) and A(s,a) separately, then combine to Q(s,a)
  - Helps when many actions have similar value

Combination:
  Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

File: appendix/rl/dueling_dqn.py
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class DuelingQNetwork(nn.Module):
    """
    Dueling network:
      shared trunk -> value head + advantage head
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        # Value stream outputs a single scalar V(s)
        self.value = nn.Linear(hidden, 1)

        # Advantage stream outputs A(s,a) for all actions
        self.adv = nn.Linear(hidden, num_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)

        V = self.value(h)           # (B, 1)
        A = self.adv(h)             # (B, A)

        # Center advantages to keep Q identifiable (otherwise V/A not unique)
        A_centered = A - A.mean(dim=1, keepdim=True)

        Q = V + A_centered          # (B, A)
        return Q


if __name__ == "__main__":
    net = DuelingQNetwork(obs_dim=8, num_actions=4)
    x = torch.randn(3, 8)
    q = net(x)
    print("Q:", q.shape)  # (3, 4)```

## Discussion

The `DuelingQNetwork` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `DuelingQNetwork` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Extend `DuelingQNetwork` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = DuelingQNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
