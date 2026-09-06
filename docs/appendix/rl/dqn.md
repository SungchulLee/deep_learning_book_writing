# DQN

DQN was introduced in the 2015 paper "Human-level control through deep reinforcement learning." - Approximate Q(s,a) with a neural network   - Train with TD target using a *target network*   - Use experience replay to break correlation.

This implementation provides a concise, educational reference for DQN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
DQN - Deep Q-Network
Paper: "Human-level control through deep reinforcement learning" (2015)
Authors: Volodymyr Mnih et al.
Key idea:
  - Approximate Q(s,a) with a neural network
  - Train with TD target using a *target network*
  - Use experience replay to break correlation

File: appendix/rl/dqn.py
Note: Educational reference: model + replay + TD loss computation (no full env loop).
"""

from dataclasses import dataclass
import random
from collections import deque

# ========================================================================
# Main
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Simple MLP Q(s,a) approximator for discrete actions."""
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, obs_dim)
        return self.net(obs)  # (B, num_actions)


@dataclass
class Transition:
    """One experience tuple stored in replay buffer."""
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        # Stack fields into batch tensors
        s = torch.stack([b.s for b in batch], dim=0)
        a = torch.stack([b.a for b in batch], dim=0)
        r = torch.stack([b.r for b in batch], dim=0)
        s2 = torch.stack([b.s2 for b in batch], dim=0)
        done = torch.stack([b.done for b in batch], dim=0)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buf)


def dqn_td_loss(q_net: nn.Module, target_net: nn.Module, batch, gamma: float = 0.99):
    """
    Compute DQN TD loss.

    For each transition (s,a,r,s',done):
      target = r + gamma * (1-done) * max_a' Q_target(s', a')
      loss = MSE( Q(s,a), target )

    Note:
      - a is discrete action index (shape: (B,))
      - done is 1 if terminal else 0
    """
    s, a, r, s2, done = batch

    # Current Q-values for all actions: (B, A)
    q_values = q_net(s)

    # Select Q(s,a) using gather:
    # a must be shape (B,1) for gather on dim=1
    q_sa = q_values.gather(1, a.long().unsqueeze(1)).squeeze(1)  # (B,)

    # Compute target using target network (no grad)
    with torch.no_grad():
        q_next = target_net(s2)                   # (B, A)
        max_q_next = q_next.max(dim=1).values     # (B,)
        target = r + gamma * (1.0 - done) * max_q_next

    loss = F.mse_loss(q_sa, target)
    return loss


if __name__ == "__main__":
    # Toy smoke test (no environment)
    obs_dim, num_actions = 8, 4
    q = QNetwork(obs_dim, num_actions)
    tgt = QNetwork(obs_dim, num_actions)

    # Fake batch
    B = 5
    s = torch.randn(B, obs_dim)
    a = torch.randint(0, num_actions, (B,))
    r = torch.randn(B)
    s2 = torch.randn(B, obs_dim)
    done = torch.randint(0, 2, (B,), dtype=torch.float32)

    loss = dqn_td_loss(q, tgt, (s, a, r, s2, done))
    print("loss:", float(loss))```

## Discussion

The implementation defines 3 classes (`QNetwork`, `Transition`, `ReplayBuffer`) that work together to form the complete reinforcement learning architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `QNetwork` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `QNetwork` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = QNetwork(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
