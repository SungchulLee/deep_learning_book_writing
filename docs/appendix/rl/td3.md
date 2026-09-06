# TD3

TD3 was introduced in the 2018 paper "Addressing Function Approximation Error in Actor-Critic Methods." 1) Twin Q networks (min of two critics)   2) Target policy smoothing (add noise to target action)   3) Delayed actor updates.

This implementation provides a concise, educational reference for TD3. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
TD3 - Twin Delayed DDPG
Paper: "Addressing Function Approximation Error in Actor-Critic Methods" (2018)
Authors: Scott Fujimoto, Herke van Hoof, David Meger
Key ideas:
  1) Twin Q networks (min of two critics)
  2) Target policy smoothing (add noise to target action)
  3) Delayed actor updates

File: appendix/rl/td3.py
Note: Educational reference (core target computation + networks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class Actor(nn.Module):
    """Deterministic policy a = pi(s)."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # assume actions scaled to [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Q(s,a) critic."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x).squeeze(1)


def td3_target(q1_tgt, q2_tgt, actor_tgt, s2, r, done, gamma=0.99, noise_std=0.2, noise_clip=0.5):
    """
    TD3 target:
      a' = actor_tgt(s') + clipped_noise
      y  = r + gamma*(1-done) * min(Q1_tgt(s',a'), Q2_tgt(s',a'))

    Target policy smoothing reduces overestimation from sharp Q peaks.
    """
    with torch.no_grad():
        a2 = actor_tgt(s2)

        # Add clipped Gaussian noise
        noise = torch.randn_like(a2) * noise_std
        noise = noise.clamp(-noise_clip, noise_clip)
        a2 = (a2 + noise).clamp(-1.0, 1.0)

        q1v = q1_tgt(s2, a2)
        q2v = q2_tgt(s2, a2)
        qmin = torch.min(q1v, q2v)

        y = r + gamma * (1.0 - done) * qmin
    return y


if __name__ == "__main__":
    pass```

## Discussion

The implementation defines 2 classes (`Actor`, `Critic`) that work together to form the complete reinforcement learning architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `Actor` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `Actor` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = Actor(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
